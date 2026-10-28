"""Dataset preparation, training, validation, and testing for CoReMatch."""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr
from torch_geometric.data import Batch
from torch_geometric.datasets import GEDDataset
from tqdm import tqdm, trange

from model import CoReMatch
from reporting import format_metrics, truncate_mse
from metrics import (
    calculate_prec_at_k,
    calculate_ranking_correlation,
)


class ConstantNodeFeature:
    """Assign the fixed feature x_i = [1] to every unlabeled node."""

    def __call__(self, data):
        data.x = torch.ones(
            (data.num_nodes, 1),
            dtype=torch.float,
        )
        return data


def load_checkpoint(path, device):
    """Load a model state dictionary."""
    return torch.load(path, map_location=device, weights_only=True)


class Trainer:
    """Train on 75%, validate on 25%, then test against all official train."""

    def __init__(self, args):
        self.args = args
        self.device = self.select_device()
        self.process_dataset()
        self.model = CoReMatch(args, self.number_of_labels).to(self.device)
        self._evaluation_batch_cache = {}
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}", flush=True)

    def select_device(self):
        if self.args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu for CPU execution.")
        return torch.device(self.args.device)

    @staticmethod
    def materialize_dataset(dataset):
        """Apply static transforms once and retain reusable CPU graph objects."""
        return [dataset[index] for index in range(len(dataset))]

    def process_dataset(self):
        official_train = GEDDataset(
            self.args.dataset_root,
            self.args.dataset,
            train=True,
        )
        official_test = GEDDataset(
            self.args.dataset_root,
            self.args.dataset,
            train=False,
        )
        if not len(official_train) or not len(official_test):
            raise ValueError("The official train/test dataset is empty.")
        self.nged_matrix = getattr(official_train, "norm_ged", None)
        if self.nged_matrix is None:
            raise ValueError("GEDDataset.norm_ged is required.")

        if official_train[0].x is None:
            if self.args.dataset not in {"LINUX", "IMDBMulti"}:
                raise ValueError(
                    f"No fixed node-feature rule for unlabeled {self.args.dataset}."
                )
            official_train.transform = ConstantNodeFeature()
            official_test.transform = ConstantNodeFeature()

        official_training_graphs = self.materialize_dataset(official_train)
        official_testing_graphs = self.materialize_dataset(official_test)
        training_count = int(len(official_training_graphs) * 0.75)
        self.training_graphs = official_training_graphs[:training_count]
        self.validation_graphs = official_training_graphs[training_count:]
        self.testing_graphs = official_testing_graphs
        self.full_training_database = official_training_graphs

        sample = self.training_graphs[0]
        self.number_of_labels = sample.x.size(1)
        print(
            "Dataset {}: train={}, validation={}, test={}, "
            "full-training-database={}".format(
                self.args.dataset,
                len(self.training_graphs),
                len(self.validation_graphs),
                len(self.testing_graphs),
                len(self.full_training_database),
            ),
            flush=True,
        )

    @staticmethod
    def graph_indices(batch):
        if not hasattr(batch, "i"):
            raise AttributeError("GEDDataset graph index attribute 'i' is missing.")
        return batch.i.view(-1).long().cpu()

    def get_target_similarity_indices(self, index_1, index_2):
        index_1 = index_1.view(-1).long().cpu()
        index_2 = index_2.view(-1).long().cpu()
        if torch.is_tensor(self.nged_matrix):
            norm_ged = self.nged_matrix[index_1, index_2].float()
        else:
            norm_ged = torch.tensor(
                self.nged_matrix[index_1.numpy(), index_2.numpy()],
                dtype=torch.float32,
            )
        return torch.exp(-norm_ged).to(self.device)

    def get_target_similarity_batch(self, graph_1, graph_2):
        return self.get_target_similarity_indices(
            self.graph_indices(graph_1),
            self.graph_indices(graph_2),
        )

    def transfer_batch_to_torch(self, graph_1, graph_2):
        # Read GED indices before moving graph tensors to the training device.
        target = self.get_target_similarity_batch(graph_1, graph_2)
        if not self.model.training and not torch.is_grad_enabled():
            data = self.model.prepare_inference_inputs(graph_1, graph_2, self.device)
            data["target"] = target
            return data
        graph_1 = graph_1.to(self.device)
        graph_2 = graph_2.to(self.device)
        return {
            "graph_1": graph_1,
            "graph_2": graph_2,
            "target": target,
        }

    def evaluation_graph_batches(self, cache_key, graphs):
        """Build immutable candidate PyG batches once and reuse every epoch."""
        cached = self._evaluation_batch_cache.get(cache_key)
        if cached is not None:
            return cached

        cached = []
        for start in range(0, len(graphs), self.args.batch_size):
            graph_batch = Batch.from_data_list(
                graphs[start : start + self.args.batch_size]
            )
            indices = self.graph_indices(graph_batch).clone()
            graph_batch.batch_shape = self.model.graph_batch_shape(graph_batch)
            graph_batch.nonempty_graphs_verified = True
            graph_batch = graph_batch.to(self.device)
            cached.append((graph_batch, indices, graph_batch.num_graphs))
        self._evaluation_batch_cache[cache_key] = cached
        return cached

    def encode_evaluation_candidates(self, cache_key, graphs):
        """Represent each candidate graph once for the current fixed weights."""
        encoded = []
        for graph_batch, indices, batch_size in self.evaluation_graph_batches(
            cache_key,
            graphs,
        ):
            anchor_scales, node_counts = self.model.represent_graph_batch(graph_batch)
            encoded.append(
                (anchor_scales, node_counts, indices, batch_size)
            )
        return encoded

    def predict_query_from_encoded_candidates(self, query, candidates):
        """Score a query against cached candidate representations."""
        query_cache = {}
        prediction_chunks = []
        target_chunks = []
        for (
            candidate_anchors,
            candidate_counts,
            candidate_indices,
            batch_size,
        ) in candidates:
            if batch_size not in query_cache:
                query_batch = Batch.from_data_list(
                    [query.clone() for _ in range(batch_size)]
                )
                query_indices = self.graph_indices(query_batch).clone()
                query_batch.batch_shape = self.model.graph_batch_shape(query_batch)
                query_batch.nonempty_graphs_verified = True
                query_batch = query_batch.to(self.device)
                query_anchors, query_counts = self.model.represent_graph_batch(
                    query_batch
                )
                query_cache[batch_size] = (
                    query_anchors,
                    query_counts,
                    query_indices,
                )
            query_anchors, query_counts, query_indices = query_cache[batch_size]
            prediction_chunks.append(
                self.model.compare_and_predict(
                    query_anchors,
                    candidate_anchors,
                    query_counts,
                    candidate_counts,
                )
            )
            target_chunks.append(
                self.get_target_similarity_indices(
                    query_indices,
                    candidate_indices,
                )
            )
        return torch.cat(prediction_chunks), torch.cat(target_chunks)

    def create_batches(self):
        """Pair two independent permutations; every training graph occurs once per side."""
        generator = torch.Generator()
        generator.manual_seed(int(self.args.seed) + self.current_epoch * 1000003 + 707)
        graph_count = len(self.training_graphs)
        source_indices = torch.randperm(graph_count, generator=generator)
        target_indices = torch.randperm(graph_count, generator=generator)
        for start in range(0, graph_count, self.args.batch_size):
            yield (
                source_indices[start : start + self.args.batch_size],
                target_indices[start : start + self.args.batch_size],
            )

    def indexed_pair_batch(self, source_indices, target_indices):
        if torch.is_tensor(source_indices):
            source_indices = source_indices.tolist()
        if torch.is_tensor(target_indices):
            target_indices = target_indices.tolist()
        source = Batch.from_data_list(
            [self.training_graphs[index] for index in source_indices]
        )
        target = Batch.from_data_list(
            [self.training_graphs[index] for index in target_indices]
        )
        return self.transfer_batch_to_torch(source, target)

    def process_batch(self, source_indices, target_indices):
        self.optimizer.zero_grad()
        data = self.indexed_pair_batch(source_indices, target_indices)
        prediction = self.model(data)
        loss = F.mse_loss(prediction, data["target"], reduction="mean")
        loss.backward()
        self.optimizer.step()
        return loss.item(), data["target"].numel()

    def retrieval_metrics(self, queries, candidate_graphs, cache_key, description):
        """Evaluate retrieval predictions once and compute all paper metrics."""
        self.model.eval()
        squared_errors = []
        rho_values = []
        tau_values = []
        p10_values = []
        p20_values = []
        with torch.no_grad():
            candidates = self.encode_evaluation_candidates(
                cache_key,
                candidate_graphs,
            )
            for query in tqdm(queries, desc=description):
                prediction_tensor, target_tensor = (
                    self.predict_query_from_encoded_candidates(
                        query,
                        candidates,
                    )
                )
                prediction_vector = (
                    prediction_tensor.detach().cpu().numpy().astype(np.float64)
                )
                target_vector = (
                    target_tensor.detach().cpu().numpy().astype(np.float64)
                )
                squared_errors.extend(
                    ((prediction_vector - target_vector) ** 2).tolist()
                )
                rho_values.append(
                    calculate_ranking_correlation(
                        spearmanr,
                        prediction_vector,
                        target_vector,
                    )
                )
                tau_values.append(
                    calculate_ranking_correlation(
                        kendalltau,
                        prediction_vector,
                        target_vector,
                    )
                )
                p10_values.append(
                    calculate_prec_at_k(10, prediction_vector, target_vector)
                )
                p20_values.append(
                    calculate_prec_at_k(20, prediction_vector, target_vector)
                )

        return {
            "mse": self.finite_mean(squared_errors),
            "rho": self.finite_mean(rho_values),
            "tau": self.finite_mean(tau_values),
            "p10": self.finite_mean(p10_values),
            "p20": self.finite_mean(p20_values),
            "pair_count": len(squared_errors),
        }

    def validation_metrics(self):
        """Evaluate validation queries against the optimization subset."""
        metrics = self.retrieval_metrics(
            self.validation_graphs,
            self.training_graphs,
            "validation_training_candidates",
            "Validation queries",
        )
        metrics.update({
            "query_count": len(self.validation_graphs),
            "candidate_count": len(self.training_graphs),
            "validation_queries_are_in_database": False,
        })
        return metrics

    def save_best_model(self, state_dict):
        save_path = Path(self.args.best_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = save_path.with_suffix(".pt.tmp")
        torch.save(state_dict, temporary)
        temporary.replace(save_path)

    def fit(self):
        """Train from scratch and retain the minimum full-precision validation MSE."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        best_validation_mse = float("inf")
        selection = None
        print(f"Training: {self.args.epochs} epochs; validation: "
              f"{self.args.validation_start_epoch}--{self.args.epochs} "
              f"every {self.args.validation_interval} epochs", flush=True)
        show_progress = sys.stderr.isatty()
        progress = trange(self.args.epochs, desc="Epoch", disable=not show_progress)
        for epoch in progress:
            self.model.train()
            self.current_epoch = epoch
            mse_sum = 0.0
            pair_seen = 0
            for source_indices, target_indices in tqdm(
                self.create_batches(),
                total=math.ceil(len(self.training_graphs) / self.args.batch_size),
                desc="Batches", leave=False, disable=not show_progress,
            ):
                mse, count = self.process_batch(source_indices, target_indices)
                mse_sum += mse * count
                pair_seen += count
            epoch_number = epoch + 1
            training_mse = truncate_mse(mse_sum / pair_seen)
            progress.set_description(f"Epoch (MSE x1000={training_mse})")
            if not show_progress and (epoch_number == 1 or epoch_number % 100 == 0):
                print(f"TRAIN_EPOCH {epoch_number} mse_x1000={training_mse} pairs={pair_seen}", flush=True)
            should_validate = (
                epoch_number >= self.args.validation_start_epoch
                and ((epoch_number - self.args.validation_start_epoch)
                     % self.args.validation_interval == 0
                     or epoch_number == self.args.epochs)
            )
            if not should_validate:
                continue
            validation = self.validation_metrics()
            validation_mse = float(validation["mse"])
            print(f"VALIDATION epoch={epoch_number} {format_metrics(validation)}", flush=True)
            if not math.isfinite(validation_mse):
                raise RuntimeError(f"Non-finite validation MSE at epoch {epoch_number}")
            if validation_mse >= best_validation_mse:
                continue
            state = {name: value.detach().cpu().clone()
                     for name, value in self.model.state_dict().items()}
            self.save_best_model(state)
            best_validation_mse = validation_mse
            selection = {
                "selection_metric": "validation_mse",
                "epoch": epoch_number,
                "checkpoint": "best_mse.pt",
                "checkpoint_sha256": hashlib.sha256(
                    Path(self.args.best_save_path).read_bytes()
                ).hexdigest(),
                "validation_metrics": validation,
            }
            destination = self.args.output_dir / "selection.json"
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
            temporary.replace(destination)
            print(f"BEST_MODEL epoch={epoch_number} path={self.args.best_save_path}", flush=True)
        if selection is None:
            raise RuntimeError("Training ended before any validation checkpoint.")
        self.model.load_state_dict(load_checkpoint(self.args.best_save_path, self.device), strict=True)
        print(f"Loaded best validation checkpoint from epoch {selection['epoch']}.", flush=True)
        return selection

    @staticmethod
    def finite_mean(values):
        finite = [float(value) for value in values if np.isfinite(value)]
        return float(np.mean(finite)) if finite else float("nan")

    def evaluate_full_training_database(self):
        """Use official test as queries and every official-train graph as database."""
        metrics = self.retrieval_metrics(
            self.testing_graphs,
            self.full_training_database,
            "test_full_training_candidates",
            "Test queries",
        )
        metrics.update({
            "query_count": len(self.testing_graphs),
            "candidate_count": len(self.full_training_database),
            "candidate_training_count": len(self.training_graphs),
            "candidate_validation_count": len(self.validation_graphs),
            "test_queries_are_in_database": False,
        })
        return metrics
