"""CoReMatch: pair-independent graph similarity learning with shared anchors."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from torch_geometric.nn import GINConv, GraphSizeNorm

from layers import (
    AnchorLevelComparator,
    NeuralTensorNetwork,
    ScaleWiseAttentionPooling,
    SharedAnchorRepresentation,
)


class MultiScaleGraphEncoder(nn.Module):
    """Extract node representations at multiple receptive-field scales."""

    def __init__(
        self,
        input_dim,
        channels=(64, 32, 16),
        residual_alpha=0.1,
        dropout=0.0,
    ):
        super().__init__()
        self.channels = tuple(int(channel) for channel in channels)
        self.residual_alpha = float(residual_alpha)
        self.dropout = float(dropout)
        self.input_projection = nn.Linear(input_dim, 32)
        self.graph_size_norm = GraphSizeNorm()
        dimensions = (32, *self.channels)
        self.gin_layers = nn.ModuleList(
            [
                GINConv(
                    self.make_gin_mlp(left, right),
                    train_eps=True,
                )
                for left, right in zip(dimensions, dimensions[1:])
            ]
        )
        self.residual_projections = nn.ModuleList(
            [
                nn.Linear(left, right, bias=False)
                for left, right in zip(dimensions, dimensions[1:])
            ]
        )

    @staticmethod
    def make_gin_mlp(input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, graph):
        current = self.input_projection(graph.x.float())
        current = self.graph_size_norm(
            current, graph.batch,
            batch_size=graph.ptr.numel() - 1,
        )
        scale_features = []
        for gin_layer, residual_projection in zip(
            self.gin_layers,
            self.residual_projections,
        ):
            updated = gin_layer(current, graph.edge_index)
            updated = (
                updated
                + self.residual_alpha * residual_projection(current)
            )
            current = F.relu(updated)
            current = F.dropout(
                current,
                p=self.dropout,
                training=self.training,
            )
            scale_features.append(current)
        return scale_features


class SimilarityPrediction(nn.Sequential):
    """Map the cross-graph comparison and fusion features to a similarity."""

    def __init__(self, feature_dim):
        super().__init__(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, comparison_representation):
        logits = super().forward(comparison_representation)
        return torch.sigmoid(logits).view(-1)


class CrossGraphComparisonAndFusion(nn.Module):
    """Fuse graph-level comparison (GLC), anchor-level comparison (ALC), and GSF."""

    def __init__(self, anchor_dim, num_scales, anchor_count):
        super().__init__()
        self.num_scales = int(num_scales)
        self.anchor_count = int(anchor_count)
        self.scale_wise_attention_pooling = ScaleWiseAttentionPooling(
            anchor_dim=anchor_dim,
            num_scales=num_scales,
            anchor_count=anchor_count,
            pool_dim=32,
        )
        self.neural_tensor_network = NeuralTensorNetwork(
            32,
            relation_channels=16,
        )
        self.anchor_level_comparator = AnchorLevelComparator(
            anchor_dim,
            num_scales,
            anchor_count,
        )
        self.feature_dim = 16 + num_scales * anchor_count + 2

    @staticmethod
    def graph_size_features(node_counts_1, node_counts_2):
        """Compute graph-size features (GSF) for paired graph batches."""
        if node_counts_1.shape != node_counts_2.shape:
            raise ValueError("Paired graph batches must have the same batch size.")
        counts_1 = node_counts_1.float()
        counts_2 = node_counts_2.float()
        count_sum = counts_1 + counts_2
        relative_gap = 2.0 * torch.abs(counts_1 - counts_2) / (
            count_sum + 1e-12
        )
        mean_scale = torch.log1p(0.5 * count_sum)
        return torch.stack((relative_gap, mean_scale), dim=1)

    def forward(
        self,
        anchor_scales_1,
        anchor_scales_2,
        node_counts_1,
        node_counts_2,
    ):
        graph_1 = self.scale_wise_attention_pooling(anchor_scales_1)
        graph_2 = self.scale_wise_attention_pooling(anchor_scales_2)
        comparison_features = [
            self.neural_tensor_network(
                graph_1.unsqueeze(1),
                graph_2.unsqueeze(1),
            ).squeeze(1),
            self.anchor_level_comparator(
                anchor_scales_1,
                anchor_scales_2,
            ),
            self.graph_size_features(node_counts_1, node_counts_2),
        ]
        comparison_representation = torch.cat(comparison_features, dim=1)
        if comparison_representation.size(1) != self.feature_dim:
            raise RuntimeError(
                "Cross-graph comparison and fusion produced {} features; "
                "expected {}.".format(
                    comparison_representation.size(1),
                    self.feature_dim,
                )
            )
        return comparison_representation


class CoReMatch(nn.Module):
    """Graph similarity prediction with shared structural anchors."""

    def __init__(self, args, number_of_labels):
        super().__init__()
        channels = (64, 32, 16)

        self.multi_scale_graph_encoder = MultiScaleGraphEncoder(
            input_dim=number_of_labels,
            channels=channels,
            residual_alpha=args.residual_alpha,
            dropout=args.dropout,
        )

        self.shared_anchor_representation = (
            SharedAnchorRepresentation(
                input_dims=channels,
                anchor_count=args.anchor_count,
                anchor_dim=args.anchor_dim,
                num_heads=args.attention_heads,
                ffn_ratio=args.ffn_ratio,
                dropout=args.attention_dropout,
                layer_scale_init=args.layer_scale_init,
            )
        )

        self.cross_graph_comparison_and_fusion = (
            CrossGraphComparisonAndFusion(
                anchor_dim=(
                    self.shared_anchor_representation.anchor_dim
                ),
                num_scales=len(channels),
                anchor_count=args.anchor_count,
            )
        )

        self.similarity_prediction = SimilarityPrediction(
            self.cross_graph_comparison_and_fusion.feature_dim
        )

    @staticmethod
    def graph_node_counts(graph):
        return graph.ptr[1:] - graph.ptr[:-1]

    def clear_parameter_state(self):
        """Discard derived parameter tensors before the next evaluation."""
        for module in self.modules():
            if module is not self and hasattr(module, "clear_parameter_state"):
                module.clear_parameter_state()
        return self

    @staticmethod
    def graph_batch_shape(graph):
        """Validate CPU graph pointers and return the graph count and padding size."""
        if graph.ptr.device.type != "cpu" or graph.batch.device.type != "cpu":
            raise ValueError("Read graph batch metadata before device transfer.")
        counts = graph.ptr[1:] - graph.ptr[:-1]
        if (counts.numel() == 0 or bool((counts <= 0).any())
                or int(graph.ptr[0]) != 0 or int(graph.ptr[-1]) != graph.x.size(0)):
            raise ValueError("Inference requires nonempty graphs with valid batch pointers.")
        expected = torch.repeat_interleave(torch.arange(counts.numel()), counts)
        if not torch.equal(graph.batch, expected):
            raise ValueError("Graph batch indices must agree with graph pointers.")
        return counts.numel(), int(counts.max())

    @staticmethod
    def prepare_inference_inputs(graph_1, graph_2, device):
        """Read CPU graph metadata and transfer both graph batches to the device."""
        shapes = [CoReMatch.graph_batch_shape(graph) for graph in (graph_1, graph_2)]
        if shapes[0][0] != shapes[1][0]:
            raise ValueError("Paired batches must have equal graph counts.")
        return {"graph_1": graph_1.to(device), "graph_2": graph_2.to(device),
                "nonempty_graphs_verified": True, "batch_shapes": shapes}

    @staticmethod
    def make_shared_batch_layout(graph, batch_shape=None):
        """One ephemeral padding layout shared by the three independent scales.

        Known CPU dimensions avoid device-to-host size inference. The fallback
        reads the maximum size once, rather than once for every feature scale.
        No graph-specific layout is retained between forward calls.
        """
        batch_size = graph.ptr.numel() - 1
        max_nodes = (int((graph.ptr[1:] - graph.ptr[:-1]).max())
                     if batch_shape is None else batch_shape[1])
        if batch_shape is not None and batch_shape[0] != batch_size:
            raise ValueError("CPU batch shape disagrees with graph pointers.")
        if batch_size <= 0 or max_nodes <= 0:
            raise ValueError("Inference requires a nonempty graph batch.")
        positions = torch.arange(graph.batch.numel(), device=graph.batch.device)
        flat_index = positions - graph.ptr.index_select(0, graph.batch) + graph.batch * max_nodes
        mask = torch.zeros(batch_size * max_nodes, dtype=torch.bool, device=graph.batch.device)
        mask.index_fill_(0, flat_index, True)
        return flat_index, mask.view(batch_size, max_nodes), batch_size, max_nodes

    def represent_graph_batch(self, graph, *, nonempty_graphs_verified=False, batch_shape=None):
        if batch_shape is None:
            batch_shape = getattr(graph, "batch_shape", None)
        nonempty_graphs_verified = nonempty_graphs_verified or bool(
            getattr(graph, "nonempty_graphs_verified", False)
        )
        if batch_shape is None and graph.ptr.device.type == "cpu":
            batch_shape = self.graph_batch_shape(graph)
            nonempty_graphs_verified = True
        scale_features = self.multi_scale_graph_encoder(graph)
        dense_layout = None
        if not self.training and not torch.is_grad_enabled():
            dense_layout = self.make_shared_batch_layout(graph, batch_shape)
        anchor_scales = self.shared_anchor_representation(
            scale_features,
            graph.batch,
            nonempty_graphs_verified=nonempty_graphs_verified,
            dense_layout=dense_layout,
        )
        return anchor_scales, self.graph_node_counts(graph)

    def compare_and_predict(
        self,
        anchor_scales_1,
        anchor_scales_2,
        node_counts_1,
        node_counts_2,
    ):
        comparison_representation = self.cross_graph_comparison_and_fusion(
            anchor_scales_1,
            anchor_scales_2,
            node_counts_1,
            node_counts_2,
        )
        return self.similarity_prediction(comparison_representation)

    def forward(self, data):
        if not self.training and not torch.is_grad_enabled():
            if data["graph_1"].ptr.device.type == "cpu" and "batch_shapes" not in data:
                prepared = self.prepare_inference_inputs(
                    data["graph_1"], data["graph_2"], next(self.parameters()).device,
                )
                data = {**data, **prepared}
            return self._paired_graph_forward(data)
        verified = data.get("nonempty_graphs_verified", False)
        shapes = data.get("batch_shapes", (None, None))
        anchors_1, counts_1 = self.represent_graph_batch(
            data["graph_1"], nonempty_graphs_verified=verified, batch_shape=shapes[0])
        anchors_2, counts_2 = self.represent_graph_batch(
            data["graph_2"], nonempty_graphs_verified=verified, batch_shape=shapes[1])
        return self.compare_and_predict(
            anchors_1,
            anchors_2,
            counts_1,
            counts_2,
        )

    def _paired_graph_forward(self, data):
        """Encode paired graph batches together, keeping each graph independent.

        Separate batch IDs, edges, and attention masks prevent information
        exchange between graphs during representation construction.
        """
        left, right = data['graph_1'], data['graph_2']
        size = left.ptr.numel() - 1
        if size != right.ptr.numel() - 1:
            raise ValueError('Paired batches must have equal graph counts.')
        node_offset = left.x.shape[0]
        joint = SimpleNamespace(
            x=torch.cat((left.x, right.x), 0),
            edge_index=torch.cat((left.edge_index, right.edge_index + node_offset), 1),
            batch=torch.cat((left.batch, right.batch + size), 0),
            ptr=torch.cat((left.ptr, right.ptr[1:] + node_offset), 0),
        )
        shapes = data.get('batch_shapes')
        shape = (2 * size, max(shapes[0][1], shapes[1][1])) if shapes is not None else None
        anchors, counts = self.represent_graph_batch(joint,
            nonempty_graphs_verified=data.get('nonempty_graphs_verified', False), batch_shape=shape)
        first, second = [a[:size] for a in anchors], [a[size:] for a in anchors]
        comparison = self.cross_graph_comparison_and_fusion
        graph_vectors = comparison.scale_wise_attention_pooling(anchors)
        graph_level_features = comparison.neural_tensor_network(
            graph_vectors[:size, None], graph_vectors[size:, None]).squeeze(1)
        features = torch.cat((graph_level_features,
            comparison.anchor_level_comparator(first, second),
            comparison.graph_size_features(counts[:size], counts[size:])), dim=1)
        return self.similarity_prediction(features)
