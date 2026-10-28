"""Train and evaluate CoReMatch."""

import argparse
import csv
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from configs import ARCHITECTURE, DATASET_CONFIGS
from reporting import display_metrics, format_metrics
from trainer import Trainer, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def parse_args(cli=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASET_CONFIGS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory; test runs default to results/<dataset>_<YYYYMMDD_HHMMSS>.")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--anchor-count", type=int)
    parser.add_argument("--anchor-dim", type=int)
    parser.add_argument("--seed", type=int)
    for name, value in ARCHITECTURE.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value,
                            help=("Number of heads for node and anchor self-attention."
                                  if name == "attention_heads" else None))
    args = parser.parse_args(cli)
    for key, value in DATASET_CONFIGS[args.dataset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.validation_start_epoch = max(1, args.epochs - 1000)
    args.dataset_root = (args.dataset_root or ROOT / "datasets" / args.dataset).resolve()
    folder = "results" if args.evaluate_only else "runs"
    output_name = args.dataset
    if args.evaluate_only and args.output_dir is None:
        timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        output_name = f"{args.dataset}_{timestamp}"
    args.output_dir = (args.output_dir or ROOT / folder / output_name).resolve()
    args.checkpoint = (args.checkpoint or ROOT / "checkpoints" / args.dataset / "best_mse.pt").resolve()
    args.best_save_path = str(args.output_dir / "best_mse.pt")
    if min(args.epochs, args.batch_size, args.validation_interval,
           args.anchor_count, args.anchor_dim, args.attention_heads, args.ffn_ratio) <= 0:
        parser.error("Epochs, batch size, validation interval and model dimensions must be positive")
    if args.anchor_dim % args.attention_heads:
        parser.error("Anchor dimension must be divisible by the attention head count")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("Learning rate must be positive and weight decay nonnegative")
    if not 0 <= args.dropout < 1 or not 0 <= args.attention_dropout < 1:
        parser.error("Dropout must be in [0, 1)")
    return args


def set_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_result(args, metrics):
    result = {
        "dataset": args.dataset,
        "mse": metrics["mse"],
        "mse_x1000": metrics["mse"] * 1000,
        **{key: metrics[key] for key in ("rho", "tau", "p10", "p20")},
    }
    stem = "test_result" if args.evaluate_only else "result"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{stem}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    row = {"dataset": args.dataset, **display_metrics(metrics)}
    csv_path = args.output_dir / f"{stem}.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    temporary.replace(csv_path)
    print(f"RESULT dataset={args.dataset} {format_metrics(metrics)}", flush=True)
    print(f"RESULT_SAVED {path}", flush=True)


def main():
    args = parse_args()
    if args.evaluate_only:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {args.output_dir}. Choose a new --output-dir.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    trainer = Trainer(args)
    if args.evaluate_only:
        trainer.model.load_state_dict(load_checkpoint(args.checkpoint, trainer.device), strict=True)
    else:
        trainer.fit()
    metrics = trainer.evaluate_full_training_database()
    if not all(math.isfinite(metrics[key]) for key in ("mse", "rho", "tau", "p10", "p20")):
        raise RuntimeError("Non-finite test metrics")
    save_result(args, metrics)


if __name__ == "__main__":
    main()
