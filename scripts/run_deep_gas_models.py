from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import mobilenet_v2, shufflenet_v2_x0_5

from prepare_gas_dataset import FEATURE_PREFIX, OUTPUT_DIR, PROJECT_ROOT, prepare_dataset

DEFAULT_MODELS = [
    "mlp",
    "cnn1d",
    "lenet1d",
    "eegnet_like",
    "mobilenet_v2",
    "shufflenet_v2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train deep models on the fixed gas dataset split."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=DEFAULT_MODELS,
        help="Model names to train.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "deep_gas_models",
        help="Directory to store experiment outputs.",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Maximum epochs.")
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience on validation macro-F1.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Default learning rate.")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="AdamW weight decay."
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_prepared_dataset() -> None:
    if not (OUTPUT_DIR / "train.csv").exists():
        prepare_dataset()


def load_split(name: str) -> pd.DataFrame:
    ensure_prepared_dataset()
    return pd.read_csv(OUTPUT_DIR / f"{name}.csv")


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith(FEATURE_PREFIX)]


def build_standardized_splits() -> dict[str, Any]:
    train_frame = load_split("train")
    val_frame = load_split("val")
    test_frame = load_split("test")

    feature_columns = get_feature_columns(train_frame)
    mean = train_frame[feature_columns].mean(axis=0).to_numpy(dtype=np.float32)
    std = train_frame[feature_columns].std(axis=0, ddof=0).to_numpy(dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)

    splits: dict[str, dict[str, Any]] = {}
    for split_name, frame in {
        "train": train_frame,
        "val": val_frame,
        "test": test_frame,
    }.items():
        features = frame[feature_columns].to_numpy(dtype=np.float32)
        features = (features - mean) / std
        labels = frame["gas_class"].to_numpy(dtype=np.int64) - 1

        flat = features
        seq = features[:, None, :]
        matrix = features.reshape(-1, 16, 8)
        matrix = matrix[:, None, :, :]
        image = F.interpolate(
            torch.from_numpy(matrix), size=(32, 32), mode="bilinear", align_corners=False
        ).numpy()

        splits[split_name] = {
            "flat": flat,
            "seq": seq,
            "matrix": matrix,
            "image": image,
            "labels": labels,
        }

    return {
        "splits": splits,
        "num_classes": int(train_frame["gas_class"].nunique()),
        "class_names": sorted(train_frame["gas_class"].unique().tolist()),
        "class_weights": compute_class_weights(train_frame["gas_class"].to_numpy(dtype=np.int64)),
        "split_summary": json.loads((OUTPUT_DIR / "split_summary.json").read_text(encoding="utf-8")),
    }


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    weights = total / (len(unique) * counts.astype(np.float32))
    ordered = np.zeros(int(unique.max()), dtype=np.float32)
    for label, weight in zip(unique, weights):
        ordered[int(label) - 1] = weight
    return ordered


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


class MLPClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1DClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class LeNet1DClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128)
            flattened = self.features(dummy).view(1, -1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class EEGNetLikeClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(16, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 3), padding=(0, 1), groups=16, bias=False),
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.25),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_mobilenet_v2(num_classes: int) -> nn.Module:
    model = mobilenet_v2(num_classes=num_classes, width_mult=0.5)
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=(1, 1),
        padding=first_conv.padding,
        bias=False,
    )
    return model


def build_shufflenet_v2(num_classes: int) -> nn.Module:
    model = shufflenet_v2_x0_5(num_classes=num_classes)
    first_conv = model.conv1[0]
    model.conv1[0] = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=(1, 1),
        padding=first_conv.padding,
        bias=False,
    )
    model.maxpool = nn.Identity()
    return model


def get_model_specs(num_classes: int, default_lr: float) -> dict[str, dict[str, Any]]:
    return {
        "mlp": {
            "builder": lambda: MLPClassifier(num_classes),
            "input_key": "flat",
            "lr": default_lr,
        },
        "cnn1d": {
            "builder": lambda: CNN1DClassifier(num_classes),
            "input_key": "seq",
            "lr": default_lr,
        },
        "lenet1d": {
            "builder": lambda: LeNet1DClassifier(num_classes),
            "input_key": "seq",
            "lr": default_lr,
        },
        "eegnet_like": {
            "builder": lambda: EEGNetLikeClassifier(num_classes),
            "input_key": "matrix",
            "lr": default_lr,
        },
        "mobilenet_v2": {
            "builder": lambda: build_mobilenet_v2(num_classes),
            "input_key": "image",
            "lr": default_lr * 0.5,
        },
        "shufflenet_v2": {
            "builder": lambda: build_shufflenet_v2(num_classes),
            "input_key": "image",
            "lr": default_lr * 0.5,
        },
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(features)
        loss = criterion(logits, labels)
        losses.append(loss.item() * labels.size(0))
        predictions = logits.argmax(dim=1)
        all_predictions.append(predictions.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_predictions)
    avg_loss = float(sum(losses) / len(y_true))
    metrics = compute_metrics(y_true, y_pred)
    return avg_loss, metrics, y_true, y_pred


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_items += labels.size(0)

    return float(total_loss / total_items)


def train_model(
    model_name: str,
    spec: dict[str, Any],
    data_bundle: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    train_split = data_bundle["splits"]["train"]
    val_split = data_bundle["splits"]["val"]
    test_split = data_bundle["splits"]["test"]
    input_key = spec["input_key"]

    train_loader = make_loader(
        train_split[input_key],
        train_split["labels"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        val_split[input_key],
        val_split["labels"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = make_loader(
        test_split[input_key],
        test_split["labels"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = spec["builder"]().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(data_bundle["class_weights"], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec["lr"],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(2, args.patience // 3),
    )

    history_path = output_dir / f"{model_name}_history.csv"
    best_state: dict[str, torch.Tensor] | None = None
    best_val_macro_f1 = -math.inf
    best_epoch = 0
    stale_epochs = 0

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_accuracy",
                "val_macro_f1",
                "val_weighted_f1",
                "val_balanced_accuracy",
                "lr",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
            val_loss, val_metrics, _, _ = evaluate(model, val_loader, device, criterion)
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_metrics["macro_f1"])

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_weighted_f1": val_metrics["weighted_f1"],
                    "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                    "lr": current_lr,
                }
            )
            handle.flush()

            if val_metrics["macro_f1"] > best_val_macro_f1:
                best_val_macro_f1 = val_metrics["macro_f1"]
                best_epoch = epoch
                stale_epochs = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale_epochs += 1

            print(
                f"[{model_name}] epoch {epoch:03d} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"lr={current_lr:.6f}",
                flush=True,
            )

            if stale_epochs >= args.patience:
                print(f"[{model_name}] early stopping at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError(f"No checkpoint captured for {model_name}.")

    model.load_state_dict(best_state)
    torch.save(best_state, output_dir / f"{model_name}_best.pt")

    _, val_metrics, _, _ = evaluate(model, val_loader, device, criterion)
    _, test_metrics, y_true, y_pred = evaluate(model, test_loader, device, criterion)

    label_order = list(range(data_bundle["num_classes"]))
    confusion = confusion_matrix(y_true, y_pred, labels=label_order)
    confusion_frame = pd.DataFrame(
        confusion,
        index=[f"true_{label + 1}" for label in label_order],
        columns=[f"pred_{label + 1}" for label in label_order],
    )
    confusion_frame.to_csv(output_dir / f"{model_name}_test_confusion_matrix.csv")

    report = classification_report(
        y_true,
        y_pred,
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        output_dir / f"{model_name}_test_classification_report.csv"
    )

    return {
        "model": model_name,
        "best_epoch": best_epoch,
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_weighted_f1": val_metrics["weighted_f1"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
    }


def run_experiments(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    data_bundle = build_standardized_splits()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_specs = get_model_specs(data_bundle["num_classes"], args.lr)

    metadata = {
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "default_lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "device": str(device),
        "models": args.models,
        "input_mapping": {name: model_specs[name]["input_key"] for name in args.models},
        "split_summary": data_bundle["split_summary"],
        "representation_note": (
            "Features are standardized with train statistics. "
            "For matrix/image models, each sample is reshaped to 16 sensors x 8 features "
            "assuming the 128 features are ordered in contiguous sensor blocks."
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    all_results: list[dict[str, Any]] = []
    error_log: list[dict[str, str]] = []

    started_at = time.time()
    for model_name in args.models:
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== Starting {model_name} on {device} ===", flush=True)
        try:
            result = train_model(
                model_name=model_name,
                spec=model_specs[model_name],
                data_bundle=data_bundle,
                args=args,
                output_dir=model_dir,
                device=device,
            )
            all_results.append(result)
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            print(f"!!! {model_name} failed: {message}", flush=True)
            error_log.append({"model": model_name, "error": message})

    elapsed = time.time() - started_at
    print(f"All requested models finished in {elapsed / 60:.2f} minutes.", flush=True)

    results_frame = pd.DataFrame(all_results)
    if not results_frame.empty:
        results_frame.sort_values(
            by=["test_macro_f1", "test_accuracy"], ascending=False, inplace=True
        )
        results_frame.to_csv(output_dir / "metrics_summary.csv", index=False)
        (output_dir / "report.md").write_text(
            build_report_markdown(results_frame),
            encoding="utf-8",
        )

    if error_log:
        (output_dir / "errors.json").write_text(
            json.dumps(error_log, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )


def build_report_markdown(results_frame: pd.DataFrame) -> str:
    lines = [
        "# Deep Gas Model Results",
        "",
        "Fixed split: per-class 60/20/20 in original row order, matching the earlier traditional-ML experiments.",
        "",
        "| Model | Val Macro-F1 | Test Macro-F1 | Test Accuracy | Test Balanced Accuracy | Best Epoch |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in results_frame.iterrows():
        lines.append(
            "| {model} | {val_macro_f1:.4f} | {test_macro_f1:.4f} | {test_accuracy:.4f} | "
            "{test_balanced_accuracy:.4f} | {best_epoch} |".format(**row.to_dict())
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()
