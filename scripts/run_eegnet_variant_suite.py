from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

from prepare_gas_dataset import FEATURE_PREFIX, OUTPUT_DIR, PROJECT_ROOT, prepare_dataset

PRESET_EXPERIMENTS: dict[str, dict[str, Any]] = {
    "baseline": {
        "display_name": "EEGNet Baseline 16x8",
        "orientation": "16x8",
        "scaler_mode": "featurewise",
        "attention": "none",
        "norm_type": "batch",
        "label_smoothing": 0.0,
        "input_noise_std": 0.0,
        "sensor_dropout_prob": 0.0,
        "use_ema": False,
    },
    "input_8x16": {
        "display_name": "EEGNet Input 8x16",
        "orientation": "8x16",
        "scaler_mode": "featurewise",
        "attention": "none",
        "norm_type": "batch",
        "label_smoothing": 0.0,
        "input_noise_std": 0.0,
        "sensor_dropout_prob": 0.0,
        "use_ema": False,
    },
    "eca": {
        "display_name": "EEGNet + ECA",
        "orientation": "16x8",
        "scaler_mode": "featurewise",
        "attention": "eca",
        "norm_type": "batch",
        "label_smoothing": 0.0,
        "input_noise_std": 0.0,
        "sensor_dropout_prob": 0.0,
        "use_ema": False,
    },
    "sensor_norm_only": {
        "display_name": "EEGNet + PerSensorNorm",
        "orientation": "16x8",
        "scaler_mode": "per_sensor",
        "attention": "none",
        "norm_type": "batch",
        "label_smoothing": 0.0,
        "input_noise_std": 0.0,
        "sensor_dropout_prob": 0.0,
        "use_ema": False,
    },
    "label_smoothing_only": {
        "display_name": "EEGNet + LabelSmoothing",
        "orientation": "16x8",
        "scaler_mode": "featurewise",
        "attention": "none",
        "norm_type": "batch",
        "label_smoothing": 0.05,
        "input_noise_std": 0.0,
        "sensor_dropout_prob": 0.0,
        "use_ema": False,
    },
    "eca_sensor_norm": {
        "display_name": "EEGNet + ECA + PerSensorNorm",
        "orientation": "16x8",
        "scaler_mode": "per_sensor",
        "attention": "eca",
        "norm_type": "batch",
        "label_smoothing": 0.0,
        "input_noise_std": 0.0,
        "sensor_dropout_prob": 0.0,
        "use_ema": False,
    },
    "eca_sensor_norm_ls": {
        "display_name": "EEGNet + ECA + PerSensorNorm + LabelSmoothing",
        "orientation": "16x8",
        "scaler_mode": "per_sensor",
        "attention": "eca",
        "norm_type": "batch",
        "label_smoothing": 0.05,
        "input_noise_std": 0.0,
        "sensor_dropout_prob": 0.0,
        "use_ema": False,
    },
    "eca_ls": {
        "display_name": "EEGNet + ECA + LabelSmoothing",
        "orientation": "16x8",
        "scaler_mode": "featurewise",
        "attention": "eca",
        "norm_type": "batch",
        "label_smoothing": 0.05,
        "input_noise_std": 0.0,
        "sensor_dropout_prob": 0.0,
        "use_ema": False,
    },
    "sensor_norm_ls": {
        "display_name": "EEGNet + PerSensorNorm + LabelSmoothing",
        "orientation": "16x8",
        "scaler_mode": "per_sensor",
        "attention": "none",
        "norm_type": "batch",
        "label_smoothing": 0.05,
        "input_noise_std": 0.0,
        "sensor_dropout_prob": 0.0,
        "use_ema": False,
    },
    "se_sensor_norm_ls": {
        "display_name": "EEGNet + SE + PerSensorNorm + LabelSmoothing",
        "orientation": "16x8",
        "scaler_mode": "per_sensor",
        "attention": "se",
        "norm_type": "batch",
        "label_smoothing": 0.05,
        "input_noise_std": 0.0,
        "sensor_dropout_prob": 0.0,
        "use_ema": False,
    },
    "eca_sensor_norm_ls_group": {
        "display_name": "EEGNet + ECA + PerSensorNorm + LabelSmoothing + GroupNorm",
        "orientation": "16x8",
        "scaler_mode": "per_sensor",
        "attention": "eca",
        "norm_type": "group",
        "label_smoothing": 0.05,
        "input_noise_std": 0.0,
        "sensor_dropout_prob": 0.0,
        "use_ema": False,
    },
    "eca_sensor_norm_ls_ema": {
        "display_name": "EEGNet + ECA + PerSensorNorm + LabelSmoothing + EMA",
        "orientation": "16x8",
        "scaler_mode": "per_sensor",
        "attention": "eca",
        "norm_type": "batch",
        "label_smoothing": 0.05,
        "input_noise_std": 0.0,
        "sensor_dropout_prob": 0.0,
        "use_ema": True,
    },
    "final_combo": {
        "display_name": "EEGNet Final Combo",
        "orientation": "16x8",
        "scaler_mode": "per_sensor",
        "attention": "eca",
        "norm_type": "group",
        "label_smoothing": 0.05,
        "input_noise_std": 0.03,
        "sensor_dropout_prob": 0.1,
        "use_ema": True,
    },
}
DEFAULT_CONFIG: dict[str, Any] = {
    "orientation": "16x8",
    "scaler_mode": "featurewise",
    "attention": "none",
    "attention_stage": "block2",
    "eca_kernel_size": 3,
    "norm_type": "batch",
    "label_smoothing": 0.0,
    "input_noise_std": 0.0,
    "sensor_dropout_prob": 0.0,
    "use_ema": False,
    "temporal_filters": 8,
    "spatial_filters": 16,
    "frontend_type": "single",
    "use_sensor_attention": False,
}
PRESET_EXPERIMENTS.update(
    {
        "eca_k5": {
            "display_name": "EEGNet + ECA(k=5)",
            "attention": "eca",
            "eca_kernel_size": 5,
        },
        "eca_k7": {
            "display_name": "EEGNet + ECA(k=7)",
            "attention": "eca",
            "eca_kernel_size": 7,
        },
        "eca_block1": {
            "display_name": "EEGNet + ECA@Block1",
            "attention": "eca",
            "attention_stage": "block1",
        },
        "eca_both": {
            "display_name": "EEGNet + ECA@BothBlocks",
            "attention": "eca",
            "attention_stage": "both",
        },
        "eca_wide_12_24": {
            "display_name": "EEGNet + ECA Wide(12/24)",
            "attention": "eca",
            "temporal_filters": 12,
            "spatial_filters": 24,
        },
        "eca_wide_10_20": {
            "display_name": "EEGNet + ECA Wide(10/20)",
            "attention": "eca",
            "temporal_filters": 10,
            "spatial_filters": 20,
        },
        "eca_wide_14_28": {
            "display_name": "EEGNet + ECA Wide(14/28)",
            "attention": "eca",
            "temporal_filters": 14,
            "spatial_filters": 28,
        },
        "eca_wide_16_32": {
            "display_name": "EEGNet + ECA Wide(16/32)",
            "attention": "eca",
            "temporal_filters": 16,
            "spatial_filters": 32,
        },
        "eca_wide_12_24_block1": {
            "display_name": "EEGNet + ECA Wide(12/24) @ Block1",
            "attention": "eca",
            "attention_stage": "block1",
            "temporal_filters": 12,
            "spatial_filters": 24,
        },
        "eca_wide_12_24_both": {
            "display_name": "EEGNet + ECA Wide(12/24) @ BothBlocks",
            "attention": "eca",
            "attention_stage": "both",
            "temporal_filters": 12,
            "spatial_filters": 24,
        },
        "eca_multiscale": {
            "display_name": "EEGNet + ECA + MultiScaleFrontend",
            "attention": "eca",
            "frontend_type": "multiscale",
        },
        "sensor_attention": {
            "display_name": "EEGNet + SensorAttention",
            "use_sensor_attention": True,
        },
        "dual_attention": {
            "display_name": "EEGNet + SensorAttention + ECA",
            "attention": "eca",
            "use_sensor_attention": True,
        },
        "eca_noise_001": {
            "display_name": "EEGNet + ECA + Noise(0.01)",
            "attention": "eca",
            "input_noise_std": 0.01,
        },
        "eca_wide_12_24_noise_0005": {
            "display_name": "EEGNet + ECA Wide(12/24) + Noise(0.005)",
            "attention": "eca",
            "temporal_filters": 12,
            "spatial_filters": 24,
            "input_noise_std": 0.005,
        },
        "eca_wide_12_24_noise_001": {
            "display_name": "EEGNet + ECA Wide(12/24) + Noise(0.01)",
            "attention": "eca",
            "temporal_filters": 12,
            "spatial_filters": 24,
            "input_noise_std": 0.01,
        },
        "eca_wide_12_24_block1_noise_0005": {
            "display_name": "EEGNet + ECA Wide(12/24) @ Block1 + Noise(0.005)",
            "attention": "eca",
            "attention_stage": "block1",
            "temporal_filters": 12,
            "spatial_filters": 24,
            "input_noise_std": 0.005,
        },
        "eca_wide_12_24_block1_noise_001": {
            "display_name": "EEGNet + ECA Wide(12/24) @ Block1 + Noise(0.01)",
            "attention": "eca",
            "attention_stage": "block1",
            "temporal_filters": 12,
            "spatial_filters": 24,
            "input_noise_std": 0.01,
        },
        "eca_sensor_mask_005": {
            "display_name": "EEGNet + ECA + SensorMask(0.05)",
            "attention": "eca",
            "sensor_dropout_prob": 0.05,
        },
    }
)
DEFAULT_EXPERIMENTS = list(PRESET_EXPERIMENTS.keys())
CANONICAL_SENSOR_COUNT = 16
CANONICAL_FEATURE_COUNT = 8


def expand_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_CONFIG)
    merged.update(config)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an EEGNet variant experiment suite on the fixed gas split."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        choices=list(PRESET_EXPERIMENTS.keys()),
        help="Preset experiments to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "eegnet_variant_suite",
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
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
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


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    weights = total / (len(unique) * counts.astype(np.float32))
    ordered = np.zeros(int(unique.max()), dtype=np.float32)
    for label, weight in zip(unique, weights):
        ordered[int(label) - 1] = weight
    return ordered


def reshape_to_canonical(features: np.ndarray) -> np.ndarray:
    return features.reshape(-1, CANONICAL_SENSOR_COUNT, CANONICAL_FEATURE_COUNT)


def fit_scaler(train_features: np.ndarray, scaler_mode: str) -> dict[str, np.ndarray]:
    canonical = reshape_to_canonical(train_features)
    if scaler_mode == "featurewise":
        mean = canonical.mean(axis=0, keepdims=True)
        std = canonical.std(axis=0, ddof=0, keepdims=True)
    elif scaler_mode == "per_sensor":
        mean = canonical.mean(axis=(0, 2), keepdims=True)
        std = canonical.std(axis=(0, 2), ddof=0, keepdims=True)
    else:
        raise ValueError(f"Unsupported scaler_mode: {scaler_mode}")
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def transform_features(
    features: np.ndarray,
    scaler: dict[str, np.ndarray],
    orientation: str,
) -> np.ndarray:
    canonical = reshape_to_canonical(features).astype(np.float32)
    canonical = (canonical - scaler["mean"]) / scaler["std"]

    if orientation == "16x8":
        matrix = canonical
    elif orientation == "8x16":
        matrix = np.transpose(canonical, (0, 2, 1))
    else:
        raise ValueError(f"Unsupported orientation: {orientation}")

    return matrix[:, None, :, :]


def build_data_bundle(config: dict[str, Any]) -> dict[str, Any]:
    train_frame = load_split("train")
    val_frame = load_split("val")
    test_frame = load_split("test")
    feature_columns = get_feature_columns(train_frame)

    train_features = train_frame[feature_columns].to_numpy(dtype=np.float32)
    val_features = val_frame[feature_columns].to_numpy(dtype=np.float32)
    test_features = test_frame[feature_columns].to_numpy(dtype=np.float32)

    scaler = fit_scaler(train_features, config["scaler_mode"])

    splits = {
        "train": {
            "features": transform_features(train_features, scaler, config["orientation"]),
            "labels": train_frame["gas_class"].to_numpy(dtype=np.int64) - 1,
        },
        "val": {
            "features": transform_features(val_features, scaler, config["orientation"]),
            "labels": val_frame["gas_class"].to_numpy(dtype=np.int64) - 1,
        },
        "test": {
            "features": transform_features(test_features, scaler, config["orientation"]),
            "labels": test_frame["gas_class"].to_numpy(dtype=np.int64) - 1,
        },
    }

    return {
        "splits": splits,
        "num_classes": int(train_frame["gas_class"].nunique()),
        "class_weights": compute_class_weights(train_frame["gas_class"].to_numpy(dtype=np.int64)),
        "split_summary": json.loads((OUTPUT_DIR / "split_summary.json").read_text(encoding="utf-8")),
        "scaler_mode": config["scaler_mode"],
        "orientation": config["orientation"],
    }


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int | None = None,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    generator = None
    worker_init_fn = None

    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

        def worker_init_fn(worker_id: int) -> None:
            worker_seed = seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def make_norm2d(num_channels: int, norm_type: str) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "group":
        for groups in (4, 2, 1):
            if num_channels % groups == 0:
                return nn.GroupNorm(groups, num_channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ECABlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1).transpose(1, 2)
        weights = self.conv(weights)
        weights = self.activation(weights).transpose(1, 2).unsqueeze(-1)
        return x * weights


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced_channels = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x).flatten(1)
        weights = self.fc(pooled).view(x.shape[0], x.shape[1], 1, 1)
        return x * weights


def split_channels(total_channels: int, branch_count: int) -> list[int]:
    base = total_channels // branch_count
    remainder = total_channels % branch_count
    channels = [base] * branch_count
    for index in range(remainder):
        channels[index] += 1
    return channels


class TemporalFrontend(nn.Module):
    def __init__(self, out_channels: int, frontend_type: str) -> None:
        super().__init__()
        self.frontend_type = frontend_type
        if frontend_type == "single":
            self.single = nn.Conv2d(
                1,
                out_channels,
                kernel_size=(1, 3),
                padding=(0, 1),
                bias=False,
            )
        elif frontend_type == "multiscale":
            kernels = (3, 5, 7)
            branch_channels = split_channels(out_channels, len(kernels))
            self.branches = nn.ModuleList(
                [
                    nn.Conv2d(
                        1,
                        branch_out,
                        kernel_size=(1, kernel),
                        padding=(0, kernel // 2),
                        bias=False,
                    )
                    for branch_out, kernel in zip(branch_channels, kernels)
                ]
            )
        else:
            raise ValueError(f"Unsupported frontend_type: {frontend_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frontend_type == "single":
            return self.single(x)
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class InputSensorAttention(nn.Module):
    def __init__(self, orientation: str, sensor_count: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(sensor_count // reduction, 1)
        self.orientation = orientation
        self.fc = nn.Sequential(
            nn.Linear(sensor_count, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, sensor_count),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.orientation == "16x8":
            pooled = x.mean(dim=3).squeeze(1)
            weights = self.fc(pooled).view(x.shape[0], 1, x.shape[2], 1)
        elif self.orientation == "8x16":
            pooled = x.mean(dim=2).squeeze(1)
            weights = self.fc(pooled).view(x.shape[0], 1, 1, x.shape[3])
        else:
            raise ValueError(f"Unsupported orientation: {self.orientation}")
        return x * weights


def build_attention(name: str, channels: int, eca_kernel_size: int) -> nn.Module:
    if name == "none":
        return nn.Identity()
    if name == "eca":
        return ECABlock(channels, kernel_size=eca_kernel_size)
    if name == "se":
        return SEBlock(channels)
    raise ValueError(f"Unsupported attention: {name}")


class EEGNetVariant(nn.Module):
    def __init__(
        self,
        num_classes: int,
        sensor_dim: int,
        orientation: str,
        attention: str,
        attention_stage: str,
        norm_type: str,
        eca_kernel_size: int,
        temporal_filters: int,
        spatial_filters: int,
        frontend_type: str,
        use_sensor_attention: bool,
    ) -> None:
        super().__init__()
        self.input_sensor_attention = (
            InputSensorAttention(orientation=orientation, sensor_count=CANONICAL_SENSOR_COUNT)
            if use_sensor_attention
            else nn.Identity()
        )
        self.temporal_frontend = TemporalFrontend(
            out_channels=temporal_filters,
            frontend_type=frontend_type,
        )
        self.temporal_norm = make_norm2d(temporal_filters, norm_type)
        self.spatial_conv = nn.Conv2d(
            temporal_filters,
            spatial_filters,
            kernel_size=(sensor_dim, 1),
            groups=temporal_filters,
            bias=False,
        )
        self.spatial_norm = make_norm2d(spatial_filters, norm_type)
        self.block1_post = nn.Sequential(
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                kernel_size=(1, 3),
                padding=(0, 1),
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(spatial_filters, spatial_filters, kernel_size=1, bias=False),
            make_norm2d(spatial_filters, norm_type),
            nn.ELU(inplace=True),
        )
        self.block1_attention = build_attention(
            attention if attention_stage in {"block1", "both"} else "none",
            spatial_filters,
            eca_kernel_size,
        )
        self.block2_attention = build_attention(
            attention if attention_stage in {"block2", "both"} else "none",
            spatial_filters,
            eca_kernel_size,
        )
        self.head = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.25),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(spatial_filters, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_sensor_attention(x)
        x = self.temporal_frontend(x)
        x = self.temporal_norm(x)
        x = self.spatial_conv(x)
        x = self.spatial_norm(x)
        x = self.block1_post(x)
        x = self.block1_attention(x)
        x = self.block2(x)
        x = self.block2_attention(x)
        return self.head(x)


class EMAHelper:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.state_dict().items()
        }
        self.backup: dict[str, torch.Tensor] | None = None

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, parameter in model.state_dict().items():
                if torch.is_floating_point(self.shadow[name]):
                    self.shadow[name].mul_(self.decay).add_(
                        parameter.detach(),
                        alpha=1.0 - self.decay,
                    )
                else:
                    self.shadow[name].copy_(parameter.detach())

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {
            name: parameter.detach().clone()
            for name, parameter in model.state_dict().items()
        }
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module) -> None:
        if self.backup is None:
            return
        model.load_state_dict(self.backup, strict=True)
        self.backup = None


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
        all_predictions.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_predictions)
    avg_loss = float(sum(losses) / len(y_true))
    metrics = compute_metrics(y_true, y_pred)
    return avg_loss, metrics, y_true, y_pred


def apply_input_augmentation(
    features: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    augmented = features

    if config["input_noise_std"] > 0:
        augmented = augmented + torch.randn_like(augmented) * float(config["input_noise_std"])

    dropout_prob = float(config["sensor_dropout_prob"])
    if dropout_prob > 0:
        if config["orientation"] == "16x8":
            sensor_count = augmented.shape[2]
            keep_mask = (
                torch.rand((augmented.shape[0], 1, sensor_count, 1), device=augmented.device)
                >= dropout_prob
            ).float()
        else:
            sensor_count = augmented.shape[3]
            keep_mask = (
                torch.rand((augmented.shape[0], 1, 1, sensor_count), device=augmented.device)
                >= dropout_prob
            ).float()

        flattened = keep_mask.view(keep_mask.shape[0], -1)
        all_dropped = flattened.sum(dim=1) == 0
        if all_dropped.any():
            if config["orientation"] == "16x8":
                keep_mask[all_dropped, :, 0, :] = 1.0
            else:
                keep_mask[all_dropped, :, :, 0] = 1.0
        augmented = augmented * keep_mask

    return augmented


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    config: dict[str, Any],
    ema: EMAHelper | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        features = apply_input_augmentation(features, config)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * labels.size(0)
        total_items += labels.size(0)

    return float(total_loss / total_items)


def train_single_experiment(
    experiment_name: str,
    config: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    config = expand_config(config)
    set_seed(args.seed)
    data_bundle = build_data_bundle(config)
    train_loader = make_loader(
        data_bundle["splits"]["train"]["features"],
        data_bundle["splits"]["train"]["labels"],
        args.batch_size,
        True,
        args.num_workers,
        seed=args.seed,
    )
    val_loader = make_loader(
        data_bundle["splits"]["val"]["features"],
        data_bundle["splits"]["val"]["labels"],
        args.batch_size,
        False,
        args.num_workers,
        seed=args.seed,
    )
    test_loader = make_loader(
        data_bundle["splits"]["test"]["features"],
        data_bundle["splits"]["test"]["labels"],
        args.batch_size,
        False,
        args.num_workers,
        seed=args.seed,
    )

    sensor_dim = CANONICAL_SENSOR_COUNT if config["orientation"] == "16x8" else CANONICAL_FEATURE_COUNT
    model = EEGNetVariant(
        num_classes=data_bundle["num_classes"],
        sensor_dim=sensor_dim,
        orientation=config["orientation"],
        attention=config["attention"],
        attention_stage=config["attention_stage"],
        norm_type=config["norm_type"],
        eca_kernel_size=int(config["eca_kernel_size"]),
        temporal_filters=int(config["temporal_filters"]),
        spatial_filters=int(config["spatial_filters"]),
        frontend_type=config["frontend_type"],
        use_sensor_attention=bool(config["use_sensor_attention"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(data_bundle["class_weights"], dtype=torch.float32, device=device),
        label_smoothing=float(config["label_smoothing"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(2, args.patience // 3),
    )
    ema = EMAHelper(model) if config["use_ema"] else None

    history_path = output_dir / f"{experiment_name}_history.csv"
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
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                criterion=criterion,
                config=config,
                ema=ema,
            )

            if ema is not None:
                ema.apply_shadow(model)
            val_loss, val_metrics, _, _ = evaluate(model, val_loader, device, criterion)
            if ema is not None:
                ema.restore(model)

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
                if ema is not None:
                    ema.apply_shadow(model)
                    best_state = copy.deepcopy(model.state_dict())
                    ema.restore(model)
                else:
                    best_state = copy.deepcopy(model.state_dict())
            else:
                stale_epochs += 1

            print(
                f"[{experiment_name}] epoch {epoch:03d} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"lr={current_lr:.6f}",
                flush=True,
            )

            if stale_epochs >= args.patience:
                print(f"[{experiment_name}] early stopping at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError(f"No checkpoint captured for {experiment_name}.")

    model.load_state_dict(best_state)
    torch.save(best_state, output_dir / f"{experiment_name}_best.pt")

    _, val_metrics, _, _ = evaluate(model, val_loader, device, criterion)
    _, test_metrics, y_true, y_pred = evaluate(model, test_loader, device, criterion)

    label_order = list(range(data_bundle["num_classes"]))
    confusion = confusion_matrix(y_true, y_pred, labels=label_order)
    confusion_frame = pd.DataFrame(
        confusion,
        index=[f"true_{label + 1}" for label in label_order],
        columns=[f"pred_{label + 1}" for label in label_order],
    )
    confusion_frame.to_csv(output_dir / f"{experiment_name}_test_confusion_matrix.csv")

    report = classification_report(
        y_true,
        y_pred,
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        output_dir / f"{experiment_name}_test_classification_report.csv"
    )

    return {
        "experiment": experiment_name,
        "display_name": config["display_name"],
        "best_epoch": best_epoch,
        "orientation": config["orientation"],
        "scaler_mode": config["scaler_mode"],
        "attention": config["attention"],
        "attention_stage": config["attention_stage"],
        "eca_kernel_size": config["eca_kernel_size"],
        "norm_type": config["norm_type"],
        "temporal_filters": config["temporal_filters"],
        "spatial_filters": config["spatial_filters"],
        "frontend_type": config["frontend_type"],
        "use_sensor_attention": config["use_sensor_attention"],
        "label_smoothing": config["label_smoothing"],
        "input_noise_std": config["input_noise_std"],
        "sensor_dropout_prob": config["sensor_dropout_prob"],
        "use_ema": config["use_ema"],
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_weighted_f1": val_metrics["weighted_f1"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
    }


def build_report(results_frame: pd.DataFrame) -> str:
    lines = [
        "# EEGNet Variant Suite Results",
        "",
        "Fixed split: per-class 60/20/20 in original row order.",
        "",
        "| Experiment | Val Macro-F1 | Test Macro-F1 | Test Accuracy | Test Balanced Accuracy | Best Epoch | Key Changes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for _, row in results_frame.iterrows():
        key_changes = (
            f"{row['orientation']}, {row['scaler_mode']}, {row['attention']}, "
            f"stage={row['attention_stage']}, k={int(row['eca_kernel_size'])}, "
            f"{row['norm_type']}, tf={int(row['temporal_filters'])}, "
            f"sf={int(row['spatial_filters'])}, frontend={row['frontend_type']}, "
            f"sensor_attn={row['use_sensor_attention']}, ls={row['label_smoothing']}, "
            f"noise={row['input_noise_std']}, sensor_do={row['sensor_dropout_prob']}, "
            f"ema={row['use_ema']}"
        )
        lines.append(
            f"| {row['display_name']} | {row['val_macro_f1']:.4f} | {row['test_macro_f1']:.4f} | "
            f"{row['test_accuracy']:.4f} | {row['test_balanced_accuracy']:.4f} | {int(row['best_epoch'])} | {key_changes} |"
        )

    lines.append("")
    return "\n".join(lines)


def run_suite(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata = {
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "device": str(device),
        "experiments": {name: expand_config(PRESET_EXPERIMENTS[name]) for name in args.experiments},
        "note": (
            "The current baseline already uses 16x8 input. "
            "This suite covers attention design, width scaling, multi-scale frontend, "
            "sensor attention, and light robustness augmentation."
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    results = []
    started_at = time.time()
    for experiment_name in args.experiments:
        config = expand_config(PRESET_EXPERIMENTS[experiment_name])
        experiment_dir = output_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== Starting {experiment_name} on {device} ===", flush=True)
        result = train_single_experiment(
            experiment_name=experiment_name,
            config=config,
            args=args,
            output_dir=experiment_dir,
            device=device,
        )
        results.append(result)

    results_frame = pd.DataFrame(results).sort_values(
        by=["test_macro_f1", "test_accuracy"], ascending=False
    )
    results_frame.to_csv(output_dir / "metrics_summary.csv", index=False)
    (output_dir / "report.md").write_text(build_report(results_frame), encoding="utf-8")

    elapsed_minutes = (time.time() - started_at) / 60.0
    print(f"Completed {len(results)} experiments in {elapsed_minutes:.2f} minutes.", flush=True)


def main() -> None:
    args = parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()
