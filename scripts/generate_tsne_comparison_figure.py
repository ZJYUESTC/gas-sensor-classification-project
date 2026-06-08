from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import run_eegnet_variant_suite as eeg_suite
from prepare_gas_dataset import FEATURE_PREFIX, OUTPUT_DIR, PROJECT_ROOT, prepare_dataset


FIGURES_DIR = PROJECT_ROOT / "figures"
TSNE_RESULTS_DIR = PROJECT_ROOT / "results" / "tsne_comparison_models"
SEED = 42
CLASS_COLORS = [
    "#6E7FA7",
    "#7E9B74",
    "#B1856B",
    "#8F7DB5",
    "#B6A05C",
    "#6F9E9A",
]
METHOD_ORDER = [
    "Best Traditional ML",
    "EEGNet-like Baseline",
    "Final Refined EEGNet (12/24)",
]


def ensure_directories() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TSNE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.edgecolor": "#707070",
            "axes.linewidth": 0.8,
            "grid.color": "#D3D6DB",
            "grid.alpha": 0.65,
            "grid.linewidth": 0.7,
            "axes.facecolor": "#FAFAF8",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_split(name: str) -> pd.DataFrame:
    split_path = OUTPUT_DIR / f"{name}.csv"
    if not split_path.exists():
        prepare_dataset()
    return pd.read_csv(split_path)


def fit_best_extra_trees() -> tuple[np.ndarray, np.ndarray]:
    train_frame = load_split("train")
    val_frame = load_split("val")
    test_frame = load_split("test")

    feature_columns = [column for column in train_frame.columns if column.startswith(FEATURE_PREFIX)]
    train_val_frame = pd.concat([train_frame, val_frame], ignore_index=True)
    X_train_val = train_val_frame[feature_columns].to_numpy(dtype=np.float32)
    y_train_val = train_val_frame["gas_class"].to_numpy(dtype=np.int64) - 1
    X_test = test_frame[feature_columns].to_numpy(dtype=np.float32)
    y_test = test_frame["gas_class"].to_numpy(dtype=np.int64) - 1

    model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight=None,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train_val, y_train_val)
    probabilities = model.predict_proba(X_test).astype(np.float32)
    response = np.log(np.clip(probabilities, 1e-6, 1.0))
    return response, y_test


def build_args(output_dir: Path) -> Namespace:
    return Namespace(
        experiments=[],
        output_dir=output_dir,
        epochs=80,
        patience=15,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-4,
        num_workers=0,
        seed=SEED,
    )


def make_model(config: dict[str, object], num_classes: int) -> torch.nn.Module:
    sensor_dim = (
        eeg_suite.CANONICAL_SENSOR_COUNT
        if config["orientation"] == "16x8"
        else eeg_suite.CANONICAL_FEATURE_COUNT
    )
    return eeg_suite.EEGNetVariant(
        num_classes=num_classes,
        sensor_dim=sensor_dim,
        orientation=str(config["orientation"]),
        attention=str(config["attention"]),
        attention_stage=str(config["attention_stage"]),
        norm_type=str(config["norm_type"]),
        eca_kernel_size=int(config["eca_kernel_size"]),
        temporal_filters=int(config["temporal_filters"]),
        spatial_filters=int(config["spatial_filters"]),
        frontend_type=str(config["frontend_type"]),
        use_sensor_attention=bool(config["use_sensor_attention"]),
    )


def train_or_load_responses(experiment_name: str) -> tuple[np.ndarray, np.ndarray]:
    experiment_dir = TSNE_RESULTS_DIR / experiment_name
    checkpoint_path = experiment_dir / f"{experiment_name}_best.pt"
    config = eeg_suite.expand_config(eeg_suite.PRESET_EXPERIMENTS[experiment_name])

    if not checkpoint_path.exists():
        experiment_dir.mkdir(parents=True, exist_ok=True)
        args = build_args(experiment_dir)
        eeg_suite.train_single_experiment(
            experiment_name=experiment_name,
            config=config,
            args=args,
            output_dir=experiment_dir,
            device=torch.device("cpu"),
        )

    eeg_suite.set_seed(SEED)
    data_bundle = eeg_suite.build_data_bundle(config)
    model = make_model(config, data_bundle["num_classes"])
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    test_features = torch.from_numpy(data_bundle["splits"]["test"]["features"])
    test_labels = data_bundle["splits"]["test"]["labels"]
    responses: list[np.ndarray] = []

    with torch.no_grad():
        for batch_start in range(0, test_features.shape[0], 512):
            batch = test_features[batch_start : batch_start + 512]
            logits = model(batch)
            responses.append(logits.cpu().numpy().astype(np.float32))

    response_matrix = np.concatenate(responses, axis=0)
    return response_matrix, test_labels


def build_tsne_embedding(response_sets: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacked_vectors: list[np.ndarray] = []
    method_ids: list[np.ndarray] = []
    class_ids: list[np.ndarray] = []

    for method_index, method_name in enumerate(METHOD_ORDER):
        response_matrix = response_sets[method_name]
        scaled = StandardScaler().fit_transform(response_matrix)
        stacked_vectors.append(scaled)
        method_ids.append(np.full(scaled.shape[0], method_index, dtype=np.int64))

    labels_reference = np.load(TSNE_RESULTS_DIR / "labels_reference.npy")
    class_ids = [labels_reference.copy() for _ in METHOD_ORDER]

    vectors = np.vstack(stacked_vectors)
    methods = np.concatenate(method_ids)
    labels = np.concatenate(class_ids)

    embedding = TSNE(
        n_components=2,
        perplexity=35,
        learning_rate="auto",
        init="pca",
        random_state=SEED,
        max_iter=1500,
    ).fit_transform(vectors)
    return embedding, methods, labels


def save_embedding_csv(
    embedding: np.ndarray,
    methods: np.ndarray,
    labels: np.ndarray,
) -> None:
    frame = pd.DataFrame(
        {
            "tsne_1": embedding[:, 0],
            "tsne_2": embedding[:, 1],
            "method": [METHOD_ORDER[index] for index in methods],
            "gas_class": labels + 1,
        }
    )
    frame.to_csv(FIGURES_DIR / "tsne_response_space_panels_2d_points.csv", index=False)


def draw_figure(
    embedding: np.ndarray,
    methods: np.ndarray,
    labels: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.6), sharex=True, sharey=True)
    fig.subplots_adjust(bottom=0.24, left=0.055, right=0.985, wspace=0.08)

    x_margin = 1.8
    y_margin = 1.8
    x_min, x_max = embedding[:, 0].min() - x_margin, embedding[:, 0].max() + x_margin
    y_min, y_max = embedding[:, 1].min() - y_margin, embedding[:, 1].max() + y_margin

    for axis, method_index in zip(axes, range(len(METHOD_ORDER))):
        method_mask = methods == method_index
        for class_index in range(6):
            class_mask = labels == class_index
            mask = method_mask & class_mask
            axis.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=12,
                alpha=0.46,
                color=CLASS_COLORS[class_index],
                edgecolors="none",
            )
        axis.grid(True)
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_xlabel("t-SNE 1")

    axes[0].set_ylabel("t-SNE 2")

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CLASS_COLORS[index],
            markeredgecolor="none",
            markersize=8,
            label=f"Gas Class {index + 1}",
        )
        for index in range(6)
    ]

    fig.legend(
        handles=class_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
        frameon=False,
        columnspacing=1.6,
        handlelength=1.4,
    )

    png_path = FIGURES_DIR / "tsne_response_space_panels_2d.png"
    svg_path = FIGURES_DIR / "tsne_response_space_panels_2d.svg"
    fig.savefig(png_path, dpi=360, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_directories()
    apply_style()

    traditional_response, reference_labels = fit_best_extra_trees()
    np.save(TSNE_RESULTS_DIR / "labels_reference.npy", reference_labels)

    baseline_response, baseline_labels = train_or_load_responses(experiment_name="baseline")
    final_response, final_labels = train_or_load_responses(experiment_name="eca_wide_12_24")

    if not (
        np.array_equal(reference_labels, baseline_labels)
        and np.array_equal(reference_labels, final_labels)
    ):
        raise RuntimeError("Test-label alignment mismatch across compared models.")

    response_sets = {
        "Best Traditional ML": traditional_response,
        "EEGNet-like Baseline": baseline_response,
        "Final Refined EEGNet (12/24)": final_response,
    }

    embedding, methods, labels = build_tsne_embedding(response_sets)
    save_embedding_csv(embedding, methods, labels)
    draw_figure(embedding, methods, labels)


if __name__ == "__main__":
    main()
