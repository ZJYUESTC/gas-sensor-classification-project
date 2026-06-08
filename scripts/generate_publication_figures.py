from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "figures"


TRADITIONAL_MODELS = [
    {
        "label": "Extra Trees",
        "val_macro_f1": 0.7798,
        "test_macro_f1": 0.6921,
        "test_accuracy": 0.6908,
        "test_balanced_accuracy": 0.7210,
        "mflops": 0.003005,
    },
    {
        "label": "Logistic Regression",
        "val_macro_f1": 0.7875,
        "test_macro_f1": 0.6911,
        "test_accuracy": 0.6998,
        "test_balanced_accuracy": 0.7200,
        "mflops": 0.001536,
    },
    {
        "label": "SVM (RBF)",
        "val_macro_f1": 0.7933,
        "test_macro_f1": 0.6739,
        "test_accuracy": 0.6707,
        "test_balanced_accuracy": 0.7022,
        "mflops": 0.554415,
    },
    {
        "label": "Random Forest",
        "val_macro_f1": 0.7585,
        "test_macro_f1": 0.6471,
        "test_accuracy": 0.6488,
        "test_balanced_accuracy": 0.6774,
        "mflops": 0.002490,
    },
    {
        "label": "kNN",
        "val_macro_f1": 0.7371,
        "test_macro_f1": 0.6317,
        "test_accuracy": 0.6338,
        "test_balanced_accuracy": 0.6633,
        "mflops": 3.195369,
    },
]


DEEP_BASELINES = [
    {
        "label": "EEGNet-like",
        "val_macro_f1": 0.7690,
        "test_macro_f1": 0.7795,
        "test_accuracy": 0.7458,
        "test_balanced_accuracy": 0.7795,
        "mflops": 0.015718,
        "params": 766,
    },
    {
        "label": "LeNet1D",
        "val_macro_f1": 0.8155,
        "test_macro_f1": 0.6970,
        "test_accuracy": 0.6944,
        "test_balanced_accuracy": 0.7135,
        "mflops": 0.201078,
        "params": 67006,
    },
    {
        "label": "MLP",
        "val_macro_f1": 0.8191,
        "test_macro_f1": 0.6917,
        "test_accuracy": 0.6926,
        "test_balanced_accuracy": 0.6997,
        "mflops": 0.149894,
        "params": 75334,
    },
    {
        "label": "MobileNetV2",
        "val_macro_f1": 0.7341,
        "test_macro_f1": 0.6340,
        "test_accuracy": 0.6420,
        "test_balanced_accuracy": 0.6363,
        "mflops": 15.940870,
        "params": 695078,
    },
    {
        "label": "1D-CNN",
        "val_macro_f1": 0.7822,
        "test_macro_f1": 0.5543,
        "test_accuracy": 0.5803,
        "test_balanced_accuracy": 0.5775,
        "mflops": 2.459142,
        "params": 32326,
    },
    {
        "label": "ShuffleNetV2",
        "val_macro_f1": 0.7241,
        "test_macro_f1": 0.5326,
        "test_accuracy": 0.5364,
        "test_balanced_accuracy": 0.5500,
        "mflops": 21.626886,
        "params": 347510,
    },
]


REFINED_EEGNET_SERIES = [
    {
        "label": "EEGNet-like",
        "val_macro_f1": 0.7690,
        "test_macro_f1": 0.7795,
        "test_accuracy": 0.7458,
        "test_balanced_accuracy": 0.7795,
    },
    {
        "label": "Refined EEGNet (12/24)",
        "val_macro_f1": 0.8113,
        "test_macro_f1": 0.7812,
        "test_accuracy": 0.7655,
        "test_balanced_accuracy": 0.8009,
    },
    {
        "label": "Refined EEGNet (12/24) + Noise 0.01",
        "val_macro_f1": 0.8291,
        "test_macro_f1": 0.7573,
        "test_accuracy": 0.7415,
        "test_balanced_accuracy": 0.7813,
    },
    {
        "label": "Refined EEGNet (12/24), Block-1 Variant",
        "val_macro_f1": 0.7975,
        "test_macro_f1": 0.7481,
        "test_accuracy": 0.7382,
        "test_balanced_accuracy": 0.7751,
    },
]


REFINEMENT_SWEEP = [
    {
        "label": "Refined EEGNet (10/20)",
        "val_macro_f1": 0.8186,
        "test_macro_f1": 0.5438,
        "test_accuracy": 0.5526,
        "test_balanced_accuracy": 0.6176,
    },
    {
        "label": "Refined EEGNet (12/24)",
        "val_macro_f1": 0.8113,
        "test_macro_f1": 0.7812,
        "test_accuracy": 0.7655,
        "test_balanced_accuracy": 0.8009,
    },
    {
        "label": "Refined EEGNet (14/28)",
        "val_macro_f1": 0.8215,
        "test_macro_f1": 0.7132,
        "test_accuracy": 0.7142,
        "test_balanced_accuracy": 0.7655,
    },
]


SELECTED_MODELS = [
    {
        "label": "Extra Trees",
        "group": "Traditional ML",
        "test_macro_f1": 0.6921,
        "test_accuracy": 0.6908,
        "test_balanced_accuracy": 0.7210,
    },
    {
        "label": "Logistic Regression",
        "group": "Traditional ML",
        "test_macro_f1": 0.6911,
        "test_accuracy": 0.6998,
        "test_balanced_accuracy": 0.7200,
    },
    {
        "label": "SVM (RBF)",
        "group": "Traditional ML",
        "test_macro_f1": 0.6739,
        "test_accuracy": 0.6707,
        "test_balanced_accuracy": 0.7022,
    },
    {
        "label": "EEGNet-like",
        "group": "Deep Learning",
        "test_macro_f1": 0.7795,
        "test_accuracy": 0.7458,
        "test_balanced_accuracy": 0.7795,
    },
    {
        "label": "LeNet1D",
        "group": "Deep Learning",
        "test_macro_f1": 0.6970,
        "test_accuracy": 0.6944,
        "test_balanced_accuracy": 0.7135,
    },
    {
        "label": "MLP",
        "group": "Deep Learning",
        "test_macro_f1": 0.6917,
        "test_accuracy": 0.6926,
        "test_balanced_accuracy": 0.6997,
    },
    {
        "label": "Refined EEGNet (12/24)",
        "group": "Deep Learning",
        "test_macro_f1": 0.7812,
        "test_accuracy": 0.7655,
        "test_balanced_accuracy": 0.8009,
    },
]


COLORS = [
    "#6E7FA7",
    "#7E9B74",
    "#B1856B",
    "#8F7DB5",
    "#B6A05C",
    "#6F9E9A",
    "#C07C91",
    "#808080",
]
GROUP_COLORS = {
    "Traditional ML": "#7A8AAE",
    "Deep Learning": "#8DAA86",
}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def apply_global_style() -> None:
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


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=360, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def radar_angles(num_axes: int) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False)
    return np.concatenate([angles, [angles[0]]])


def radar_values(item: dict[str, float], keys: list[str]) -> np.ndarray:
    values = [item[key] for key in keys]
    return np.array(values + [values[0]])


def add_bottom_legend(fig: plt.Figure, handles, ncol: int) -> None:
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=ncol,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.4,
    )


def make_radar_chart(stem: str, items: list[dict[str, float]], palette: list[str]) -> None:
    metric_keys = ["val_macro_f1", "test_macro_f1", "test_accuracy", "test_balanced_accuracy"]
    metric_labels = ["Val Macro-F1", "Test Macro-F1", "Test Accuracy", "Balanced Accuracy"]
    angles = radar_angles(len(metric_keys))

    fig, ax = plt.subplots(figsize=(8.8, 7.2), subplot_kw={"projection": "polar"})
    fig.subplots_adjust(bottom=0.2)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.5, 0.9)
    ax.set_yticks([0.55, 0.65, 0.75, 0.85])
    ax.set_yticklabels(["0.55", "0.65", "0.75", "0.85"])
    ax.grid(True)

    handles = []
    for index, item in enumerate(items):
        color = palette[index % len(palette)]
        values = radar_values(item, metric_keys)
        ax.plot(angles, values, color=color, linewidth=2.0)
        ax.fill(angles, values, color=color, alpha=0.10)
        handles.append(Line2D([0], [0], color=color, lw=2.4, label=item["label"]))

    add_bottom_legend(fig, handles, ncol=min(3, len(items)))
    save_figure(fig, stem)


def make_winner_radar(stem: str) -> None:
    winners = [
        {
            "label": "Best Traditional ML",
            "val_macro_f1": 0.7798,
            "test_macro_f1": 0.6921,
            "test_accuracy": 0.6908,
            "test_balanced_accuracy": 0.7210,
        },
        {
            "label": "Best Deep Baseline",
            "val_macro_f1": 0.7690,
            "test_macro_f1": 0.7795,
            "test_accuracy": 0.7458,
            "test_balanced_accuracy": 0.7795,
        },
        {
            "label": "Final Refined EEGNet",
            "val_macro_f1": 0.8113,
            "test_macro_f1": 0.7812,
            "test_accuracy": 0.7655,
            "test_balanced_accuracy": 0.8009,
        },
    ]
    make_radar_chart(stem, winners, [COLORS[4], COLORS[0], COLORS[1]])


def make_selected_models_panels_2d(stem: str) -> None:
    metric_keys = ["test_macro_f1", "test_accuracy", "test_balanced_accuracy"]
    metric_labels = ["Test Macro-F1", "Test Accuracy", "Balanced Accuracy"]
    ordered_items = sorted(
        SELECTED_MODELS,
        key=lambda item: (item["test_macro_f1"], item["test_accuracy"]),
        reverse=True,
    )
    y_positions = np.arange(len(ordered_items))
    colors = [GROUP_COLORS[item["group"]] for item in ordered_items]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 6.5), sharey=True)
    fig.subplots_adjust(bottom=0.22, left=0.19, right=0.985, wspace=0.08)

    for index, (axis, metric_key, metric_label) in enumerate(
        zip(axes, metric_keys, metric_labels, strict=False)
    ):
        values = [item[metric_key] for item in ordered_items]
        axis.barh(
            y_positions,
            values,
            color=colors,
            alpha=0.88,
            edgecolor="#5C5C5C",
            linewidth=0.4,
            height=0.68,
        )
        axis.grid(True, axis="x")
        axis.set_axisbelow(True)
        axis.set_xlim(0.62, 0.82)
        axis.set_xlabel(metric_label)
        axis.invert_yaxis()
        if index == 0:
            axis.set_yticks(y_positions)
            axis.set_yticklabels([item["label"] for item in ordered_items])
        else:
            axis.tick_params(axis="y", length=0, labelleft=False)

    handles = [
        Line2D([0], [0], color=color, lw=7, label=group)
        for group, color in GROUP_COLORS.items()
    ]
    add_bottom_legend(fig, handles, ncol=2)
    save_figure(fig, stem)


def make_efficiency_3d_scatter(stem: str) -> None:
    fig = plt.figure(figsize=(12.0, 7.8))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.27, left=0.04, right=0.98)

    handles = []

    all_points = []
    for item in TRADITIONAL_MODELS:
        all_points.append((item, GROUP_COLORS["Traditional ML"]))
    for item in DEEP_BASELINES:
        all_points.append((item, GROUP_COLORS["Deep Learning"]))

    for item, color in all_points:
        ax.scatter(
            math.log10(item["mflops"] + 1e-6),
            item["test_accuracy"],
            item["test_macro_f1"],
            s=62,
            color=color,
            edgecolors="#4D4D4D",
            linewidths=0.5,
            alpha=0.92,
            label=item["label"],
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="#4D4D4D",
                markersize=7.5,
                label=item["label"],
            )
        )

    ax.set_xlabel("log10 Inference Cost (MFLOPs)", labelpad=12)
    ax.set_ylabel("Test Accuracy", labelpad=12)
    ax.set_zlabel("Test Macro-F1", labelpad=10)
    ax.set_xlim(-3.2, 1.6)
    ax.set_ylim(0.52, 0.76)
    ax.set_zlim(0.52, 0.80)
    ax.view_init(elev=23, azim=-57)

    add_bottom_legend(fig, handles, ncol=4)
    save_figure(fig, stem)


def main() -> None:
    ensure_output_dir()
    apply_global_style()

    make_radar_chart("traditional_models_radar", TRADITIONAL_MODELS, COLORS[:5])
    make_radar_chart("deep_models_radar", DEEP_BASELINES + [REFINED_EEGNET_SERIES[1]], COLORS[:7])
    make_selected_models_panels_2d("overall_selected_models_panels_2d")
    make_efficiency_3d_scatter("efficiency_tradeoff_3d_scatter")
    make_radar_chart("eegnet_refinement_radar", REFINED_EEGNET_SERIES, [COLORS[0], COLORS[1], COLORS[5], COLORS[3]])
    make_radar_chart("width_refinement_radar", REFINEMENT_SWEEP, [COLORS[2], COLORS[1], COLORS[4]])
    make_winner_radar("winner_profile_radar")


if __name__ == "__main__":
    main()
