from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from prepare_gas_dataset import FEATURE_PREFIX, OUTPUT_DIR, PROJECT_ROOT, prepare_dataset

RESULTS_DIR = PROJECT_ROOT / "results" / "gas_baselines"


def build_logistic_regression(params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=params["C"],
                    class_weight=params["class_weight"],
                    max_iter=5000,
                ),
            ),
        ]
    )


def build_knn(params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=params["n_neighbors"],
                    weights=params["weights"],
                ),
            ),
        ]
    )


def build_svm(params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    C=params["C"],
                    gamma=params["gamma"],
                    kernel="rbf",
                ),
            ),
        ]
    )


def build_random_forest(params: dict[str, Any]) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        class_weight=params["class_weight"],
        n_jobs=-1,
        random_state=42,
    )


def build_extra_trees(params: dict[str, Any]) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        class_weight=params["class_weight"],
        n_jobs=-1,
        random_state=42,
    )


MODEL_SPECS = [
    {
        "name": "logistic_regression",
        "builder": build_logistic_regression,
        "param_grid": {
            "C": [0.1, 1.0, 10.0],
            "class_weight": [None, "balanced"],
        },
    },
    {
        "name": "knn",
        "builder": build_knn,
        "param_grid": {
            "n_neighbors": [3, 5, 9],
            "weights": ["uniform", "distance"],
        },
    },
    {
        "name": "svm_rbf",
        "builder": build_svm,
        "param_grid": {
            "C": [1.0, 10.0],
            "gamma": ["scale", 0.01],
        },
    },
    {
        "name": "random_forest",
        "builder": build_random_forest,
        "param_grid": {
            "n_estimators": [300],
            "max_depth": [None, 30],
            "min_samples_leaf": [1, 2],
            "class_weight": [None, "balanced"],
        },
    },
    {
        "name": "extra_trees",
        "builder": build_extra_trees,
        "param_grid": {
            "n_estimators": [300],
            "max_depth": [None, 30],
            "min_samples_leaf": [1, 2],
            "class_weight": [None, "balanced"],
        },
    },
]


def iter_param_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(param_grid.keys())
    combinations = []
    for values in product(*(param_grid[key] for key in keys)):
        combinations.append(dict(zip(keys, values)))
    return combinations


def compute_metrics(y_true: pd.Series, y_pred: Any) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def load_split(split_name: str) -> pd.DataFrame:
    split_path = OUTPUT_DIR / f"{split_name}.csv"
    if not split_path.exists():
        prepare_dataset()
    return pd.read_csv(split_path)


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith(FEATURE_PREFIX)]


def evaluate_models() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_frame = load_split("train")
    val_frame = load_split("val")
    test_frame = load_split("test")

    feature_columns = select_feature_columns(train_frame)

    X_train = train_frame[feature_columns]
    y_train = train_frame["gas_class"]
    X_val = val_frame[feature_columns]
    y_val = val_frame["gas_class"]
    X_test = test_frame[feature_columns]
    y_test = test_frame["gas_class"]

    train_val_frame = pd.concat([train_frame, val_frame], ignore_index=True)
    X_train_val = train_val_frame[feature_columns]
    y_train_val = train_val_frame["gas_class"]

    split_summary = json.loads((OUTPUT_DIR / "split_summary.json").read_text(encoding="utf-8"))
    (RESULTS_DIR / "split_summary.json").write_text(
        json.dumps(split_summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    labels = sorted(y_train.unique().tolist())
    experiment_rows: list[dict[str, Any]] = []
    best_params_by_model: dict[str, dict[str, Any]] = {}

    for model_spec in MODEL_SPECS:
        best_result: dict[str, Any] | None = None

        for params in iter_param_grid(model_spec["param_grid"]):
            estimator = model_spec["builder"](params)
            started_at = perf_counter()
            estimator.fit(X_train, y_train)
            training_seconds = perf_counter() - started_at

            val_predictions = estimator.predict(X_val)
            val_metrics = compute_metrics(y_val, val_predictions)
            candidate = {
                "params": params,
                "estimator": estimator,
                "training_seconds": training_seconds,
                **{f"val_{name}": value for name, value in val_metrics.items()},
            }

            if best_result is None or (
                candidate["val_macro_f1"],
                candidate["val_accuracy"],
                candidate["val_balanced_accuracy"],
            ) > (
                best_result["val_macro_f1"],
                best_result["val_accuracy"],
                best_result["val_balanced_accuracy"],
            ):
                best_result = candidate

        assert best_result is not None
        best_params_by_model[model_spec["name"]] = best_result["params"]

        final_estimator = model_spec["builder"](best_result["params"])
        final_started_at = perf_counter()
        final_estimator.fit(X_train_val, y_train_val)
        final_training_seconds = perf_counter() - final_started_at
        test_predictions = final_estimator.predict(X_test)
        test_metrics = compute_metrics(y_test, test_predictions)

        confusion = confusion_matrix(y_test, test_predictions, labels=labels)
        confusion_frame = pd.DataFrame(
            confusion,
            index=[f"true_{label}" for label in labels],
            columns=[f"pred_{label}" for label in labels],
        )
        confusion_frame.to_csv(
            RESULTS_DIR / f"test_confusion_matrix_{model_spec['name']}.csv"
        )

        report = classification_report(
            y_test,
            test_predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).transpose().to_csv(
            RESULTS_DIR / f"test_classification_report_{model_spec['name']}.csv"
        )

        experiment_rows.append(
            {
                "model": model_spec["name"],
                "best_params": json.dumps(best_result["params"], ensure_ascii=True, sort_keys=True),
                "val_accuracy": best_result["val_accuracy"],
                "val_macro_f1": best_result["val_macro_f1"],
                "val_weighted_f1": best_result["val_weighted_f1"],
                "val_balanced_accuracy": best_result["val_balanced_accuracy"],
                "train_fit_seconds": best_result["training_seconds"],
                "final_fit_seconds_train_plus_val": final_training_seconds,
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            }
        )

    results_frame = pd.DataFrame(experiment_rows).sort_values(
        by=["test_macro_f1", "test_accuracy"], ascending=False
    )
    results_frame.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)

    with (RESULTS_DIR / "best_params.json").open("w", encoding="utf-8") as handle:
        json.dump(best_params_by_model, handle, ensure_ascii=True, indent=2)

    write_markdown_report(results_frame)


def write_markdown_report(results_frame: pd.DataFrame) -> None:
    lines = [
        "# Gas Baseline Results",
        "",
        "Fixed split: per class, preserve original row order and slice 60%/20%/20% into train/val/test.",
        "",
        "## Model Summary",
        "",
        "| Model | Val Macro-F1 | Test Macro-F1 | Test Accuracy | Test Balanced Accuracy | Best Params |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for _, row in results_frame.iterrows():
        lines.append(
            "| {model} | {val_macro_f1:.4f} | {test_macro_f1:.4f} | {test_accuracy:.4f} | "
            "{test_balanced_accuracy:.4f} | `{best_params}` |".format(**row.to_dict())
        )

    (RESULTS_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    evaluate_models()
    print(f"Saved experiment outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
