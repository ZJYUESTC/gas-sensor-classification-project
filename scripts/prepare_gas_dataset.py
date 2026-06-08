from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "gas_sensor_array_drift_at_different_concentrations_uci_270.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gas_sensor_array_drift_at_different_concentrations_uci_270"
)
SPLIT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}
FEATURE_PREFIX = "Feature"


def parse_raw_dataset(raw_data_path: Path) -> pd.DataFrame:
    with raw_data_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        feature_columns = [column for column in header[1:] if column]
        expected_feature_count = len(feature_columns)

        rows: list[dict[str, Any]] = []
        for row_id, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            if line.endswith(","):
                line = line[:-1]

            tokens = line.split(",")
            if len(tokens) != expected_feature_count + 1:
                raise ValueError(
                    f"Row {row_id} has {len(tokens) - 1} features, expected "
                    f"{expected_feature_count}."
                )

            gas_class_text, concentration_text = tokens[0].split(";", 1)
            row: dict[str, Any] = {
                "row_id": row_id,
                "gas_class": int(gas_class_text),
                "concentration": float(concentration_text),
            }

            for expected_index, feature_token in enumerate(tokens[1:], start=1):
                feature_index_text, feature_value_text = feature_token.split(":", 1)
                feature_index = int(feature_index_text)
                if feature_index != expected_index:
                    raise ValueError(
                        f"Row {row_id} feature index {feature_index} does not match "
                        f"expected index {expected_index}."
                    )
                row[f"{FEATURE_PREFIX}{feature_index}"] = float(feature_value_text)

            rows.append(row)

    dataset = pd.DataFrame(rows)
    dataset.sort_values("row_id", inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    return dataset


def assign_fixed_splits(
    dataset: pd.DataFrame, raw_data_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    split_by_index: dict[int, str] = {}
    class_counts: dict[str, dict[str, Any]] = {}

    for gas_class, class_frame in dataset.groupby("gas_class", sort=True):
        ordered = class_frame.sort_values("row_id")
        ordered_indices = ordered.index.to_list()
        sample_count = len(ordered_indices)

        train_end = int(sample_count * SPLIT_RATIOS["train"])
        val_end = int(sample_count * (SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"]))

        if train_end == 0 or val_end <= train_end or val_end >= sample_count:
            raise ValueError(
                f"Class {gas_class} cannot be split with ratios {SPLIT_RATIOS}."
            )

        train_indices = ordered_indices[:train_end]
        val_indices = ordered_indices[train_end:val_end]
        test_indices = ordered_indices[val_end:]

        for index in train_indices:
            split_by_index[index] = "train"
        for index in val_indices:
            split_by_index[index] = "val"
        for index in test_indices:
            split_by_index[index] = "test"

        class_counts[str(gas_class)] = {
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_indices),
            "train_row_range": [
                int(dataset.loc[train_indices[0], "row_id"]),
                int(dataset.loc[train_indices[-1], "row_id"]),
            ],
            "val_row_range": [
                int(dataset.loc[val_indices[0], "row_id"]),
                int(dataset.loc[val_indices[-1], "row_id"]),
            ],
            "test_row_range": [
                int(dataset.loc[test_indices[0], "row_id"]),
                int(dataset.loc[test_indices[-1], "row_id"]),
            ],
        }

    prepared = dataset.copy()
    prepared["split"] = prepared.index.map(split_by_index)
    prepared.sort_values(["split", "gas_class", "row_id"], inplace=True)
    prepared.reset_index(drop=True, inplace=True)

    overall_counts = {
        split_name: int((prepared["split"] == split_name).sum())
        for split_name in ("train", "val", "test")
    }
    summary = {
        "raw_data_path": str(raw_data_path.resolve()),
        "num_samples": int(len(prepared)),
        "num_features": int(
            len([column for column in prepared.columns if column.startswith(FEATURE_PREFIX)])
        ),
        "split_rule": "Per class, preserve original row order and slice 60%/20%/20%.",
        "split_ratios": SPLIT_RATIOS,
        "overall_counts": overall_counts,
        "class_counts": class_counts,
    }
    return prepared, summary


def write_split_artifacts(dataset: pd.DataFrame, summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = [
        column for column in dataset.columns if column.startswith(FEATURE_PREFIX)
    ]
    ordered_columns = ["row_id", "gas_class", "concentration", "split", *feature_columns]
    dataset = dataset[ordered_columns]

    dataset.to_csv(output_dir / "full_dataset_with_splits.csv", index=False)
    for split_name in ("train", "val", "test"):
        split_frame = dataset.loc[dataset["split"] == split_name].copy()
        split_frame.to_csv(output_dir / f"{split_name}.csv", index=False)

    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)


def prepare_dataset(
    raw_data_path: Path = RAW_DATA_PATH, output_dir: Path = OUTPUT_DIR
) -> dict[str, Path]:
    dataset = parse_raw_dataset(raw_data_path)
    prepared, summary = assign_fixed_splits(dataset, raw_data_path)
    write_split_artifacts(prepared, summary, output_dir)
    return {
        "full_dataset": output_dir / "full_dataset_with_splits.csv",
        "train": output_dir / "train.csv",
        "val": output_dir / "val.csv",
        "test": output_dir / "test.csv",
        "summary": output_dir / "split_summary.json",
    }


def main() -> None:
    output_paths = prepare_dataset()
    print("Prepared dataset artifacts:")
    for name, path in output_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
