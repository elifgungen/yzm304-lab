from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    BOOLEAN_COLUMNS,
    CONTINUOUS_COLUMNS,
    DATASET_URL,
    GLOBAL_SEED,
    RAW_DIR,
    SPLIT_DIR,
    TARGET_COLUMN,
    TARGET_NAMES,
    TEST_SIZE,
    VALIDATION_SIZE_WITHIN_TRAIN,
)


@dataclass
class PreparedSplit:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    scaler: StandardScaler | None
    train_indices: list[int]


@dataclass
class DatasetBundle:
    dataframe: pd.DataFrame
    feature_names: list[str]
    target_names: list[str]
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    manifest: dict[str, object]


def _download_dataframe() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = RAW_DIR / "heart_failure_clinical_records_dataset.csv"
    if not dataset_path.exists():
        with urlopen(DATASET_URL) as response:
            dataset_path.write_bytes(response.read())
    dataframe = pd.read_csv(dataset_path)
    dataframe = dataframe.copy()
    dataframe["target_name"] = dataframe[TARGET_COLUMN].map(
        {0: TARGET_NAMES[0], 1: TARGET_NAMES[1]}
    )
    return dataframe


def _build_split_manifest(dataframe: pd.DataFrame) -> dict[str, object]:
    target = dataframe[TARGET_COLUMN]
    train_full_idx, test_idx = train_test_split(
        dataframe.index.to_numpy(),
        test_size=TEST_SIZE,
        random_state=GLOBAL_SEED,
        stratify=target,
    )
    train_idx, val_idx = train_test_split(
        train_full_idx,
        test_size=VALIDATION_SIZE_WITHIN_TRAIN,
        random_state=GLOBAL_SEED,
        stratify=target.loc[train_full_idx],
    )
    train_fraction_indices = _build_train_fraction_indices(
        target.to_numpy(dtype=int),
        np.sort(train_idx),
        fractions=(0.50, 0.75, 1.0),
        random_state=GLOBAL_SEED,
    )
    return {
        "seed": GLOBAL_SEED,
        "dataset_url": DATASET_URL,
        "continuous_columns": list(CONTINUOUS_COLUMNS),
        "boolean_columns": list(BOOLEAN_COLUMNS),
        "class_balance_total": _class_balance(target.to_numpy(dtype=int)),
        "split_sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "splits": {
            "train": sorted(int(index) for index in train_idx),
            "validation": sorted(int(index) for index in val_idx),
            "test": sorted(int(index) for index in test_idx),
        },
        "train_fraction_sizes": {
            key: int(len(indices)) for key, indices in train_fraction_indices.items()
        },
        "train_fraction_indices": {
            key: [int(index) for index in indices] for key, indices in train_fraction_indices.items()
        },
        "train_fraction_class_balance": {
            key: _class_balance(target.iloc[indices].to_numpy(dtype=int))
            for key, indices in train_fraction_indices.items()
        },
    }


def _class_balance(y_values: np.ndarray) -> dict[int, int]:
    labels, counts = np.unique(y_values, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def _build_train_fraction_indices(
    y_values: np.ndarray,
    train_indices: np.ndarray,
    *,
    fractions: tuple[float, ...],
    random_state: int,
) -> dict[str, list[int]]:
    rng = np.random.default_rng(random_state)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    per_label_orders = {
        int(label): rng.permutation(train_indices[y_values[train_indices] == label])
        for label in np.unique(y_values[train_indices])
    }

    outputs: dict[str, list[int]] = {}
    for fraction in fractions:
        if fraction >= 1.0:
            outputs[f"{fraction:.2f}"] = np.sort(train_indices).astype(int).tolist()
            continue
        subset_parts: list[np.ndarray] = []
        for _, shuffled in per_label_orders.items():
            take_count = max(1, int(round(len(shuffled) * fraction)))
            subset_parts.append(shuffled[:take_count])
        outputs[f"{fraction:.2f}"] = np.sort(np.concatenate(subset_parts)).astype(int).tolist()
    return outputs


def export_dataset_artifacts() -> DatasetBundle:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    dataframe = _download_dataframe()
    manifest = _build_split_manifest(dataframe)

    metadata = {
        "dataset_name": "Heart Failure Clinical Records",
        "dataset_url": DATASET_URL,
        "samples": int(len(dataframe)),
        "features": 12,
        "target_column": TARGET_COLUMN,
        "target_names": list(TARGET_NAMES),
    }
    (RAW_DIR / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (SPLIT_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    split_frames: dict[str, pd.DataFrame] = {}
    for split_name, indices in manifest["splits"].items():
        split_frame = dataframe.loc[indices].copy().reset_index(names="row_id")
        split_frames[split_name] = split_frame
        split_frame.to_csv(SPLIT_DIR / f"{split_name}.csv", index=False)

    return DatasetBundle(
        dataframe=dataframe.reset_index(names="row_id"),
        feature_names=[*CONTINUOUS_COLUMNS, *BOOLEAN_COLUMNS],
        target_names=list(TARGET_NAMES),
        train_df=split_frames["train"],
        val_df=split_frames["validation"],
        test_df=split_frames["test"],
        manifest=manifest,
    )


def prepare_features(
    bundle: DatasetBundle,
    feature_names: list[str],
    *,
    use_scaler: bool,
    train_fraction: float,
) -> PreparedSplit:
    feature_names = list(feature_names)
    train_df = bundle.train_df.copy()
    if train_fraction < 1.0:
        fraction_key = f"{train_fraction:.2f}"
        manifest_indices = bundle.manifest["train_fraction_indices"][fraction_key]
        train_df = train_df.loc[train_df["row_id"].isin(manifest_indices)].copy()
        train_df = train_df.sort_values("row_id").reset_index(drop=True)

    val_df = bundle.val_df.copy()
    test_df = bundle.test_df.copy()

    X_train = train_df[feature_names].astype(float)
    X_val = val_df[feature_names].astype(float)
    X_test = test_df[feature_names].astype(float)
    y_train = train_df[TARGET_COLUMN].astype(int)
    y_val = val_df[TARGET_COLUMN].astype(int)
    y_test = test_df[TARGET_COLUMN].astype(int)

    scaler = None
    if use_scaler:
        scaler = StandardScaler()
        continuous_columns = list(CONTINUOUS_COLUMNS)
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled.loc[:, continuous_columns] = scaler.fit_transform(X_train[continuous_columns])
        X_val_scaled.loc[:, continuous_columns] = scaler.transform(X_val[continuous_columns])
        X_test_scaled.loc[:, continuous_columns] = scaler.transform(X_test[continuous_columns])
        X_train = X_train_scaled
        X_val = X_val_scaled
        X_test = X_test_scaled

    return PreparedSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        scaler=scaler,
        train_indices=train_df["row_id"].astype(int).tolist(),
    )
