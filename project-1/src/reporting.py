from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DATASET_URL, EXPECTED_OUTPUTS


def save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_markdown(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_class_distribution(dataframe: pd.DataFrame, output_path: Path) -> None:
    counts = dataframe["target_name"].value_counts().sort_index()
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.bar(counts.index.tolist(), counts.values, color=["#2E86C1", "#C0392B"])
    axis.set_title("Heart Failure Veri Seti Sinif Dagilimi")
    axis.set_ylabel("Ornek Sayisi")
    axis.grid(axis="y", alpha=0.25)
    for index, value in enumerate(counts.values):
        axis.text(index, value + 2, str(int(value)), ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_learning_curves(
    history_map: dict[str, pd.DataFrame],
    output_path: Path,
    selected_runs: list[str],
) -> None:
    fig, axes = plt.subplots(len(selected_runs), 2, figsize=(12, 3.5 * len(selected_runs)))
    if len(selected_runs) == 1:
        axes = np.array([axes])
    for row_index, run_name in enumerate(selected_runs):
        history = history_map[run_name]
        axes[row_index, 0].plot(history["epoch"], history["train_loss"], label="Train loss")
        axes[row_index, 0].plot(history["epoch"], history["val_loss"], label="Validation loss")
        axes[row_index, 0].set_title(f"{run_name} loss")
        axes[row_index, 0].set_xlabel("Epoch")
        axes[row_index, 0].set_ylabel("BCE loss")
        axes[row_index, 0].grid(alpha=0.25)
        axes[row_index, 0].legend()

        axes[row_index, 1].plot(history["epoch"], history["train_accuracy"], label="Train accuracy")
        axes[row_index, 1].plot(history["epoch"], history["val_accuracy"], label="Validation accuracy")
        axes[row_index, 1].set_title(f"{run_name} accuracy")
        axes[row_index, 1].set_xlabel("Epoch")
        axes[row_index, 1].set_ylabel("Accuracy")
        axes[row_index, 1].set_ylim(0.0, 1.05)
        axes[row_index, 1].grid(alpha=0.25)
        axes[row_index, 1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_data_fraction_comparison(metrics_frame: pd.DataFrame, output_path: Path) -> None:
    fraction_rows = metrics_frame.loc[
        metrics_frame["run_name"].isin(
            ["deep_scaled_l2", "deep_scaled_l2_data50", "deep_scaled_l2_data75"]
        )
    ].copy()
    fraction_rows["fraction_label"] = fraction_rows["train_fraction"].astype(float) * 100.0
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        fraction_rows["fraction_label"],
        fraction_rows["val_accuracy"],
        marker="o",
        label="Validation accuracy",
    )
    axis.plot(
        fraction_rows["fraction_label"],
        fraction_rows["test_accuracy"],
        marker="o",
        label="Test accuracy",
    )
    axis.set_title("Veri Miktari ve Accuracy Iliskisi")
    axis.set_xlabel("Train verisi yuzdesi")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.05)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(
    confusion: np.ndarray,
    labels: list[str],
    output_path: Path,
    *,
    title: str,
) -> None:
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(confusion, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels=labels)
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Tahmin")
    axis.set_ylabel("Gercek")
    axis.set_title(title)
    for row_index in range(confusion.shape[0]):
        for column_index in range(confusion.shape[1]):
            axis.text(column_index, row_index, str(int(confusion[row_index, column_index])), ha="center", va="center")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_selected_model_report(
    selected_row: pd.Series,
    metrics_frame: pd.DataFrame,
    backend_frame: pd.DataFrame,
) -> str:
    top_rows = metrics_frame.sort_values(
        ["val_balanced_accuracy", "n_steps", "parameter_count", "val_roc_auc"],
        ascending=[False, True, True, False],
    ).head(5)
    return "\n".join(
        [
            "# Secilen Model Raporu",
            "",
            "## Ozet",
            (
                "Model secimi validation balanced accuracy > dusuk n_steps > dusuk "
                "parametre sayisi > yuksek validation ROC-AUC sirasi ile yapildi."
            ),
            "",
            "## Secilen Model",
            pd.DataFrame([selected_row]).to_markdown(index=False),
            "",
            "## Ilk 5 NumPy Deneyi",
            top_rows.to_markdown(index=False),
            "",
            "## Backend Karsilastirmasi",
            backend_frame.to_markdown(index=False),
        ]
    )


def build_experiment_summary(
    metrics_frame: pd.DataFrame,
    backend_frame: pd.DataFrame,
    selected_row: pd.Series,
) -> str:
    numpy_summary = metrics_frame[
        [
            "run_name",
            "architecture",
            "use_scaler",
            "train_fraction",
            "val_accuracy",
            "val_balanced_accuracy",
            "test_accuracy",
            "test_f1",
            "test_roc_auc",
        ]
    ].sort_values(
        ["val_balanced_accuracy", "val_accuracy", "test_accuracy"],
        ascending=[False, False, False],
    )
    backend_summary = backend_frame[
        ["run_name", "backend", "test_accuracy", "test_balanced_accuracy", "test_f1", "test_roc_auc"]
    ]
    return "\n".join(
        [
            "# Deney Ozeti",
            "",
            "## Genel Sonuc",
            (
                f"Secilen model `{selected_row['run_name']}` oldu. Validation balanced accuracy "
                f"`{selected_row['val_balanced_accuracy']:.4f}`, test accuracy "
                f"`{selected_row['test_accuracy']:.4f}` ve test ROC-AUC "
                f"`{selected_row['test_roc_auc']:.4f}` olarak olculdu."
            ),
            "",
            "## NumPy Deneyleri",
            numpy_summary.to_markdown(index=False),
            "",
            "## Backend Karsilastirmasi",
            backend_summary.to_markdown(index=False),
        ]
    )


def build_traceability_matrix() -> str:
    lines = [
        "# Izlenebilirlik Matrisi",
        "",
        "| PDF Gereksinimi | Kanit | Durum |",
        "| --- | --- | --- |",
        f"| Ikili siniflandirma verisi | `{DATASET_URL}` ve `data/raw/heart_failure_clinical_records_dataset.csv` | Tamam |",
        "| Veri analizi ve on isleme | `outputs/figures/class_distribution.png`, `src/dataset.py` | Tamam |",
        "| Laboratuvar temel modeli | `baseline_raw`, `baseline_scaled`, `src/numpy_mlp.py` | Tamam |",
        "| Overfitting / underfitting incelemesi | `outputs/figures/numpy_learning_curves.png` | Tamam |",
        "| Cok katmanli model / veri miktari / regülarizasyon | `deep_scaled*`, `outputs/figures/data_fraction_comparison.png` | Tamam |",
        "| Class tabanli temiz yapi | `src/numpy_mlp.py`, `src/sklearn_backend.py`, `src/pytorch_backend.py` | Tamam |",
        "| Model secimi | `outputs/tables/model_selection.csv`, `outputs/reports/selected_model_report.md` | Tamam |",
        "| sklearn ve PyTorch tekrar yazimi | `src/sklearn_backend.py`, `src/pytorch_backend.py`, `outputs/tables/backend_comparison_metrics.csv` | Tamam |",
        "| Confusion matrix ve temel metrikler | `outputs/figures/confusion_matrix_selected_*`, `outputs/tables/*.csv` | Tamam |",
        "| Ayni split / agirlik / SGD | `data/splits/split_manifest.json`, `data/weights/*.npz`, backend modulleri | Tamam |",
        "| BCE loss | `src/numpy_mlp.py`, `src/pytorch_backend.py`, `src/metrics.py` | Tamam |",
        "| Uygun repo hiyerarsisi | `src/`, `data/`, `outputs/`, `tests/` | Tamam |",
        "| IMRAD README | `README.md` | Tamam |",
    ]
    return "\n".join(lines)


def build_run_summary(
    metrics_frame: pd.DataFrame,
    backend_frame: pd.DataFrame,
    selected_row: pd.Series,
) -> dict[str, object]:
    return {
        "selected_model": selected_row.to_dict(),
        "selection_rule": (
            "validation_balanced_accuracy_desc_then_n_steps_asc_"
            "then_parameter_count_asc_then_val_roc_auc_desc"
        ),
        "numpy_runs": metrics_frame.to_dict(orient="records"),
        "backend_runs": backend_frame.to_dict(orient="records"),
        "expected_outputs": [str(path) for path in EXPECTED_OUTPUTS],
    }
