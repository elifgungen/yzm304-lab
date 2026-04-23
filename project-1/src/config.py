from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SPLIT_DIR = DATA_DIR / "splits"
WEIGHT_DIR = DATA_DIR / "weights"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"

GLOBAL_SEED = 42
TARGET_COLUMN = "DEATH_EVENT"
TARGET_NAMES = ("survived", "death")
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/"
    "heart_failure_clinical_records_dataset.csv"
)
TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_TRAIN = 0.25

CONTINUOUS_COLUMNS = (
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "time",
)
BOOLEAN_COLUMNS = (
    "anaemia",
    "diabetes",
    "high_blood_pressure",
    "sex",
    "smoking",
)
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + BOOLEAN_COLUMNS


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    architecture: tuple[int, ...]
    hidden_activation: str
    learning_rate: float
    l2_lambda: float
    use_scaler: bool
    train_fraction: float
    epochs: int
    notes: str

    @property
    def architecture_label(self) -> str:
        return "-".join(str(unit) for unit in self.architecture)

    @property
    def parameter_count(self) -> int:
        total = 0
        for fan_in, fan_out in zip(self.architecture[:-1], self.architecture[1:]):
            total += fan_in * fan_out + fan_out
        return total


NUMPY_EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        name="baseline_raw",
        architecture=(12, 8, 1),
        hidden_activation="sigmoid",
        learning_rate=0.05,
        l2_lambda=0.0,
        use_scaler=False,
        train_fraction=1.0,
        epochs=1200,
        notes="Laboratuvar varsayimi temel model, ham ozelliklerle egitildi.",
    ),
    ExperimentSpec(
        name="baseline_scaled",
        architecture=(12, 8, 1),
        hidden_activation="sigmoid",
        learning_rate=0.05,
        l2_lambda=0.0,
        use_scaler=True,
        train_fraction=1.0,
        epochs=1200,
        notes="Temel model standardizasyon ile tekrar egitildi.",
    ),
    ExperimentSpec(
        name="wide_scaled",
        architecture=(12, 16, 1),
        hidden_activation="sigmoid",
        learning_rate=0.03,
        l2_lambda=0.0,
        use_scaler=True,
        train_fraction=1.0,
        epochs=1400,
        notes="Tek gizli katman korunup nöron sayisi artirildi.",
    ),
    ExperimentSpec(
        name="deep_scaled",
        architecture=(12, 24, 12, 1),
        hidden_activation="relu",
        learning_rate=0.01,
        l2_lambda=0.0,
        use_scaler=True,
        train_fraction=1.0,
        epochs=1400,
        notes="Iki gizli katmanli derin model.",
    ),
    ExperimentSpec(
        name="deep_scaled_l2",
        architecture=(12, 24, 12, 1),
        hidden_activation="relu",
        learning_rate=0.01,
        l2_lambda=1e-3,
        use_scaler=True,
        train_fraction=1.0,
        epochs=1400,
        notes="Derin modele L2 regülarizasyon eklendi.",
    ),
    ExperimentSpec(
        name="deep_scaled_l2_data50",
        architecture=(12, 24, 12, 1),
        hidden_activation="relu",
        learning_rate=0.01,
        l2_lambda=1e-3,
        use_scaler=True,
        train_fraction=0.50,
        epochs=1400,
        notes="Derin L2 model train verisinin yuzde 50'si ile egitildi.",
    ),
    ExperimentSpec(
        name="deep_scaled_l2_data75",
        architecture=(12, 24, 12, 1),
        hidden_activation="relu",
        learning_rate=0.01,
        l2_lambda=1e-3,
        use_scaler=True,
        train_fraction=0.75,
        epochs=1400,
        notes="Derin L2 model train verisinin yuzde 75'i ile egitildi.",
    ),
)

BACKEND_COMPARISON_RUNS: tuple[ExperimentSpec, ...] = (
    NUMPY_EXPERIMENTS[1],
    NUMPY_EXPERIMENTS[2],
    NUMPY_EXPERIMENTS[4],
)

EXPECTED_OUTPUTS = (
    TABLE_DIR / "numpy_experiment_metrics.csv",
    TABLE_DIR / "backend_comparison_metrics.csv",
    TABLE_DIR / "model_selection.csv",
    FIGURE_DIR / "class_distribution.png",
    FIGURE_DIR / "numpy_learning_curves.png",
    FIGURE_DIR / "data_fraction_comparison.png",
    FIGURE_DIR / "confusion_matrix_selected_numpy.png",
    FIGURE_DIR / "confusion_matrix_selected_sklearn.png",
    FIGURE_DIR / "confusion_matrix_selected_pytorch.png",
    REPORT_DIR / "experiment_summary.md",
    REPORT_DIR / "experiment_summary.json",
    REPORT_DIR / "selected_model_report.md",
    REPORT_DIR / "traceability_matrix.md",
)
