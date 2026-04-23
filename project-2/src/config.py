from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
FEATURES_DIR = OUTPUT_DIR / "features"
REPORTS_DIR = OUTPUT_DIR / "reports"

SEED = 42
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
TEST_SIZE = 0.15
VAL_SIZE = 0.15

CLASS_NAMES = [str(i) for i in range(10)]


def ensure_output_dirs() -> None:
    for path in [FIGURES_DIR, TABLES_DIR, FEATURES_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
