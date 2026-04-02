from src.config import CONTINUOUS_COLUMNS
from src.dataset import export_dataset_artifacts, prepare_features


def test_split_manifest_is_reproducible() -> None:
    bundle_one = export_dataset_artifacts()
    bundle_two = export_dataset_artifacts()
    assert bundle_one.manifest == bundle_two.manifest


def test_scaler_is_fit_only_on_train_columns() -> None:
    bundle = export_dataset_artifacts()
    prepared = prepare_features(bundle, bundle.feature_names, use_scaler=True, train_fraction=1.0)
    means = prepared.X_train.loc[:, list(CONTINUOUS_COLUMNS)].mean().abs()
    assert float(means.max()) < 1e-9
    assert prepared.scaler is not None


def test_train_fraction_indices_are_nested_in_manifest() -> None:
    bundle = export_dataset_artifacts()
    fractions = bundle.manifest["train_fraction_indices"]
    half = set(fractions["0.50"])
    three_quarters = set(fractions["0.75"])
    full = set(fractions["1.00"])
    assert half.issubset(three_quarters)
    assert three_quarters.issubset(full)
    assert len(half) < len(three_quarters) < len(full)
