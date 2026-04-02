import numpy as np

from src.metrics import binary_cross_entropy, compute_binary_metrics


def test_binary_cross_entropy_and_metrics() -> None:
    y_true = np.array([0, 1, 1, 0], dtype=int)
    y_prob = np.array([0.1, 0.8, 0.7, 0.2], dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    loss = binary_cross_entropy(y_true, y_prob)
    metrics = compute_binary_metrics(y_true, y_pred, y_prob)
    assert loss > 0.0
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0

