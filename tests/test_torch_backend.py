import numpy as np

from src.pytorch_backend import fit_torch_model


def test_torch_backend_loads_weights_and_returns_expected_shapes() -> None:
    X = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 1], dtype=np.int64)
    weights = [np.full((2, 2), 0.1, dtype=np.float64), np.full((2, 1), -0.2, dtype=np.float64)]
    biases = [np.zeros((1, 2), dtype=np.float64), np.zeros((1, 1), dtype=np.float64)]
    result = fit_torch_model(
        (2, 2, 1),
        hidden_activation="sigmoid",
        learning_rate=0.1,
        l2_lambda=0.0,
        weights=weights,
        biases=biases,
        X_train=X,
        y_train=y,
        X_val=X,
        y_val=y,
        X_test=X,
        y_test=y,
        epochs=5,
    )
    assert result.train_probabilities.shape == (4,)
    assert result.final_weights[0].shape == weights[0].shape
