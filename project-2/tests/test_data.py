from src.data import load_digits_splits


def test_digits_splits_have_expected_shape():
    data = load_digits_splits()
    assert data.x_train.ndim == 4
    assert data.x_train.shape[1:] == (1, 8, 8)
    assert data.n_classes == 10
    assert len(data.y_train) > len(data.y_test)
