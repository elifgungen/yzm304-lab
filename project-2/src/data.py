from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.config import BATCH_SIZE, SEED, TEST_SIZE, VAL_SIZE


@dataclass(frozen=True)
class DigitsData:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    @property
    def n_classes(self) -> int:
        return int(np.unique(self.y_train).size)


def load_digits_splits(seed: int = SEED) -> DigitsData:
    """Load sklearn digits as 1x8x8 normalized image tensors."""
    digits = load_digits()
    images = digits.images.astype(np.float32) / 16.0
    labels = digits.target.astype(np.int64)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        images,
        labels,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=labels,
    )
    relative_val_size = VAL_SIZE / (1.0 - TEST_SIZE)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=seed,
        stratify=y_train_val,
    )

    return DigitsData(
        x_train=x_train[:, None, :, :],
        x_val=x_val[:, None, :, :],
        x_test=x_test[:, None, :, :],
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int = BATCH_SIZE, shuffle: bool = False) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
