from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.metrics import binary_cross_entropy, compute_binary_metrics
from src.weights import clone_weight_bundle, weight_l2_norm


@dataclass
class NumpyTrainingResult:
    history: pd.DataFrame
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    train_loss: float
    val_loss: float
    test_loss: float
    train_probabilities: np.ndarray
    val_probabilities: np.ndarray
    test_probabilities: np.ndarray
    train_predictions: np.ndarray
    val_predictions: np.ndarray
    test_predictions: np.ndarray
    final_weights: list[np.ndarray]
    final_biases: list[np.ndarray]
    weight_norm: float


class NumpyMLP:
    gradient_clip_value = 5.0
    weight_clip_value = 25.0

    def __init__(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        *,
        hidden_activation: str,
        learning_rate: float,
        l2_lambda: float,
    ) -> None:
        self.weights, self.biases = clone_weight_bundle(weights, biases)
        self.hidden_activation = hidden_activation
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    @staticmethod
    def _sigmoid_grad(activated: np.ndarray) -> np.ndarray:
        return activated * (1.0 - activated)

    @staticmethod
    def _relu(values: np.ndarray) -> np.ndarray:
        return np.maximum(values, 0.0)

    @staticmethod
    def _relu_grad(values: np.ndarray) -> np.ndarray:
        return (values > 0.0).astype(np.float64)

    def _hidden_forward(self, z_values: np.ndarray) -> np.ndarray:
        if self.hidden_activation == "sigmoid":
            return self._sigmoid(z_values)
        if self.hidden_activation == "relu":
            return self._relu(z_values)
        raise ValueError(f"Unsupported activation: {self.hidden_activation}")

    def _hidden_backward(self, activated: np.ndarray, z_values: np.ndarray) -> np.ndarray:
        if self.hidden_activation == "sigmoid":
            return self._sigmoid_grad(activated)
        if self.hidden_activation == "relu":
            return self._relu_grad(z_values)
        raise ValueError(f"Unsupported activation: {self.hidden_activation}")

    def forward(self, features: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [features]
        z_values: list[np.ndarray] = []
        current = features
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
                z_layer = current @ weight + bias
                z_values.append(z_layer)
                if layer_index == len(self.weights) - 1:
                    current = self._sigmoid(z_layer)
                else:
                    current = self._hidden_forward(z_layer)
                activations.append(current)
        return activations, z_values

    def _loss_with_regularization(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        loss = binary_cross_entropy(y_true, y_prob)
        if self.l2_lambda == 0.0:
            return loss
        penalty = sum(np.sum(weight * weight) for weight in self.weights)
        return float(loss + 0.5 * self.l2_lambda * penalty / len(y_true))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        *,
        epochs: int,
    ) -> NumpyTrainingResult:
        y_train_column = y_train.reshape(-1, 1).astype(np.float64)
        history_rows: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                activations, z_values = self.forward(X_train)
                train_prob = activations[-1]
                delta = train_prob - y_train_column
                batch_size = float(len(X_train))

                grad_weights: list[np.ndarray] = []
                grad_biases: list[np.ndarray] = []
                for layer_index in reversed(range(len(self.weights))):
                    activation_prev = activations[layer_index]
                    grad_weight = (activation_prev.T @ delta) / batch_size
                    if self.l2_lambda:
                        grad_weight += (self.l2_lambda / batch_size) * self.weights[layer_index]
                    grad_bias = np.mean(delta, axis=0, keepdims=True)
                    grad_weight = np.nan_to_num(
                        grad_weight,
                        nan=0.0,
                        posinf=self.gradient_clip_value,
                        neginf=-self.gradient_clip_value,
                    )
                    grad_bias = np.nan_to_num(
                        grad_bias,
                        nan=0.0,
                        posinf=self.gradient_clip_value,
                        neginf=-self.gradient_clip_value,
                    )
                    grad_weight = np.clip(
                        grad_weight,
                        -self.gradient_clip_value,
                        self.gradient_clip_value,
                    )
                    grad_bias = np.clip(
                        grad_bias,
                        -self.gradient_clip_value,
                        self.gradient_clip_value,
                    )
                    grad_weights.append(grad_weight)
                    grad_biases.append(grad_bias)

                    if layer_index > 0:
                        delta = (delta @ self.weights[layer_index].T) * self._hidden_backward(
                            activations[layer_index],
                            z_values[layer_index - 1],
                        )

            grad_weights.reverse()
            grad_biases.reverse()
            for layer_index in range(len(self.weights)):
                self.weights[layer_index] -= self.learning_rate * grad_weights[layer_index]
                self.biases[layer_index] -= self.learning_rate * grad_biases[layer_index]
                self.weights[layer_index] = np.clip(
                    np.nan_to_num(
                        self.weights[layer_index],
                        nan=0.0,
                        posinf=self.weight_clip_value,
                        neginf=-self.weight_clip_value,
                    ),
                    -self.weight_clip_value,
                    self.weight_clip_value,
                )
                self.biases[layer_index] = np.clip(
                    np.nan_to_num(
                        self.biases[layer_index],
                        nan=0.0,
                        posinf=self.weight_clip_value,
                        neginf=-self.weight_clip_value,
                    ),
                    -self.weight_clip_value,
                    self.weight_clip_value,
                )

            train_prob_epoch = self.predict_proba(X_train)
            val_prob_epoch = self.predict_proba(X_val)
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": self._loss_with_regularization(y_train, train_prob_epoch),
                    "val_loss": binary_cross_entropy(y_val, val_prob_epoch),
                    "train_accuracy": float(np.mean((train_prob_epoch >= 0.5).astype(int) == y_train)),
                    "val_accuracy": float(np.mean((val_prob_epoch >= 0.5).astype(int) == y_val)),
                }
            )

        history = pd.DataFrame(history_rows)
        train_prob = self.predict_proba(X_train)
        val_prob = self.predict_proba(X_val)
        test_prob = self.predict_proba(X_test)
        train_pred = (train_prob >= 0.5).astype(int)
        val_pred = (val_prob >= 0.5).astype(int)
        test_pred = (test_prob >= 0.5).astype(int)
        return NumpyTrainingResult(
            history=history,
            train_metrics=compute_binary_metrics(y_train, train_pred, train_prob),
            val_metrics=compute_binary_metrics(y_val, val_pred, val_prob),
            test_metrics=compute_binary_metrics(y_test, test_pred, test_prob),
            train_loss=self._loss_with_regularization(y_train, train_prob),
            val_loss=binary_cross_entropy(y_val, val_prob),
            test_loss=binary_cross_entropy(y_test, test_prob),
            train_probabilities=train_prob,
            val_probabilities=val_prob,
            test_probabilities=test_prob,
            train_predictions=train_pred,
            val_predictions=val_pred,
            test_predictions=test_pred,
            final_weights=[weight.copy() for weight in self.weights],
            final_biases=[bias.copy() for bias in self.biases],
            weight_norm=weight_l2_norm(self.weights),
        )

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        activations, _ = self.forward(features)
        return activations[-1].reshape(-1)
