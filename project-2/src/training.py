import random

import numpy as np
import torch
from torch import nn

from src.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED
from src.data import DigitsData, make_loader


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def evaluate_model(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> tuple[float, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    loader = make_loader(x, y, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            losses.append(float(criterion(logits, targets).item()))
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    return float(np.mean(losses)), np.concatenate(predictions)


def train_model(model: nn.Module, data: DigitsData, model_name: str, device: torch.device) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.to(device)

    train_loader = make_loader(data.x_train, data.y_train, batch_size=BATCH_SIZE, shuffle=True)
    history: list[dict[str, float]] = []
    best_state = None
    best_val_accuracy = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss, val_pred = evaluate_model(model, data.x_val, data.y_val, device)
        val_accuracy = float((val_pred == data.y_val).mean())
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_pred = evaluate_model(model, data.x_test, data.y_test, device)
    return {
        "model_name": model_name,
        "model": model,
        "history": history,
        "test_loss": test_loss,
        "test_pred": test_pred,
        "best_val_accuracy": best_val_accuracy,
    }


def extract_features(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    features: list[np.ndarray] = []
    loader = make_loader(x, np.zeros(len(x), dtype=np.int64), batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            batch_features = model.extract_features(inputs).cpu().numpy()
            features.append(batch_features)
    return np.concatenate(features, axis=0)
