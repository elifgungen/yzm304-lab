import torch
from torch import nn


class LeNetLikeCNN(nn.Module):
    """LeNet-5 style baseline adapted to 1x8x8 digits images."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.features(x), start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ImprovedLeNetCNN(nn.Module):
    """Same convolution and classifier widths as LeNetLikeCNN, with BN and dropout."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 120),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(84, n_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.features(x), start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class AlexNetSmallCNN(nn.Module):
    """Compact AlexNet-inspired CNN for small grayscale images."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(p=0.35),
            nn.Linear(256, n_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.features(x), start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
