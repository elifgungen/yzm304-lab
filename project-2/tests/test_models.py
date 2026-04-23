import torch

from src.models import AlexNetSmallCNN, ImprovedLeNetCNN, LeNetLikeCNN


def test_models_return_class_logits():
    x = torch.rand(4, 1, 8, 8)
    for model_cls in [LeNetLikeCNN, ImprovedLeNetCNN, AlexNetSmallCNN]:
        model = model_cls(n_classes=10)
        logits = model(x)
        assert logits.shape == (4, 10)
        features = model.extract_features(x)
        assert features.shape[0] == 4
