from models.cnn import CNN
from models.mlp import MLP
from models.vgg import VGG11
from models.base import BaseNN

def create_model(
    model_name,
    num_classes=10
    ):

    if model_name == "mlp":
        return MLP(num_classes=num_classes)

    elif model_name == "cnn":
        return CNN(num_classes=num_classes)

    elif model_name == "vgg11":
        return VGG11(
            num_classes=num_classes
        )

    raise ValueError(f"Unknown model name: {model_name}")
