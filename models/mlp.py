import torch.nn as nn
from models.base import BaseNN

class MLP(BaseNN):
    def __init__(self, num_classes=10):
        super().__init__(name="mlp")

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)
