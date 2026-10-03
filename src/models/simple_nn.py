"""Basic fully connected neural network."""

import torch.nn as nn
from src.config import NUM_CLASSES, IMG_SIZE_M1

class SimpleNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        input_size = IMG_SIZE_M1 * IMG_SIZE_M1 * 3
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)
