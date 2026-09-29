"""M2: Deep CNN — nâng cấp so với Simple NN.

Kiến trúc đề xuất (placeholder):
    Block 1: Conv2d(3, 32) -> BatchNorm2d -> ReLU -> MaxPool2d
    Block 2: Conv2d(32, 64) -> BatchNorm2d -> ReLU -> MaxPool2d
    Block 3: Conv2d(64, 128) -> BatchNorm2d -> ReLU -> MaxPool2d
    AdaptiveAvgPool2d(1) -> Flatten -> Dropout(0.5) -> Linear(128, NUM_CLASSES)

Regularizations đề xuất: BatchNorm2d, Dropout, L2 (weight_decay).
"""

import torch.nn as nn

from src.config import NUM_CLASSES


class DeepCNN(nn.Module):
    """
    Mạng CNN sâu.

    TODO: dùng nn.Sequential hoặc custom module để build kiến trúc trên.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super(DeepCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: tensor [batch, 3, IMG_SIZE_M2, IMG_SIZE_M2]
        Returns:
            logits: tensor [batch, num_classes]
        """
        return self.classifier(self.features(x))
