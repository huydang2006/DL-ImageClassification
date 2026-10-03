"""Transfer learning models (MobileNetV2 / ResNet50 / EfficientNet-B0)."""

import torch.nn as nn
import torchvision.models as models
from src.config import NUM_CLASSES

_BUILDERS = {
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
}


class TransferModel(nn.Module):
    """Pretrained backbone with a replacement classification head."""

    def __init__(
        self,
        backbone_name="mobilenet_v2",
        num_classes: int = NUM_CLASSES,
        freeze_backbone: bool = True,
        pretrained: bool = False,
    ):
        super().__init__()
        if backbone_name not in _BUILDERS:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        builder, weights = _BUILDERS[backbone_name]
        self.backbone = builder(weights=weights if pretrained else None)

        if backbone_name == "resnet50":
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        else:
            self.backbone.classifier[-1] = nn.Linear(
                self.backbone.classifier[-1].in_features, num_classes
            )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            classifier = (
                self.backbone.classifier
                if hasattr(self.backbone, "classifier")
                else self.backbone.fc
            )
            for parameter in classifier.parameters():
                parameter.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


def unfreeze_backbone(model: TransferModel, unfreeze_from: int = 100):
    """Unfreeze the backbone, keeping the first `unfreeze_from` parameters frozen."""
    parameters = list(model.backbone.parameters())
    for parameter in parameters:
        parameter.requires_grad = True
    if unfreeze_from > 0:
        for parameter in parameters[:unfreeze_from]:
            parameter.requires_grad = False
    return model
