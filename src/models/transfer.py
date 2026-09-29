"""M3: Transfer Learning + Fine-tuning (PyTorch + torchvision).

Quy trình:
    1. Chọn backbone pretrained từ torchvision.models (MobileNetV2/ResNet50/EfficientNet_B0).
    2. Giai đoạn 1 (Transfer Learning): đóng băng backbone, training head.
    3. Giai đoạn 2 (Fine-tuning): mở một phần/top backbone, tiếp tục train với lr nhỏ.

TODO: chọn backbone trong hàm dưới đây, cấu hình theo từng giai đoạn.
"""

import torch.nn as nn
import torchvision.models as models

from src.config import NUM_CLASSES, IMG_SIZE_M3


class TransferModel(nn.Module):
    """
    Model transfer learning.

    Args:
        backbone_name: str — "mobilenet_v2" / "resnet50" / "efficientnet_b0".
        num_classes: int — số lớp output (mặc định 28).
        freeze_backbone: bool — nếu True, đóng băng backbone để training head.

    TODO:
    - backbone = models.<backbone_name>(pretrained=True)
    - thay đổi lớp classifier/output layer cho đúng num_classes
    - nếu freeze_backbone: for param in backbone.parameters(): param.requires_grad = False
    """

    def __init__(
        self,
        backbone_name="mobilenet_v2",
        num_classes: int = NUM_CLASSES,
        freeze_backbone: bool = True,
        pretrained: bool = False,
    ):
        super(TransferModel, self).__init__()
        builders = {
            "mobilenet_v2": (
                models.mobilenet_v2,
                models.MobileNet_V2_Weights.DEFAULT,
            ),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
            "efficientnet_b0": (
                models.efficientnet_b0,
                models.EfficientNet_B0_Weights.DEFAULT,
            ),
        }
        if backbone_name not in builders:
            raise ValueError(f"Backbone không hợp lệ: {backbone_name}")

        builder, weights = builders[backbone_name]
        self.backbone = builder(weights=weights if pretrained else None)
        if backbone_name == "mobilenet_v2":
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        elif backbone_name == "efficientnet_b0":
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)

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
        """
        Args:
            x: tensor [batch, 3, IMG_SIZE_M3, IMG_SIZE_M3]
        Returns:
            logits: tensor [batch, num_classes]
        """
        return self.backbone(x)


def unfreeze_backbone(model: TransferModel, unfreeze_from: int = 100):
    """
    Mở backbone để fine-tuning từ layer thứ `unfreeze_from` trở lên.

    Mở toàn bộ backbone để fine-tuning.
    """
    parameters = list(model.backbone.parameters())
    for parameter in parameters:
        parameter.requires_grad = True
    if unfreeze_from > 0:
        for parameter in parameters[:unfreeze_from]:
            parameter.requires_grad = False
    return model
