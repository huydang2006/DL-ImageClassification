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

    def __init__(self, backbone_name="mobilenet_v2", num_classes: int = NUM_CLASSES, freeze_backbone: bool = True):
        super(TransferModel, self).__init__()
        # TODO: self.backbone = models.<backbone_name>(pretrained=True)
        #       self.classifier = nn.Linear(...)  # hoặc thay classifier gốc
        #       if freeze_backbone: freeze các param
        return

    def forward(self, x):
        """
        Args:
            x: tensor [batch, 3, IMG_SIZE_M3, IMG_SIZE_M3]
        Returns:
            logits: tensor [batch, num_classes]
        """
        # x = self.backbone(x)
        # logits = self.classifier(x)
        # return logits
        return NotImplemented


def unfreeze_backbone(model: TransferModel, unfreeze_from: int = 100):
    """
    Mở backbone để fine-tuning từ layer thứ `unfreeze_from` trở lên.

    TODO:
    - for param in model.parameters(): param.requires_grad = True
    - Có thể set requires_grad=False cho các layer trước `unfreeze_from`
    """
    return NotImplemented
