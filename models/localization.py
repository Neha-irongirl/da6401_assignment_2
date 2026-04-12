"""Localization modules."""

import torch
import torch.nn as nn

from models.layers import CustomDropout
from models.vgg11 import VGG11Encoder


class VGG11Localizer(nn.Module):
    """VGG11-based localizer with a regression head for bounding boxes."""

    def __init__(
        self,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        pretrained_path: str | None = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if in_channels != 3:
            raise ValueError("VGG11Localizer currently supports only 3-channel input.")

        encoder = VGG11Encoder(num_classes=37, dropout_p=dropout_p)
        self.features = encoder.features
        self.avgpool = encoder.avgpool
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
        )

        if pretrained_path:
            self.load_pretrained_backbone(pretrained_path, freeze_backbone=freeze_backbone)

    def load_pretrained_backbone(
        self,
        pretrained_path: str,
        freeze_backbone: bool = False,
    ) -> None:
        """Load matching backbone weights from a classifier checkpoint."""
        state_dict = torch.load(pretrained_path, map_location="cpu")
        filtered_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith("features.")
        }
        self.load_state_dict(filtered_state, strict=False)

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict bounding boxes as [cx, cy, w, h]."""
        x = self.features(x)
        x = self.avgpool(x)
        x = self.regressor(x)
        return x


LocalizationModel = VGG11Localizer
