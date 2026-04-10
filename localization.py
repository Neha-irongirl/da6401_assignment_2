
"""Localization modules"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder


class VGG11Localizer(nn.Module):
    """VGG11-based localizer.

    Reuses VGG11Encoder backbone from Task 1.
    Adds a regression head to predict bounding box coordinates.
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            dropout_p  : Dropout probability for localization head
        """
        super().__init__()

        # ── Encoder: reuse VGG11 convolutional backbone ────────────
        # Outputs (B, 512, 7, 7) feature maps
        encoder       = VGG11Encoder(num_classes=37, dropout_p=dropout_p)
        self.features = encoder.features
        self.avgpool  = encoder.avgpool

        # ── Regression head ────────────────────────────────────────
        # Takes flattened features → outputs 4 bbox coordinates
        # ReLU at end ensures coordinates are always >= 0 (pixel space)
        self.regressor = nn.Sequential(
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 4),
            nn.ReLU(inplace=True),  # pixel coords must be >= 0
        )

    def load_pretrained_backbone(self, classifier_path: str,
                                  freeze: bool = False):
        """
        Load backbone weights from a trained VGG11Classifier checkpoint.

        Args:
            classifier_path: Path to saved classifier .pth file
            freeze         : If True, freeze backbone weights during training
                             If False, fine-tune backbone (usually better)
        """
        # Load the saved classifier
        state_dict = torch.load(classifier_path, map_location='cpu')

        # Extract only the feature extractor weights
        backbone_weights = {
            k.replace('features.', ''): v
            for k, v in state_dict.items()
            if k.startswith('features.')
        }
        self.features.load_state_dict(backbone_weights)
        print(f"✅ Loaded pretrained backbone from {classifier_path}")

        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False
            print("🔒 Backbone FROZEN — only regression head will train")
        else:
            print("🔓 Backbone will be FINE-TUNED")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for localization.

        Args:
            x: Input tensor [B, in_channels, H, W]

        Returns:
            Bounding box coordinates [B, 4]
            in (x_center, y_center, width, height) format in pixel space
        """
        x = self.features(x)    # (B, 512, 7, 7)
        x = self.avgpool(x)     # (B, 512, 7, 7)
        x = self.regressor(x)   # (B, 4)
        return x
