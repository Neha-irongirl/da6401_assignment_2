
"""Classification components"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder backbone + classification head."""

    def __init__(self, num_classes: int = 37,
                 in_channels: int = 3,
                 dropout_p: float = 0.5):
        """
        Args:
            num_classes: Number of output classes (37 pet breeds)
            in_channels: Number of input channels (3 for RGB)
            dropout_p  : Dropout probability for classifier head
        """
        super().__init__()

        # ── Encoder: VGG11 convolutional backbone ──────────────────
        # We reuse VGG11Encoder but only take its feature extractor
        # The encoder outputs (B, 512, 7, 7) feature maps
        encoder      = VGG11Encoder(num_classes=num_classes,
                                    dropout_p=dropout_p)
        self.features = encoder.features
        self.avgpool  = encoder.avgpool

        # ── Classification head ────────────────────────────────────
        # Takes flattened features and outputs class scores
        self.classifier = encoder.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            x: Input tensor [B, in_channels, H, W]

        Returns:
            Classification logits [B, num_classes]
        """
        x = self.features(x)     # (B, 512, 7, 7)
        x = self.avgpool(x)      # (B, 512, 7, 7)
        x = torch.flatten(x, 1)  # (B, 25088)
        x = self.classifier(x)   # (B, num_classes)
        return x
