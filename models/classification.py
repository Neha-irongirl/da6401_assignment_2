

import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder backbone + classification head."""

    def __init__(self, num_classes: int = 37,
                 in_channels: int = 3,
                 dropout_p: float = 0.5):
        super().__init__()

        encoder         = VGG11Encoder(num_classes=num_classes,
                                       dropout_p=dropout_p)
        self.features   = encoder.features
        self.avgpool    = encoder.avgpool
        self.classifier = encoder.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
