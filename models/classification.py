
import torch
import torch.nn as nn
from models.vgg11 import VGG11


class ClassificationModel(nn.Module):
    """
    Thin wrapper around VGG11 for the 37-class pet breed classification task.
    This class exists so train.py can import it consistently across all tasks.
    """

    def __init__(self, num_classes=37, dropout_p=0.5):
        super().__init__()
        self.model = VGG11(num_classes=num_classes, dropout_p=dropout_p)

    def forward(self, x):
        return self.model(x)
