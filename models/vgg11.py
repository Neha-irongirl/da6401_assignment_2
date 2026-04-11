
import torch
import torch.nn as nn
from models.layers import CustomDropout


class VGG11Encoder(nn.Module):
    """
    VGG11 from scratch following Simonyan & Zisserman (arXiv:1409.1556).
    Modifications:
      - BatchNorm2d after every Conv2d  (stabilizes training, allows higher LR)
      - BatchNorm1d after first two FC layers
      - CustomDropout instead of standard Dropout in classifier
    Input:  (B, 3, 224, 224)  — normalized images, fixed size per VGG paper
    Output: (B, num_classes)  — raw logits
    Feature map sizes after each MaxPool (with 224x224 input):
      After Block1 MaxPool: (B, 64,  112, 112)
      After Block2 MaxPool: (B, 128,  56,  56)
      After Block3 MaxPool: (B, 256,  28,  28)
      After Block4 MaxPool: (B, 512,  14,  14)
      After Block5 MaxPool: (B, 512,   7,   7)
    """

    def __init__(self, num_classes=37, dropout_p=0.5):
        super().__init__()

        self.features = nn.Sequential(
            # --- Block 1 ---  224×224 → 112×112
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),         # [0:4]

            # --- Block 2 ---  112×112 → 56×56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),         # [4:8]

            # --- Block 3 ---  56×56 → 28×28
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),         # [8:15]

            # --- Block 4 ---  28×28 → 14×14
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),         # [15:22]

            # --- Block 5 ---  14×14 → 7×7
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),         # [22:29]
        )

        # Ensures output is always 7×7 regardless of input size variation
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),      # after BN — see design justification in report

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, num_classes),    # no BN/activation — raw logits for CrossEntropy
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
