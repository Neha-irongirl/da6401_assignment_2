"""Segmentation model"""

import torch
import torch.nn as nn


class VGG11UNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5
    ):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.MaxPool2d(2)

        self.up5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(384, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(96, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(32, num_classes, 1)

    def load_pretrained_backbone(self, classifier_path: str, freeze: str = "none"):
        state_dict = torch.load(classifier_path, map_location="cpu")

        block_map = {
            "enc1": [0, 1, 2],
            "enc2": [4, 5, 6],
            "enc3": [8, 9, 10, 11, 12, 13],
            "enc4": [15, 16, 17, 18, 19, 20],
            "enc5": [22, 23, 24, 25, 26, 27],
        }

        layer_fields = [
            "weight", "bias",
            "running_mean", "running_var",
            "num_batches_tracked"
        ]

        for name, idxs in block_map.items():
            module = getattr(self, name)
            for local_idx, global_idx in enumerate(idxs):
                layer = module[local_idx]
                for field in layer_fields:
                    key = f"features.{global_idx}.{field}"
                    if key in state_dict and hasattr(layer, field):
                        try:
                            getattr(layer, field).data.copy_(state_dict[key])
                        except Exception:
                            continue

        if freeze == "all":
            freeze_blocks = [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]
        elif freeze == "partial":
            freeze_blocks = [self.enc1, self.enc2, self.enc3]
        else:
            freeze_blocks = []

        for block in freeze_blocks:
            for param in block.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool(skip1))
        skip3 = self.enc3(self.pool(skip2))
        skip4 = self.enc4(self.pool(skip3))
        bottleneck = self.enc5(self.pool(skip4))

        up = self.up5(bottleneck)
        up = self.dec5(torch.cat([up, skip4], dim=1))

        up = self.up4(up)
        up = self.dec4(torch.cat([up, skip3], dim=1))

        up = self.up3(up)
        up = self.dec3(torch.cat([up, skip2], dim=1))

        up = self.up2(up)
        up = self.dec2(torch.cat([up, skip1], dim=1))

        return self.final_conv(up)