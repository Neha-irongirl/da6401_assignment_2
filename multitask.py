
"""Unified multi-task model"""

import os
import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self,
                 num_breeds: int = 37,
                 seg_classes: int = 3,
                 in_channels: int = 3,
                 classifier_path: str = "classifier.pth",
                 localizer_path: str = "localizer.pth",
                 unet_path: str = "unet.pth"):
        super().__init__()

        # ── Shared backbone ────────────────────────────────────────
        encoder       = VGG11Encoder(num_classes=num_breeds)
        self.features = encoder.features
        self.avgpool  = encoder.avgpool

        # ── Encoder blocks for skip connections ────────────────────
        self.enc1 = nn.Sequential(*list(encoder.features.children())[0:3])
        self.enc2 = nn.Sequential(*list(encoder.features.children())[4:7])
        self.enc3 = nn.Sequential(*list(encoder.features.children())[8:14])
        self.enc4 = nn.Sequential(*list(encoder.features.children())[15:21])
        self.enc5 = nn.Sequential(*list(encoder.features.children())[22:28])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Task 1: Classification head ────────────────────────────
        self.cls_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, num_breeds),
        )

        # ── Task 2: Localization head ──────────────────────────────
        self.loc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.ReLU(inplace=True),
        )

        # ── Task 3: Segmentation decoder ──────────────────────────
        self.up5  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(256 + 512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.up4  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(128 + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.up3  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64 + 128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.up2  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32 + 64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.seg_final = nn.Conv2d(32, seg_classes, kernel_size=1)

        # ── Load checkpoints if they exist ─────────────────────────
        # NOTE: These will be skipped until you finish training
        # Tasks 1, 2, 3 and the .pth files exist
        if os.path.exists(classifier_path):
            self._load_backbone(classifier_path)
            self._load_cls_head(classifier_path)
        else:
            print(f"⚠️  classifier.pth not found — skipping load")

        if os.path.exists(localizer_path):
            self._load_loc_head(localizer_path)
        else:
            print(f"⚠️  localizer.pth not found — skipping load")

        if os.path.exists(unet_path):
            self._load_seg_decoder(unet_path)
        else:
            print(f"⚠️  unet.pth not found — skipping load")

    def _load_backbone(self, path):
        state = torch.load(path, map_location='cpu')
        backbone_w = {k.replace('features.', ''): v
                      for k, v in state.items()
                      if k.startswith('features.')}
        self.features.load_state_dict(backbone_w)
        print(f"✅ Backbone loaded from {path}")

    def _load_cls_head(self, path):
        state = torch.load(path, map_location='cpu')
        cls_w = {k.replace('classifier.', ''): v
                 for k, v in state.items()
                 if k.startswith('classifier.')}
        self.cls_head.load_state_dict(cls_w)
        print("✅ Classification head loaded")

    def _load_loc_head(self, path):
        state = torch.load(path, map_location='cpu')
        loc_w = {k.replace('regressor.', ''): v
                 for k, v in state.items()
                 if k.startswith('regressor.')}
        self.loc_head.load_state_dict(loc_w)
        print("✅ Localization head loaded")

    def _load_seg_decoder(self, path):
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state, strict=False)
        print("✅ Segmentation decoder loaded")

    def forward(self, x: torch.Tensor):
        """
        Single forward pass — all three tasks.

        Returns dict with keys:
          'classification': [B, num_breeds]
          'localization'  : [B, 4]
          'segmentation'  : [B, seg_classes, H, W]
        """
        # ── Encoder ────────────────────────────────────────────────
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        pooled = self.avgpool(e5)
        flat   = torch.flatten(pooled, 1)

        # ── Task 1: Classification ─────────────────────────────────
        cls_out = self.cls_head(flat)

        # ── Task 2: Localization ───────────────────────────────────
        loc_out = self.loc_head(pooled)

        # ── Task 3: Segmentation ───────────────────────────────────
        d5 = self.dec5(torch.cat([self.up5(e5), e4], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        seg_out = self.seg_final(d2)

        return {
            'classification': cls_out,
            'localization'  : loc_out,
            'segmentation'  : seg_out,
        }
