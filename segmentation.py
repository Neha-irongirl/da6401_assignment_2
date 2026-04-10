
"""Segmentation model"""

import torch
import torch.nn as nn


class VGG11UNet(nn.Module):
    """
    U-Net style segmentation network using VGG11 as encoder.

    Encoder: VGG11 convolutional blocks (5 blocks, each ending with MaxPool)
    Decoder: Mirror of encoder using TransposedConv for upsampling
             + skip connections from encoder at each stage
    Output : Per-pixel class scores [B, num_classes, H, W]
    """

    def __init__(self, num_classes: int = 3,
                 in_channels: int = 3,
                 dropout_p: float = 0.5):
        """
        Args:
            num_classes: Number of output classes (3: fg, bg, boundary)
            in_channels: Number of input channels (3 for RGB)
            dropout_p  : Dropout probability
        """
        super().__init__()

        # ── ENCODER blocks ─────────────────────────────────────────
        # Each block ends WITHOUT MaxPool — we apply pool separately
        # so we can save the pre-pool feature maps for skip connections

        # Block 1: (B, 3,   224, 224) → (B, 64,  224, 224)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Block 2: (B, 64,  112, 112) → (B, 128, 112, 112)
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Block 3: (B, 128, 56, 56) → (B, 256, 56, 56)
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Block 4: (B, 256, 28, 28) → (B, 512, 28, 28)
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Block 5 (bottleneck): (B, 512, 14, 14) → (B, 512, 14, 14)
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Shared MaxPool for all encoder blocks
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── DECODER blocks ─────────────────────────────────────────
        # TransposedConv doubles spatial size (learnable upsampling)
        # After concat with skip: channels double → conv reduces them

        # Up 5→4: 14x14 → 28x28
        # TransposedConv: (B, 512, 14, 14) → (B, 256, 28, 28)
        # After concat with enc4 skip (512 ch): (B, 768, 28, 28)
        self.up5    = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec5   = nn.Sequential(
            nn.Conv2d(256 + 512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Up 4→3: 28x28 → 56x56
        # TransposedConv: (B, 256, 28, 28) → (B, 128, 56, 56)
        # After concat with enc3 skip (256 ch): (B, 384, 56, 56)
        self.up4    = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4   = nn.Sequential(
            nn.Conv2d(128 + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Up 3→2: 56x56 → 112x112
        # TransposedConv: (B, 128, 56, 56) → (B, 64, 112, 112)
        # After concat with enc2 skip (128 ch): (B, 192, 112, 112)
        self.up3    = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3   = nn.Sequential(
            nn.Conv2d(64 + 128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Up 2→1: 112x112 → 224x224
        # TransposedConv: (B, 64, 112, 112) → (B, 32, 224, 224)
        # After concat with enc1 skip (64 ch): (B, 96, 224, 224)
        self.up2    = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2   = nn.Sequential(
            nn.Conv2d(32 + 64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ── Final output layer ─────────────────────────────────────
        # 1x1 conv to map to num_classes scores per pixel
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def load_pretrained_backbone(self, classifier_path: str,
                                  freeze: str = 'none'):
        """
        Load VGG11Classifier weights into encoder blocks.

        Args:
            classifier_path: Path to saved classifier .pth file
            freeze: 'all'     = freeze entire backbone
                    'partial'  = freeze enc1-enc3, train enc4-enc5
                    'none'     = fine-tune everything (default)
        """
        state_dict = torch.load(classifier_path, map_location='cpu')

        # Map classifier feature indices to our encoder blocks
        # VGG11Encoder.features indices:
        # 0-3   → Block1 (enc1) : Conv,BN,ReLU,Pool
        # 4-7   → Block2 (enc2) : Conv,BN,ReLU,Pool
        # 8-14  → Block3 (enc3) : Conv,BN,ReLU,Conv,BN,ReLU,Pool
        # 15-21 → Block4 (enc4) : Conv,BN,ReLU,Conv,BN,ReLU,Pool
        # 22-28 → Block5 (enc5) : Conv,BN,ReLU,Conv,BN,ReLU,Pool

        block_map = {
            'enc1': [0, 1, 2],
            'enc2': [4, 5, 6],
            'enc3': [8, 9, 10, 11, 12, 13],
            'enc4': [15, 16, 17, 18, 19, 20],
            'enc5': [22, 23, 24, 25, 26, 27],
        }

        layer_names = ['weight', 'bias', 'running_mean',
                       'running_var', 'num_batches_tracked']

        for block_name, indices in block_map.items():
            block = getattr(self, block_name)
            for local_idx, global_idx in enumerate(indices):
                layer = block[local_idx]
                for param in layer_names:
                    key = f'features.{global_idx}.{param}'
                    if key in state_dict and hasattr(layer, param.split('.')[0]):
                        try:
                            getattr(layer, param).data.copy_(state_dict[key])
                        except Exception:
                            pass

        print(f"✅ Loaded pretrained backbone from {classifier_path}")

        # Apply freezing strategy
        if freeze == 'all':
            for block in [self.enc1, self.enc2, self.enc3,
                          self.enc4, self.enc5]:
                for p in block.parameters():
                    p.requires_grad = False
            print("🔒 Entire backbone FROZEN")

        elif freeze == 'partial':
            for block in [self.enc1, self.enc2, self.enc3]:
                for p in block.parameters():
                    p.requires_grad = False
            print("🔒 enc1-enc3 frozen | 🔓 enc4-enc5 fine-tuned")

        else:
            print("🔓 Full backbone will be FINE-TUNED")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for segmentation.

        Args:
            x: Input tensor [B, in_channels, H, W]

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        # ── Encoder (save outputs for skip connections) ────────────
        e1 = self.enc1(x)           # (B, 64,  224, 224)
        e2 = self.enc2(self.pool(e1))  # (B, 128, 112, 112)
        e3 = self.enc3(self.pool(e2))  # (B, 256, 56,  56)
        e4 = self.enc4(self.pool(e3))  # (B, 512, 28,  28)
        e5 = self.enc5(self.pool(e4))  # (B, 512, 14,  14)

        # ── Decoder (upsample + concat skip connections) ───────────
        # Up from bottleneck
        d5 = self.up5(e5)                        # (B, 256, 28, 28)
        d5 = torch.cat([d5, e4], dim=1)          # (B, 768, 28, 28)
        d5 = self.dec5(d5)                       # (B, 256, 28, 28)

        d4 = self.up4(d5)                        # (B, 128, 56, 56)
        d4 = torch.cat([d4, e3], dim=1)          # (B, 384, 56, 56)
        d4 = self.dec4(d4)                       # (B, 128, 56, 56)

        d3 = self.up3(d4)                        # (B, 64,  112, 112)
        d3 = torch.cat([d3, e2], dim=1)          # (B, 192, 112, 112)
        d3 = self.dec3(d3)                       # (B, 64,  112, 112)

        d2 = self.up2(d3)                        # (B, 32,  224, 224)
        d2 = torch.cat([d2, e1], dim=1)          # (B, 96,  224, 224)
        d2 = self.dec2(d2)                       # (B, 32,  224, 224)

        # ── Final pixel-wise classification ────────────────────────
        out = self.final_conv(d2)                # (B, num_classes, 224, 224)
        return out
