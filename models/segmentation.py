
import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder


class DoubleConv(nn.Module):
    """Two Conv→BN→ReLU blocks (standard U-Net decoder unit)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SegmentationModel(nn.Module):
    """
    U-Net style semantic segmentation using VGG11 as the encoder.

    Architecture:
      Encoder  = VGG11 convolutional blocks (5 blocks with MaxPool)
      Decoder  = Symmetric expansive path using ConvTranspose2d (NO bilinear)
      Skip connections: each decoder stage concatenates the matching encoder output

    Loss choice: CrossEntropyLoss
      Reason: trimap has 3 discrete classes (foreground/background/boundary).
      CrossEntropy is natural for multi-class pixel classification and handles
      class imbalance better than MSE. We optionally add Dice loss on top
      for improved boundary sensitivity.

    num_classes=3 for Oxford trimaps (foreground, background, boundary).
    """

    def __init__(self, num_classes=3, pretrained_path=None, freeze_mode='partial'):
        """
        freeze_mode:
          'full'    — freeze entire VGG11 backbone
          'partial' — freeze blocks 1-2 only, fine-tune blocks 3-5
          'none'    — fine-tune entire backbone end-to-end
        """
        super().__init__()

        vgg = VGG11(num_classes=37)
        if pretrained_path:
            vgg.load_state_dict(
                torch.load(pretrained_path, map_location='cpu')
            )
            print(f"Loaded backbone from: {pretrained_path}")

        # ---- Encoder: 5 VGG11 blocks ----
        feat = list(vgg.features.children())
        self.enc1 = nn.Sequential(*feat[0:4])    # → (B, 64,  112, 112)
        self.enc2 = nn.Sequential(*feat[4:8])    # → (B, 128,  56,  56)
        self.enc3 = nn.Sequential(*feat[8:15])   # → (B, 256,  28,  28)
        self.enc4 = nn.Sequential(*feat[15:22])  # → (B, 512,  14,  14)
        self.enc5 = nn.Sequential(*feat[22:29])  # → (B, 512,   7,   7)

        # Apply freezing strategy
        if freeze_mode == 'full':
            for enc in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]:
                for p in enc.parameters():
                    p.requires_grad = False
            print("Encoder: fully frozen")
        elif freeze_mode == 'partial':
            for enc in [self.enc1, self.enc2]:
                for p in enc.parameters():
                    p.requires_grad = False
            print("Encoder: blocks 1-2 frozen, blocks 3-5 fine-tuned")
        else:
            print("Encoder: fully fine-tuned (end-to-end)")

        # ---- Decoder: ConvTranspose2d upsampling + skip concat ----
        # Level 5 → 4:  7×7  → 14×14
        self.up5   = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5  = DoubleConv(512 + 512, 512)   # 512 up + 512 skip

        # Level 4 → 3:  14×14 → 28×28
        self.up4   = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4  = DoubleConv(256 + 256, 256)   # 256 up + 256 skip

        # Level 3 → 2:  28×28 → 56×56
        self.up3   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3  = DoubleConv(128 + 128, 128)   # 128 up + 128 skip

        # Level 2 → 1:  56×56 → 112×112
        self.up2   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2  = DoubleConv(64 + 64, 64)      # 64 up + 64 skip

        # Level 1 → 0:  112×112 → 224×224
        self.up1   = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1  = DoubleConv(32, 32)

        # Final 1×1 conv → class logits per pixel
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)   # (B, 64,  112, 112)
        s2 = self.enc2(s1)  # (B, 128,  56,  56)
        s3 = self.enc3(s2)  # (B, 256,  28,  28)
        s4 = self.enc4(s3)  # (B, 512,  14,  14)
        s5 = self.enc5(s4)  # (B, 512,   7,   7)

        # Decoder with skip connections
        x = self.up5(s5)                          # → (B, 512, 14, 14)
        x = self.dec5(torch.cat([x, s4], dim=1))  # concat skip s4

        x = self.up4(x)                           # → (B, 256, 28, 28)
        x = self.dec4(torch.cat([x, s3], dim=1))  # concat skip s3

        x = self.up3(x)                           # → (B, 128, 56, 56)
        x = self.dec3(torch.cat([x, s2], dim=1))  # concat skip s2

        x = self.up2(x)                           # → (B, 64, 112, 112)
        x = self.dec2(torch.cat([x, s1], dim=1))  # concat skip s1

        x = self.up1(x)                           # → (B, 32, 224, 224)
        x = self.dec1(x)

        return self.final_conv(x)                 # (B, num_classes, 224, 224)
