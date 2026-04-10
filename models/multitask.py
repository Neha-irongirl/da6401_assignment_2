
import torch
import torch.nn as nn
import os


class MultiTaskPerceptionModel(nn.Module):
    """
    Unified multi-task model combining:
      - Classification head  (37 pet breed logits)
      - Localization head    ([x_center, y_center, width, height] in pixels)
      - Segmentation head    (pixel-wise trimap mask, 3 classes)
    Loads pretrained weights from saved checkpoints.
    A single forward pass returns all three outputs simultaneously.
    """

    def __init__(self,
                 classifier_path='checkpoints/classifier.pth',
                 localizer_path='checkpoints/localizer.pth',
                 unet_path='checkpoints/unet.pth',
                 num_classes=37):
        super().__init__()

        # ---- Download checkpoints from Google Drive ----
        # (Replace the drive IDs below with your actual IDs before submission)
        import gdown
        gdown.download(id="YOUR_CLASSIFIER_DRIVE_ID", output=classifier_path, quiet=False)
        gdown.download(id="YOUR_LOCALIZER_DRIVE_ID",  output=localizer_path,  quiet=False)
        gdown.download(id="YOUR_UNET_DRIVE_ID",       output=unet_path,       quiet=False)

        # ---- Load individual models to extract components ----
        from models.vgg11 import VGG11
        from models.localization import LocalizationModel
        from models.segmentation import SegmentationModel

        # Shared backbone: loaded from classifier checkpoint
        vgg = VGG11(num_classes=num_classes)
        vgg.load_state_dict(torch.load(classifier_path, map_location='cpu'))

        feat = list(vgg.features.children())
        self.enc1 = nn.Sequential(*feat[0:4])
        self.enc2 = nn.Sequential(*feat[4:8])
        self.enc3 = nn.Sequential(*feat[8:15])
        self.enc4 = nn.Sequential(*feat[15:22])
        self.enc5 = nn.Sequential(*feat[22:29])
        self.avgpool = vgg.avgpool

        # ---- Classification head ----
        self.cls_head = vgg.classifier

        # ---- Localization head ----
        loc = LocalizationModel()
        loc.load_state_dict(torch.load(localizer_path, map_location='cpu'))
        self.loc_head = loc.regressor

        # ---- Segmentation decoder ----
        seg = SegmentationModel(num_classes=3)
        seg.load_state_dict(torch.load(unet_path, map_location='cpu'))
        self.up5        = seg.up5
        self.dec5       = seg.dec5
        self.up4        = seg.up4
        self.dec4       = seg.dec4
        self.up3        = seg.up3
        self.dec3       = seg.dec3
        self.up2        = seg.up2
        self.dec2       = seg.dec2
        self.up1        = seg.up1
        self.dec1       = seg.dec1
        self.final_conv = seg.final_conv

    def forward(self, x):
        # ---- Shared encoder (one forward pass through backbone) ----
        s1 = self.enc1(x)   # (B, 64,  112, 112)
        s2 = self.enc2(s1)  # (B, 128,  56,  56)
        s3 = self.enc3(s2)  # (B, 256,  28,  28)
        s4 = self.enc4(s3)  # (B, 512,  14,  14)
        s5 = self.enc5(s4)  # (B, 512,   7,   7)

        pooled = self.avgpool(s5)
        flat   = torch.flatten(pooled, 1)  # (B, 512*7*7)

        # ---- Task heads ----
        cls_out = self.cls_head(flat)      # (B, 37)

        loc_out = self.loc_head(flat)      # (B, 4)

        # Segmentation decoder uses skip connections from encoder
        d = self.up5(s5)
        d = self.dec5(torch.cat([d, s4], dim=1))
        d = self.up4(d)
        d = self.dec4(torch.cat([d, s3], dim=1))
        d = self.up3(d)
        d = self.dec3(torch.cat([d, s2], dim=1))
        d = self.up2(d)
        d = self.dec2(torch.cat([d, s1], dim=1))
        d = self.up1(d)
        d = self.dec1(d)
        seg_out = self.final_conv(d)       # (B, 3, 224, 224)

        return cls_out, loc_out, seg_out
