
import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """
    Custom Intersection over Union loss.
    Input boxes: [x_center, y_center, width, height] in pixel space (NOT normalized).
    Loss = 1 - IoU, so range is [0, 1].
    reduction: 'mean' (default) or 'sum'
    Epsilon added to union to avoid division by zero and ensure gradient stability.
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        assert reduction in ('mean', 'sum'), "reduction must be 'mean' or 'sum'"
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred:   (B, 4) predicted boxes   [cx, cy, w, h]
        target: (B, 4) ground truth boxes [cx, cy, w, h]
        """
        # --- Step 1: convert center format → corner format ---
        pred_x1 = pred[:, 0] - pred[:, 2] / 2
        pred_y1 = pred[:, 1] - pred[:, 3] / 2
        pred_x2 = pred[:, 0] + pred[:, 2] / 2
        pred_y2 = pred[:, 1] + pred[:, 3] / 2

        tgt_x1 = target[:, 0] - target[:, 2] / 2
        tgt_y1 = target[:, 1] - target[:, 3] / 2
        tgt_x2 = target[:, 0] + target[:, 2] / 2
        tgt_y2 = target[:, 1] + target[:, 3] / 2

        # --- Step 2: intersection rectangle ---
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        intersection = inter_w * inter_h

        # --- Step 3: union ---
        pred_area = pred[:, 2] * pred[:, 3]
        tgt_area  = target[:, 2] * target[:, 3]
        union = pred_area + tgt_area - intersection + 1e-6  # epsilon for stability

        # --- Step 4: IoU loss in [0, 1] ---
        iou  = intersection / union
        loss = 1.0 - iou

        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()
