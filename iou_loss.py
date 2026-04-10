
"""Custom IoU loss"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression."""

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Args:
            eps      : Small value to avoid division by zero
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        assert reduction in {"mean", "sum"}, \
            f"reduction must be 'mean' or 'sum', got '{reduction}'"
        self.eps       = eps
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor,
                target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU loss between predicted and target bounding boxes.

        Args:
            pred_boxes  : [B, 4] in (x_center, y_center, width, height) format
            target_boxes: [B, 4] in (x_center, y_center, width, height) format

        Returns:
            Scalar loss value
        """
        # ── Step 1: Convert center format → corner format ──────────
        # pred box corners
        px1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2   # left
        py1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2   # top
        px2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2   # right
        py2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2   # bottom

        # target box corners
        tx1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        ty1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        tx2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        ty2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

        # ── Step 2: Compute intersection rectangle ─────────────────
        ix1 = torch.max(px1, tx1)
        iy1 = torch.max(py1, ty1)
        ix2 = torch.min(px2, tx2)
        iy2 = torch.min(py2, ty2)

        # Width and height of intersection (clamp at 0 if no overlap)
        inter_w = torch.clamp(ix2 - ix1, min=0)
        inter_h = torch.clamp(iy2 - iy1, min=0)
        inter   = inter_w * inter_h

        # ── Step 3: Compute union ──────────────────────────────────
        pred_area   = pred_boxes[:, 2]   * pred_boxes[:, 3]
        target_area = target_boxes[:, 2] * target_boxes[:, 3]
        union       = pred_area + target_area - inter + self.eps

        # ── Step 4: IoU and loss ───────────────────────────────────
        iou  = inter / union          # 0 = no overlap, 1 = perfect
        loss = 1.0 - iou              # 0 = perfect, 1 = no overlap

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()
