import torch
from torch import nn

from rotated.iou.mgiou import MGIoU2D


class MGIoU2DLoss(nn.Module):
    """MGIoU2D Loss for training oriented object detectors."""

    def __init__(self, eps: float = 1e-7, fast_mode: bool = False, reduction: str = "mean"):
        super().__init__()
        self.mgiou = MGIoU2D(fast_mode=fast_mode, eps=eps)
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute MGIoU2D loss.

        Args:
            pred_boxes: Predicted boxes [N, 5] - (x, y, w, h, angle_in_radians)
            target_boxes: Target boxes [N, 5] - (x, y, w, h, angle_in_radians)

        Returns:
            Loss tensor
        """
        iou = self.mgiou(pred_boxes, target_boxes)

        loss = (1.0 - iou) * 0.5

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()

        return loss
