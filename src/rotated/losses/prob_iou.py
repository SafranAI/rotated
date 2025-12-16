import torch
import torch.nn as nn

from rotated.iou.prob_iou import ProbIoU


class ProbIoULoss(nn.Module):
    """ProbIoU Loss for training object detectors."""

    def __init__(self, eps: float = 1e-7, mode: str = "l1", reduction: str = "mean"):
        super().__init__()
        self.probiou = ProbIoU(eps=eps)
        self.mode = mode
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute ProbIoU loss.

        Args:
            pred_boxes: Predicted boxes [N, 5] - (x, y, w, h, angle_in_radians)
            target_boxes: Target boxes [N, 5] - (x, y, w, h, angle_in_radians)

        Returns:
            Loss tensor

        Raises:
            ValueError: If unsupported loss mode
        """
        iou = self.probiou(pred_boxes, target_boxes)

        if self.mode == "iou" or self.mode == "l1":
            loss = 1.0 - iou
        elif self.mode == "l2":
            l1_loss = 1.0 - iou
            l_squared = l1_loss.pow(2)
            loss = -torch.log(1.0 - l_squared + self.probiou.eps)
        else:
            raise ValueError(f"Unsupported loss mode: {self.mode}")

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()

        return loss
