import torch
from torch import nn

from rotated.iou.mgiou import MGIoU2D


class MGIoU2DLoss(nn.Module):
    """MGIoU2D Loss for training oriented object detectors.

    As presented in the paper, this loss is supposed to eliminate the need for specific angle loss.

    Claimed benefits:
        - Fast computation (faster than KFIoU, GWD and KLD)
        - Unifies position, dimension, and orientation optimization into a single, differentiable objective
        - Can be applied to arbitrary convex parametric shapes in any dimension, though here it is only
          implement for 2D boxes, for computational efficiency.

    Reference: https://arxiv.org/abs/2504.16443
    """

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
