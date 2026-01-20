import torch
from torch import Tensor

from rotated.boxes.conversion import obb_to_corners_format
from rotated.iou.prob_iou import ProbIoU


class MGIoU2D:
    """Marginalized Generalized IoU.

    Projects the rotated bounding boxes onto their unique normals and computes the 1-D normalized GIoU.
    It is not the most precise approximation of real IoU but has the benefit of unifying position, dimension, and
    orientation optimization into a single, differentiable objective, eliminating the need for task-specific loss
    balancing or tuning.
    This provides differentiable IoU computation with better gradient flow compared to discrete polygon IoU,
    and better latency compared to KFIoU, GWD and KLD.

    NOTE: It is not meant to be used as a replacement/approximation of the real IoU, but rather as a loss function,
    see `MGIoU2DLoss`. Hence, we dont recommend using it as `iou_calculator` in `TaskAlignedAssigner`.

    NOTE: For degenerated boxes, we fallback to ProbIoU.

    Reference:
        Title: Marginalized Generalized IoU (MGIoU): A Unified Objective Function for Optimizing
               Any Convex Parametric Shapes.
        Authors: Duy-Tho Le, Trung Pham, Jianfei Cai, Hamid Rezatofighi
        Paper link: https://arxiv.org/abs/2504.16443
        Website: https://ldtho.github.io/MGIoU/

    Args:
        fast_mode: if True, will approximate GIoU1D (skipping convex hull computation)
        eps: Small constant for numerical stability
    """

    def __init__(self, fast_mode: bool = False, eps: float = 1e-7):
        self.fast_mode = fast_mode
        self.eps = eps

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.shape != target.shape or pred.shape[-1] != 5:
            raise ValueError("Expected boxes of shape (B, 5)")
        B = pred.size(0)

        # detect degenerate GTs → fallback to ProbIoU
        all_zero = target.abs().sum(dim=1) == 0
        iou = pred.new_zeros(B)
        if all_zero.any():
            prob_iou = ProbIoU()
            iou[all_zero] = prob_iou(pred[all_zero], target[all_zero])

        mask = ~all_zero
        if mask.any():
            # convert to corners
            c1 = obb_to_corners_format(pred[mask], degrees=False)
            c2 = obb_to_corners_format(target[mask], degrees=False)
            iou[mask] = self._mgiou_boxes(c1, c2)

        return iou.clamp(0.0, 1.0)

    def _mgiou_boxes(self, c1: Tensor, c2: Tensor) -> Tensor:
        # c1, c2: [N, 4, 2]
        axes = torch.cat((self._rect_axes(c1), self._rect_axes(c2)), dim=1)  # [N, 4, 2]
        proj1 = c1 @ axes.transpose(1, 2)  # [N, 4, 4]
        proj2 = c2 @ axes.transpose(1, 2)

        mn1, mx1 = proj1.min(dim=1).values, proj1.max(dim=1).values
        mn2, mx2 = proj2.min(dim=1).values, proj2.max(dim=1).values

        if self.fast_mode:
            num = torch.minimum(mx1, mx2) - torch.maximum(mn1, mn2)
            den = torch.maximum(mx1, mx2) - torch.minimum(mn1, mn2)
            giou1d = num / (den + self.eps)
        else:
            inter = (torch.minimum(mx1, mx2) - torch.maximum(mn1, mn2)).clamp(min=0.0)
            union = (mx1 - mn1) + (mx2 - mn2) - inter
            hull = torch.maximum(mx1, mx2) - torch.minimum(mn1, mn2)
            giou1d = inter / (union + self.eps) - (hull - union) / (hull + self.eps)

        return giou1d.mean(dim=-1)

    @staticmethod
    def _rect_axes(corners: Tensor) -> Tensor:
        e1 = corners[:, 1] - corners[:, 0]
        e2 = corners[:, 3] - corners[:, 0]
        normals = torch.stack((-e1[..., 1:], e1[..., :1], -e2[..., 1:], e2[..., :1]), dim=1)
        return normals.view(-1, 2, 2)
