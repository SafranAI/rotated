"""IoU approximation using Signed Distance Function with L1-Norm.

Adapted from: https://numbersmithy.com/an-algorithm-for-computing-the-approximate-iou-between-oriented-2d-boxes/
"""

import torch

from rotated.boxes.conversion import obb_to_corners_format
from rotated.boxes.utils import check_aabb_overlap


class ApproxSDFL1:
    def __init__(self, n_samples: int = 40, eps: float = 1e-7):
        self.n_samples = n_samples
        self.eps = eps

    def __call__(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        N = pred_boxes.shape[0]
        if N == 0:
            return torch.empty(0, device=pred_boxes.device, dtype=pred_boxes.dtype)

        # Step 1: AABB filtering
        overlap_mask = check_aabb_overlap(pred_boxes, target_boxes)
        ious = torch.zeros(N, device=pred_boxes.device, dtype=pred_boxes.dtype)

        if not overlap_mask.any():
            return ious

        # Step 2: Process overlapping candidates
        candidates = torch.where(overlap_mask)[0]
        pred_candidates = pred_boxes[candidates]
        target_candidates = target_boxes[candidates]
        if pred_candidates.shape[0] == 0:
            return torch.empty(0, device=pred_boxes.device, dtype=pred_boxes.dtype)

        # Box areas
        pred_area = pred_candidates[:, 2] * pred_candidates[:, 3]
        target_area = target_candidates[:, 2] * target_candidates[:, 3]

        a_extra = _saf_obox2obox_vec(pred_candidates, target_candidates, n_samples=self.n_samples)
        union = target_area + a_extra
        ious[candidates] = (pred_area + target_area) / (union + self.eps) - 1
        return ious.clamp(0.0, 1.0)


def _saf_obox2obox_vec(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, n_samples: int = 40) -> torch.Tensor:
    """Vectorized implementation of Signed area difference formed between pairs of predictions and target boxes.

    Args:
        pred_boxes: prediction box in format (xc, yc, w, h, angle). Shape (n, 5)
        target_boxes: reference box in fortmat (xc, yc, w, h, angle). Shape (n, 5)
        n_samples: number of points to sample along box perimeter.

    Returns:
        saf: mean sdf difference, i.e. the area of <target_boxes> diff <pred_boxes>.
    """

    # from (xc, yc, w, h, angle) -> (x1,y1, x2,y2, x3,y3, x4,y4)
    poly = obb_to_corners_format(pred_boxes, degrees=False)
    factors2 = torch.arange(n_samples, device=pred_boxes.device, dtype=pred_boxes.dtype) / n_samples
    factors1 = 1.0 - factors2

    center = target_boxes[:, :2]  # (m, 2)
    cos = torch.cos(target_boxes[:, -1])  # (m, 1)
    sin = torch.sin(target_boxes[:, -1])  # (m, 1)

    # linearly sample n_samples points along each edge
    poly_next = torch.roll(poly, -1, dims=1)  # (n, 4, 2)
    pnew = (
        poly[:, None, :, :] * factors1[None, :, None, None] + poly_next[:, None, :, :] * factors2[None, :, None, None]
    )
    pnew = pnew - center[:, None, None, :]  # (n, n_samples, 4, 2)

    ppx = pnew[..., 0] * cos[:, None, None] + pnew[..., 1] * sin[:, None, None]  # (n, n_samples, 4)
    ppy = -pnew[..., 0] * sin[:, None, None] + pnew[..., 1] * cos[:, None, None]  # (n, n_samples, 4)
    ppxy = torch.stack([ppx, ppy], dim=-1)  # (n, n_samples, 4, 2)
    qqxy = torch.abs(ppxy) - 0.5 * target_boxes[:, None, None, 2:4]  # (n, n_samples, 4, 2)

    sign = qqxy[..., 0] > 0  # (n, n_samples, 4)
    zeros = torch.zeros_like(qqxy[..., 0])
    x_comp = torch.maximum(qqxy[..., 0], zeros) * sign * torch.sign(ppxy[..., 0])  # (n, n_samples, 4)
    y_comp = torch.maximum(qqxy[..., 1], zeros) * (~sign) * torch.sign(ppxy[..., 1])

    dx = torch.gradient(ppx, dim=1)[0]  # (n, n_samples, 4)
    dy = torch.gradient(ppy, dim=1)[0]  # (n, n_samples, 4)

    safii = x_comp * dy - y_comp * dx
    return safii.sum(dim=[-2, -1], dtype=pred_boxes.dtype)


def _pairwise_saf_obox2obox(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, n_samples: int = 40) -> torch.Tensor:
    """Signed area difference between 2 sets of oriented boxes, vectorized

    Args:
        pred_boxes: prediction oboxes, in shape (n, 5). Columns: (xc, yc, w, h, angle).
        target_boxes: target oboxes, in shape (m, 5). Columns: (xc, yc, w, h, angle).
        n_samples: number of samples along each edge.

    Returns:
        saf: (n, m) array, mean saf between pairs of pred/target.
    """

    # from (xc, yc, w, h, angle) -> (x1,y1, x2,y2, x3,y3, x4,y4)
    poly = obb_to_corners_format(pred_boxes, degrees=False)
    factors2 = torch.arange(n_samples, device=pred_boxes.device, dtype=pred_boxes.dtype) / n_samples
    factors1 = 1.0 - factors2

    center = target_boxes[:, :2]  # (m, 2)
    cos = torch.cos(target_boxes[:, -1])  # (m, 1)
    sin = torch.sin(target_boxes[:, -1])  # (m, 1)
    saf = torch.zeros((pred_boxes.shape[0], target_boxes.shape[0]), device=pred_boxes.device, dtype=pred_boxes.dtype)

    for i1 in range(4):
        # linearly sample n_samples points along each edge
        i2 = (i1 + 1) % 4
        p1 = poly[:, i1, :]
        p2 = poly[:, i2, :]
        pnew = p1[:, None, :] * factors1[None, :, None] + p2[:, None, :] * factors2[None, :, None]
        pnew = pnew[:, None, :, :] - center[None, :, None, :]  # (n, m, n_samples, 2)

        ppx = pnew[..., 0] * cos[None, :, None] + pnew[..., 1] * sin[None, :, None]  # (n, m, n_samples)
        ppy = -pnew[..., 0] * sin[None, :, None] + pnew[..., 1] * cos[None, :, None]  # (n, m, n_samples)
        ppxy = torch.stack([ppx, ppy], dim=3)  # [n, m, n_samples, 2)
        qqxy = torch.abs(ppxy) - 0.5 * target_boxes[None, :, None, 2:4]  # (n, m, n_samples, 2)

        sign = qqxy[..., 0] > 0  # (n, m, n_samples]
        zeros = torch.zeros_like(qqxy[..., 0])
        x_comp = torch.maximum(qqxy[..., 0], zeros) * sign * torch.sign(ppxy[..., 0])  # (n, m, n_samples)
        y_comp = torch.maximum(qqxy[..., 1], zeros) * (~sign) * torch.sign(ppxy[..., 1])

        dx = torch.gradient(ppx, dim=-1)[0]  # (n, m, n_samples)
        dy = torch.gradient(ppy, dim=-1)[0]  # (n, m, n_samples)

        safii = x_comp * dy - y_comp * dx
        saf = saf + safii.sum(axis=-1)

    return saf
