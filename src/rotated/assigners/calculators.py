from typing import TYPE_CHECKING

import torch

from rotated.boxes.utils import check_is_valid_box

if TYPE_CHECKING:
    from rotated.iou import IoUKwargs, IoUMethodName


class RotatedIoUCalculator:
    """IoU calculator for rotated bounding boxes using ProbIoU.

    Args:
        iou_method: Method name to compute Intersection Over Union
        iou_kwargs: Dictionary with parameters for the IoU method
        min_box_size: Minimum valid box dimension (width or height). Boxes smaller than this are invalid.
    """

    def __init__(
        self,
        iou_method: "IoUMethodName" = "prob_iou",
        iou_kwargs: "IoUKwargs" = None,
        min_box_size: float = 1e-2,
    ):
        from rotated.iou import iou_picker

        self.iou_calculator = iou_picker(iou_method=iou_method, iou_kwargs=iou_kwargs)
        self.min_box_size = min_box_size

    def __call__(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise IoU matrix between two sets of rotated boxes.

        Args:
            boxes1: First set of rotated boxes [N, 5] - (cx, cy, w, h, angle)
            boxes2: Second set of rotated boxes [M, 5] - (cx, cy, w, h, angle)

        Returns:
            IoU matrix [N, M] with values in [0, 1]
        """
        N, M = boxes1.shape[0], boxes2.shape[0]

        if N == 0 or M == 0:
            return torch.zeros(N, M, device=boxes1.device, dtype=boxes1.dtype)

        # Identify valid boxes (non-degenerate)
        valid_boxes1 = check_is_valid_box(boxes1, self.min_box_size)  # [N]
        valid_boxes2 = check_is_valid_box(boxes2, self.min_box_size)  # [M]

        # Expand for pairwise computation
        boxes1_expanded = boxes1.unsqueeze(1).expand(-1, M, -1)  # [N, M, 5]
        boxes2_expanded = boxes2.unsqueeze(0).expand(N, -1, -1)  # [N, M, 5]

        # Create validity mask for pairs
        valid_pairs = valid_boxes1.unsqueeze(1) & valid_boxes2.unsqueeze(0)  # [N, M]

        # Flatten for element-wise computation
        boxes1_flat = boxes1_expanded.reshape(-1, 5)  # [N*M, 5]
        boxes2_flat = boxes2_expanded.reshape(-1, 5)  # [N*M, 5]

        # Compute element-wise IoU (will be 0 for invalid boxes anyway)
        iou_flat = self.iou_calculator(boxes1_flat, boxes2_flat)  # [N*M]

        # Reshape back to matrix
        iou_matrix = iou_flat.view(N, M)  # [N, M]

        # Zero out IoU for invalid box pairs
        iou_matrix = torch.where(valid_pairs, iou_matrix, torch.zeros_like(iou_matrix))

        return iou_matrix.clamp_(0.0, 1.0)


class HorizontalIoUCalculator:
    """IoU calculator for horizontal (axis-aligned) bounding boxes.

    Args:
        eps: Small constant to prevent division by zero
        min_box_size: Minimum valid box dimension (width or height). Boxes smaller than this are invalid.
    """

    def __init__(self, eps: float = 1e-7, min_box_size: float = 1e-2):
        self.eps = eps
        self.min_box_size = min_box_size

    def __call__(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise IoU matrix between two sets of horizontal boxes.

        Args:
            boxes1: First set of boxes [N, 4] - (cx, cy, w, h)
            boxes2: Second set of boxes [M, 4] - (cx, cy, w, h)

        Returns:
            IoU matrix [N, M] with values in [0, 1]
        """
        N, M = boxes1.shape[0], boxes2.shape[0]

        if N == 0 or M == 0:
            return torch.zeros(N, M, device=boxes1.device, dtype=boxes1.dtype)

        # Check validity
        valid_boxes1 = check_is_valid_box(boxes1, self.min_box_size)
        valid_boxes2 = check_is_valid_box(boxes2, self.min_box_size)
        valid_pairs = valid_boxes1.unsqueeze(1) & valid_boxes2.unsqueeze(0)

        boxes1_xyxy = self._cxcywh_to_xyxy(boxes1)
        boxes2_xyxy = self._cxcywh_to_xyxy(boxes2)

        # Compute intersection
        x1_inter = torch.max(boxes1_xyxy[:, None, 0], boxes2_xyxy[None, :, 0])
        y1_inter = torch.max(boxes1_xyxy[:, None, 1], boxes2_xyxy[None, :, 1])
        x2_inter = torch.min(boxes1_xyxy[:, None, 2], boxes2_xyxy[None, :, 2])
        y2_inter = torch.min(boxes1_xyxy[:, None, 3], boxes2_xyxy[None, :, 3])

        inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)

        # Box areas
        area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
        area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])

        union_area = area1[:, None] + area2[None, :] - inter_area

        # IoU
        iou = inter_area / (union_area + self.eps)

        # Zero out invalid pairs
        iou = torch.where(valid_pairs, iou, torch.zeros_like(iou))

        return torch.clamp_(iou, 0.0, 1.0)

    def _cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert center format to corner format."""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5
        return torch.stack([x1, y1, x2, y2], dim=-1)
