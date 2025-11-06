"""Base class for Rotated IoU computation.

Provides common functionality for AABB filtering and candidate selection.
"""

from abc import ABC, abstractmethod

import torch


class BaseRotatedIoU(ABC):
    """Base class for rotated IoU computation with common preprocessing steps.

    Args:
        eps: Small constant for numerical stability
    """

    def __init__(self, eps: float = 1e-7):
        self.eps = eps

    def __call__(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU between rotated boxes.

        Args:
            pred_boxes: [N, 5] (x, y, w, h, angle)
            target_boxes: [N, 5] (x, y, w, h, angle)

        Returns:
            [N] IoU values
        """
        N = pred_boxes.shape[0]
        if N == 0:
            return torch.empty(0, device=pred_boxes.device, dtype=pred_boxes.dtype)

        # Step 1: AABB filtering to find overlap candidates
        overlap_mask = self._check_aabb_overlap(pred_boxes, target_boxes)
        ious = torch.zeros(N, device=pred_boxes.device, dtype=pred_boxes.dtype)

        if not overlap_mask.any():
            return ious

        # Step 2: Get candidates that passed AABB filtering
        candidates = torch.where(overlap_mask)[0]
        pred_candidates = pred_boxes[candidates]
        target_candidates = target_boxes[candidates]

        # Step 3: Compute IoU for candidates (implementation-specific)
        candidate_ious = self._compute_candidate_ious(pred_candidates, target_candidates)
        ious[candidates] = candidate_ious

        return ious

    @abstractmethod
    def _compute_candidate_ious(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU for candidate box pairs.

        This method must be implemented by subclasses to define the specific
        IoU computation strategy (approximate sampling, exact polygon intersection, etc.).

        Args:
            pred_boxes: [M, 5] candidate predicted boxes
            target_boxes: [M, 5] candidate target boxes

        Returns:
            [M] IoU values for candidates
        """

    def _check_aabb_overlap(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Check for axis-aligned bounding box overlap.

        Args:
            boxes1: [N, 5] first set of boxes
            boxes2: [N, 5] second set of boxes

        Returns:
            [N] boolean mask indicating which box pairs might overlap
        """
        bounds1 = self._compute_aabb_bounds(boxes1)
        bounds2 = self._compute_aabb_bounds(boxes2)

        no_overlap_x = (bounds1[:, 1] < bounds2[:, 0]) | (bounds2[:, 1] < bounds1[:, 0])
        no_overlap_y = (bounds1[:, 3] < bounds2[:, 2]) | (bounds2[:, 3] < bounds1[:, 2])

        return ~(no_overlap_x | no_overlap_y)

    def _compute_aabb_bounds(self, boxes: torch.Tensor) -> torch.Tensor:
        """Compute axis-aligned bounding box bounds for rotated boxes.

        Args:
            boxes: [N, 5] (x, y, w, h, angle)

        Returns:
            [N, 4] bounds (min_x, max_x, min_y, max_y)
        """
        x, y, w, h, angle = boxes.unbind(-1)
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)

        # Half extents after rotation
        ext_x = 0.5 * (w * torch.abs(cos_a) + h * torch.abs(sin_a))
        ext_y = 0.5 * (w * torch.abs(sin_a) + h * torch.abs(cos_a))

        min_x = x - ext_x
        max_x = x + ext_x
        min_y = y - ext_y
        max_y = y + ext_y

        return torch.stack([min_x, max_x, min_y, max_y], dim=-1)
