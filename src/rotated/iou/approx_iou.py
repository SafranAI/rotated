"""Rotated IoU Approximation with Sampling Strategy.

Uses adaptive and stratified sampling to improve accuracy over uniform random sampling."""

import torch

from rotated.boxes.utils import check_aabb_overlap


# NOTE: Not relying on a Base class as we encountered issues with
# torchscript export, when using inheritance, with NMS / post-processing
class ApproxRotatedIoU:
    """Rotated IoU approximation with sampling strategy.

    Args:
        base_samples: Base number of samples (will be adapted per box pair)
        eps: Small constant for numerical stability
    """

    def __init__(self, base_samples: int = 4000, eps: float = 1e-7):
        self.base_samples = base_samples
        self.eps = eps

    def __call__(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute approximate IoU between rotated boxes.

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
        overlap_mask = check_aabb_overlap(pred_boxes, target_boxes)
        ious = torch.zeros(N, device=pred_boxes.device, dtype=pred_boxes.dtype)

        if not overlap_mask.any():
            return ious

        # Step 2: Get candidates that passed AABB filtering
        candidates = torch.where(overlap_mask)[0]
        pred_candidates = pred_boxes[candidates]
        target_candidates = target_boxes[candidates]

        # Step 3: Sampling-based IoU
        candidate_ious = self._compute_candidate_ious(pred_candidates, target_candidates)
        ious[candidates] = candidate_ious

        return ious

    def _compute_candidate_ious(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU for candidate box pairs using sampling strategy.

        Args:
            pred_boxes: [M, 5] candidate predicted boxes
            target_boxes: [M, 5] candidate target boxes

        Returns:
            [M] IoU values for candidates
        """
        M = pred_boxes.shape[0]
        device = pred_boxes.device

        if M == 0:
            return torch.empty(0, device=device, dtype=pred_boxes.dtype)

        # Box areas
        area1 = pred_boxes[:, 2] * pred_boxes[:, 3]
        area2 = target_boxes[:, 2] * target_boxes[:, 3]

        # Adaptive sample count based on box sizes
        intersection_areas = self._estimate_intersection_area(pred_boxes, target_boxes, self.base_samples)

        # Compute IoU
        union_areas = area1 + area2 - intersection_areas
        ious = torch.where(
            union_areas > self.eps, intersection_areas / union_areas, torch.zeros_like(intersection_areas)
        )

        return torch.clamp(ious, 0.0, 1.0)

    def _estimate_intersection_area(self, box1: torch.Tensor, box2: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Estimate intersection area using stratified sampling."""
        device = box1.device
        dtype = box1.dtype
        N = box1.shape[0]

        # Get the intersection region bounds
        bounds1 = self._compute_multi_box_bounds(box1)
        bounds2 = self._compute_multi_box_bounds(box2)

        # Simpler stratified sampling - use reasonable grid size
        grid_size = min(int((num_samples / 100) ** 0.5), 20)  # Cap at 20x20 grid
        grid_size = max(grid_size, 4)

        total_samples = grid_size**2 * 4  # 4 samples per cell

        # Intersection bounds
        min_x = torch.max(bounds1[0], bounds2[0]).repeat_interleave(total_samples).reshape(N, total_samples)
        max_x = torch.min(bounds1[1], bounds2[1]).repeat_interleave(total_samples).reshape(N, total_samples)
        min_y = torch.max(bounds1[2], bounds2[2]).repeat_interleave(total_samples).reshape(N, total_samples)
        max_y = torch.min(bounds1[3], bounds2[3]).repeat_interleave(total_samples).reshape(N, total_samples)

        # Create grid coordinates
        i_coords = (
            torch.arange(grid_size, device=device, dtype=dtype)
            .repeat_interleave(grid_size * 4)
            .repeat(N)
            .reshape(N, -1)
        )
        j_coords = (
            torch.arange(grid_size, device=device, dtype=dtype)
            .repeat(grid_size)
            .repeat_interleave(4)
            .repeat(N)
            .reshape(N, -1)
        )

        # Random offsets within each cell
        rand_offsets = torch.rand(N, total_samples, 2, device=device, dtype=dtype)

        # Compute sample coordinates
        cell_width = (max_x - min_x) / grid_size
        cell_height = (max_y - min_y) / grid_size

        samples_x = min_x + (i_coords + rand_offsets[:, :, 0]) * cell_width
        samples_y = min_y + (j_coords + rand_offsets[:, :, 1]) * cell_height
        samples = torch.stack([samples_x, samples_y], dim=-1)  # (N, total_samples, 2)

        # Check if samples are in both boxes
        in_box1 = self._points_in_rotated_boxes(samples, box1)
        in_box2 = self._points_in_rotated_boxes(samples, box2)

        intersection_count = (in_box1 & in_box2).sum(-1).float()

        # Estimate intersection area
        sampling_region_area = (max_x - min_x) * (max_y - min_y)
        intersection_area = (intersection_count / total_samples) * sampling_region_area[:, 0]

        return intersection_area

    def _compute_multi_box_bounds(self, boxes: torch.Tensor) -> tuple:
        """Compute AABB bounds for a boxes."""
        x, y, w, h, angle = boxes.unbind(-1)
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)

        ext_x = 0.5 * (w * torch.abs(cos_a) + h * torch.abs(sin_a))
        ext_y = 0.5 * (w * torch.abs(sin_a) + h * torch.abs(cos_a))

        return (x - ext_x), (x + ext_x), (y - ext_y), (y + ext_y)

    def _points_in_rotated_boxes(self, points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Check if points are inside rotated boxes.

        Args:
            points: Tensor of shape (N, num_samples, 2)
            boxes: Tensor of shape (N, 5)

        Returns:
            Boolean tensor indicating if points are inside boxes
        """

        # Transform points to box-local coordinates
        N, num_samples = points.shape[:2]
        local_points = points - boxes[:, :2].repeat((1, num_samples)).reshape(N, num_samples, 2)

        # Rotate points to align with box axes
        cos_a = torch.cos(-boxes[:, 4]).repeat_interleave(num_samples).reshape(N, num_samples)
        sin_a = torch.sin(-boxes[:, 4]).repeat_interleave(num_samples).reshape(N, num_samples)

        rotated_x = local_points[:, :, 0] * cos_a - local_points[:, :, 1] * sin_a
        rotated_y = local_points[:, :, 0] * sin_a + local_points[:, :, 1] * cos_a

        # Check if within box bounds
        half_w, half_h = boxes[:, 2] * 0.5, boxes[:, 3] * 0.5
        inside_x = torch.abs(rotated_x) <= half_w.repeat_interleave(num_samples).reshape(N, num_samples)
        inside_y = torch.abs(rotated_y) <= half_h.repeat_interleave(num_samples).reshape(N, num_samples)

        return inside_x & inside_y
