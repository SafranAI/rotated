"""Rotated IoU Approximation with Sampling Strategy.

Uses adaptive and stratified sampling to improve accuracy over uniform random sampling."""

import torch

from rotated.iou.base import BaseRotatedIoU


class ApproxRotatedIoU(BaseRotatedIoU):
    """Rotated IoU approximation with sampling strategy.

    Args:
        base_samples: Base number of samples (will be adapted per box pair)
        eps: Small constant for numerical stability
    """

    def __init__(self, base_samples: int = 2000, eps: float = 1e-7):
        super().__init__(eps=eps)
        self.base_samples = base_samples

    def _compute_candidate_ious(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU for candidate box pairs using sampling strategy.

        Args:
            pred_boxes: [M, 5] candidate predicted boxes
            target_boxes: [M, 5] candidate target boxes

        Returns:
            [M] IoU values for candidates
        """
        return self._improved_sampling_iou(pred_boxes, target_boxes)

    def _improved_sampling_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute IoU using sampling strategy."""
        N = boxes1.shape[0]
        device = boxes1.device

        if N == 0:
            return torch.empty(0, device=device, dtype=boxes1.dtype)

        # Box areas
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]

        # Adaptive sample count based on box sizes
        sample_counts = self._compute_adaptive_sample_counts(boxes1, boxes2)

        # Compute intersection estimates for each box pair
        intersection_areas = torch.zeros(N, device=device, dtype=boxes1.dtype)

        for i in range(N):
            intersection_areas[i] = self._estimate_intersection_area(boxes1[i], boxes2[i], sample_counts[i])

        # Compute IoU
        union_areas = area1 + area2 - intersection_areas
        ious = torch.where(
            union_areas > self.eps, intersection_areas / union_areas, torch.zeros_like(intersection_areas)
        )

        return torch.clamp(ious, 0.0, 1.0)

    def _compute_adaptive_sample_counts(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute adaptive sample count based on box characteristics."""
        # Simplified adaptive sampling - just scale by average area
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]
        avg_area = (area1 + area2) * 0.5

        # Scale factor based on area (but keep it simple)
        area_factor = torch.sqrt(avg_area / 100.0)
        area_factor = torch.clamp(area_factor, 0.7, 1.5)

        sample_counts = (self.base_samples * area_factor).long()
        sample_counts = torch.clamp(sample_counts, 1000, 4000)

        return sample_counts

    def _estimate_intersection_area(self, box1: torch.Tensor, box2: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Estimate intersection area using stratified sampling."""
        device = box1.device
        dtype = box1.dtype

        # Get the intersection region bounds
        bounds1 = self._compute_single_box_bounds(box1)
        bounds2 = self._compute_single_box_bounds(box2)

        # Intersection bounds
        min_x = max(bounds1[0], bounds2[0])
        max_x = min(bounds1[1], bounds2[1])
        min_y = max(bounds1[2], bounds2[2])
        max_y = min(bounds1[3], bounds2[3])

        if max_x <= min_x or max_y <= min_y:
            return torch.zeros(1, device=device, dtype=dtype).squeeze()

        # Simpler stratified sampling - use reasonable grid size
        grid_size = min(int((num_samples / 100) ** 0.5), 20)  # Cap at 20x20 grid
        grid_size = max(grid_size, 4)

        total_samples = grid_size * grid_size * 4  # 4 samples per cell

        # Create grid coordinates
        i_coords = torch.arange(grid_size, device=device, dtype=dtype).repeat_interleave(grid_size * 4)
        j_coords = torch.arange(grid_size, device=device, dtype=dtype).repeat(grid_size).repeat_interleave(4)

        # Random offsets within each cell
        rand_offsets = torch.rand(total_samples, 2, device=device, dtype=dtype)

        # Compute sample coordinates
        cell_width = (max_x - min_x) / grid_size
        cell_height = (max_y - min_y) / grid_size

        samples_x = min_x + (i_coords + rand_offsets[:, 0]) * cell_width
        samples_y = min_y + (j_coords + rand_offsets[:, 1]) * cell_height
        samples = torch.stack([samples_x, samples_y], dim=-1)

        # Check if samples are in both boxes
        in_box1 = self._point_in_rotated_box(samples, box1)
        in_box2 = self._point_in_rotated_box(samples, box2)

        intersection_count = (in_box1 & in_box2).sum().float()

        # Estimate intersection area
        sampling_region_area = (max_x - min_x) * (max_y - min_y)
        intersection_area = (intersection_count / total_samples) * sampling_region_area

        return intersection_area

    def _compute_single_box_bounds(self, box: torch.Tensor) -> tuple:
        """Compute AABB bounds for a single box."""
        x, y, w, h, angle = box
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)

        ext_x = 0.5 * (w * torch.abs(cos_a) + h * torch.abs(sin_a))
        ext_y = 0.5 * (w * torch.abs(sin_a) + h * torch.abs(cos_a))

        return (x - ext_x).item(), (x + ext_x).item(), (y - ext_y).item(), (y + ext_y).item()

    def _point_in_rotated_box(self, points: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """Check if points are inside a rotated box."""
        x, y, w, h, angle = box

        # Transform points to box-local coordinates
        local_points = points - torch.stack([x, y])

        # Rotate points to align with box axes
        cos_a = torch.cos(-angle)
        sin_a = torch.sin(-angle)

        rotated_x = local_points[:, 0] * cos_a - local_points[:, 1] * sin_a
        rotated_y = local_points[:, 0] * sin_a + local_points[:, 1] * cos_a

        # Check if within box bounds
        half_w, half_h = w * 0.5, h * 0.5
        inside_x = torch.abs(rotated_x) <= half_w
        inside_y = torch.abs(rotated_y) <= half_h

        return inside_x & inside_y
