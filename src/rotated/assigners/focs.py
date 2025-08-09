# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Inspired by FCOS-R implementations from MMRotate and PaddleDetection.
# This implementation follows the core Gaussian-based assignment strategy
# with IoU-based quality scoring as used in the actual FCOS-R configurations.
#
# References:
# - MMRotate FCOS-R: https://github.com/open-mmlab/mmrotate
# - PaddleDetection FCOS-R: https://github.com/PaddlePaddle/PaddleDetection
# - Original Paper: "FCOS-R: A Novel Rotated Object Detector" (arXiv:2111.10780)

import math

import torch
import torch.nn as nn

from rotated.iou.prob_iou import ProbIoU


def rbox_to_poly(rboxes: torch.Tensor) -> torch.Tensor:
    """Convert rotated bounding boxes to polygon corner format.

    Args:
        rboxes: (..., 5) - Rotated boxes [cx, cy, w, h, angle]

    Returns:
        (..., 4, 2) - 4 corner points per box (counter-clockwise from bottom-left)
    """
    *batch_dims, _ = rboxes.shape
    cx, cy, w, h, angle = rboxes.unbind(-1)

    # Corner offsets in local coordinate system (counter-clockwise from bottom-left)
    corners_x = torch.stack([-w/2, w/2, w/2, -w/2], dim=-1)  # (..., 4)
    corners_y = torch.stack([-h/2, -h/2, h/2, h/2], dim=-1)  # (..., 4)

    # Apply rotation
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    cos_a, sin_a = cos_a.unsqueeze(-1), sin_a.unsqueeze(-1)  # (..., 1)

    rot_corners_x = corners_x * cos_a - corners_y * sin_a
    rot_corners_y = corners_x * sin_a + corners_y * cos_a

    # Translate to global coordinates
    global_corners_x = rot_corners_x + cx.unsqueeze(-1)
    global_corners_y = rot_corners_y + cy.unsqueeze(-1)

    # Stack into corner format
    polys = torch.stack([global_corners_x, global_corners_y], dim=-1)  # (..., 4, 2)
    return polys


def points_in_polygons(points: torch.Tensor, polygons: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Check if points are inside polygons using cross-product method.

    Args:
        points: (1, L, 2) - Points to test
        polygons: (B, n, 4, 2) - Polygon vertices (counter-clockwise)
        eps: Small value for numerical stability

    Returns:
        (B, n, L) - Boolean mask indicating point containment
    """
    batch_size, num_polygons, _, _ = polygons.shape
    num_points = points.shape[1]

    # Expand for pairwise computation
    points_exp = points.unsqueeze(1).expand(-1, num_polygons, -1, -1)  # (1, n, L, 2)
    polygons_exp = polygons.unsqueeze(2).expand(-1, -1, num_points, -1, -1)  # (B, n, L, 4, 2)

    # Initialize inside mask (all points start as inside)
    inside_mask = torch.ones(batch_size, num_polygons, num_points, dtype=torch.bool, device=points.device)

    # Check against each polygon edge using cross product
    for i in range(4):
        j = (i + 1) % 4

        # Current edge from vertex i to vertex j
        edge_start = polygons_exp[..., i, :]  # (B, n, L, 2)
        edge_end = polygons_exp[..., j, :]    # (B, n, L, 2)

        # Vectors
        to_point = points_exp[0] - edge_start  # (B, n, L, 2)
        edge_vec = edge_end - edge_start       # (B, n, L, 2)

        # 2D cross product (point should be on left/inside of directed edge)
        cross = edge_vec[..., 0] * to_point[..., 1] - edge_vec[..., 1] * to_point[..., 0]
        inside_mask = inside_mask & (cross >= -eps)  # Allow small numerical errors

    return inside_mask


class RotatedGaussianAssigner(nn.Module):
    """Gaussian-based assigner for rotated object detection.

    Implements the FCOS-R assignment strategy using Gaussian distribution scoring
    for rotated objects. The assignment process:

    1. Convert rotated boxes to corner polygons for geometric operations
    2. Compute Gaussian distribution scores based on normalized distance to GT centers
    3. Apply geometric constraints (inside polygon + score threshold)
    4. Apply multi-level range constraints with rotated object adaptations
    5. Select best GT assignment using refined Gaussian scores
    6. Compute IoU scores between predictions and assigned GTs for loss weighting

    The IoU scores are always computed for both classification and regression
    loss weighting as per the standard FCOS-R configuration.

    Args:
        num_classes: Number of object classes
        gaussian_factor: Controls Gaussian distribution spread (default: 12.0)
        score_threshold: Minimum score threshold for positive assignment (default: 0.23)
        regress_ranges: Distance boundaries per FPN level
        use_refined_scoring: Enable refined score normalization (default: True)
    """

    def __init__(
        self,
        num_classes: int,
        gaussian_factor: float = 12.0,
        score_threshold: float = 0.23,
        regress_ranges: list[tuple[float, float]] = None,
        use_refined_scoring: bool = True,
    ):
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if gaussian_factor <= 0:
            raise ValueError(f"gaussian_factor must be positive, got {gaussian_factor}")
        if score_threshold < 0 or score_threshold > 1:
            raise ValueError(f"score_threshold must be in [0, 1], got {score_threshold}")

        self.num_classes = num_classes
        self.gaussian_factor = gaussian_factor
        self.score_threshold = score_threshold
        self.use_refined_scoring = use_refined_scoring

        # Default regression ranges (matching FCOS-R config)
        if regress_ranges is None:
            regress_ranges = [(-1, 64), (64, 128), (128, 256), (256, 512), (512, 1e8)]

        # Store boundaries as buffers for device management
        for i, (low, high) in enumerate(regress_ranges):
            self.register_buffer(f'range_{i}', torch.tensor([low, high], dtype=torch.float32))

        self.num_levels = len(regress_ranges)
        self.eps = 1e-9

        # IoU calculator for quality scoring
        self.iou_calculator = ProbIoU()

    @torch.no_grad()
    def forward(
        self,
        pred_scores: torch.Tensor,
        pred_boxes: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
        pad_gt_mask: torch.Tensor,
        bg_index: int,
        stride_tensor: torch.Tensor = None,
        num_anchors_per_level: list[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign GT objects to anchor points using Gaussian scoring.

        Args:
            pred_scores: (B, L, C) - Predicted classification scores (post-sigmoid)
            pred_boxes: (B, L, 5) - Predicted rotated boxes [cx, cy, w, h, angle]
            anchor_points: (1, L, 2) - Anchor point coordinates in absolute pixels
            gt_labels: (B, N, 1) - Ground truth class labels
            gt_boxes: (B, N, 5) - Ground truth rotated boxes [cx, cy, w, h, angle]
            pad_gt_mask: (B, N, 1) - Valid GT mask (1=valid, 0=padding)
            bg_index: Background class index
            stride_tensor: (1, L, 1) - Stride values per anchor (optional)
            num_anchors_per_level: Number of anchors per FPN level (optional)

        Returns:
            assigned_labels: (B, L) - Assigned class labels per anchor
            assigned_boxes: (B, L, 5) - Assigned GT rotated boxes per anchor
            assigned_scores: (B, L, C) - IoU-weighted quality scores per anchor
        """
        batch_size, num_anchors, num_classes = pred_scores.shape
        num_max_boxes = gt_boxes.shape[1]

        # Handle empty GT case
        if num_max_boxes == 0:
            return self._handle_empty_gt(batch_size, num_anchors, num_classes, pred_scores.device, bg_index)

        # For FCOS-R, we need stride information for multi-level constraints
        # If not provided, create default stride tensor
        if stride_tensor is None:
            stride_tensor = torch.ones(1, num_anchors, 1, device=anchor_points.device) * 16.0
        if num_anchors_per_level is None:
            num_anchors_per_level = [num_anchors]  # Single level

        # Convert rotated boxes to corner format for geometric operations
        gt_polys = rbox_to_poly(gt_boxes)

        # Compute Gaussian distribution scores
        gaussian_scores, refined_scores = self._compute_gaussian_scores(
            anchor_points, gt_boxes, gt_polys
        )

        # Apply geometric constraints (inside polygon + score threshold)
        inside_mask = self._apply_geometric_constraints(anchor_points, gt_polys, gaussian_scores)

        # Apply multi-level range constraints
        # Create axis-aligned approximation for range constraints
        gt_bboxes = self._rboxes_to_axis_aligned(gt_boxes)
        regress_ranges = self._build_regression_ranges(anchor_points.device, num_anchors_per_level)
        range_mask = self._apply_range_constraints(
            anchor_points, gt_bboxes, gt_boxes, stride_tensor, regress_ranges
        )

        # Combine all constraints for positive assignment candidates
        positive_mask = inside_mask * range_mask * pad_gt_mask.transpose(1, 2)

        # Apply constraints to refined scores and find best assignments
        masked_scores = refined_scores * positive_mask - (1.0 - positive_mask)
        best_gt_indices = masked_scores.argmax(dim=1)  # (B, L)
        best_scores = masked_scores.max(dim=1)[0]  # (B, L)

        # Create final assignments
        assigned_labels, assigned_boxes = self._create_assignments(
            best_gt_indices, best_scores, gt_labels, gt_boxes, bg_index, batch_size, num_max_boxes
        )

        # Create IoU-weighted quality scores (always computed for loss weighting)
        assigned_scores = self._create_iou_weighted_scores(
            assigned_labels, assigned_boxes, pred_boxes, bg_index
        )

        return assigned_labels, assigned_boxes, assigned_scores

    def _handle_empty_gt(
        self, batch_size: int, num_anchors: int, num_classes: int, device: torch.device, bg_index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Handle case when no ground truth objects are present."""
        assigned_labels = torch.full((batch_size, num_anchors), bg_index, dtype=torch.long, device=device)
        assigned_boxes = torch.zeros(batch_size, num_anchors, 5, device=device)
        assigned_scores = torch.zeros(batch_size, num_anchors, num_classes, device=device)
        return assigned_labels, assigned_boxes, assigned_scores

    def _rboxes_to_axis_aligned(self, gt_rboxes: torch.Tensor) -> torch.Tensor:
        """Convert rotated boxes to axis-aligned approximation for range constraints.

        Args:
            gt_rboxes: (B, N, 5) - [cx, cy, w, h, angle]

        Returns:
            (B, N, 4) - Axis-aligned boxes [x1, y1, x2, y2]
        """
        cx, cy, w, h, angle = gt_rboxes.unbind(-1)

        # Compute axis-aligned bounding box of rotated box
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)

        # Half extents of rotated box
        extent_w = 0.5 * (w * torch.abs(cos_a) + h * torch.abs(sin_a))
        extent_h = 0.5 * (w * torch.abs(sin_a) + h * torch.abs(cos_a))

        # Axis-aligned bounding box
        x1 = cx - extent_w
        y1 = cy - extent_h
        x2 = cx + extent_w
        y2 = cy + extent_h

        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _compute_gaussian_scores(
        self, anchor_points: torch.Tensor, gt_rboxes: torch.Tensor, gt_polys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Gaussian distribution scores following paddle implementation.

        Uses the ab/ad edge vector approach for projecting points to rotated coordinate systems
        and computing normalized Gaussian distributions based on distance to GT centers.

        Args:
            anchor_points: (1, L, 2) - Anchor points
            gt_rboxes: (B, n, 5) - Rotated boxes
            gt_polys: (B, n, 4, 2) - Corner polygons

        Returns:
            gaussian_scores: (B, n, L) - Normalized Gaussian scores
            refined_scores: (B, n, L) - Refined scores for assignment
        """
        # Extract corner vectors for coordinate system definition
        polys_corners = gt_polys  # (B, n, 4, 2)
        a = polys_corners[..., 0, :]  # (B, n, 2) - bottom-left corner
        b = polys_corners[..., 1, :]  # (B, n, 2) - bottom-right corner
        d = polys_corners[..., 3, :]  # (B, n, 2) - top-left corner

        # Compute edge vectors
        vec_ab = b - a  # (B, n, 2) - bottom edge vector
        vec_ad = d - a  # (B, n, 2) - left edge vector

        # Get box centers and dimensions
        gt_centers = gt_rboxes[..., :2]  # (B, n, 2)
        gt_dims = gt_rboxes[..., 2:4]   # (B, n, 2)

        # Expand anchor points for pairwise computation
        points_expanded = anchor_points.unsqueeze(0)  # (1, 1, L, 2)
        centers_expanded = gt_centers.unsqueeze(2)     # (B, n, 1, 2)

        # Vector from GT center to each anchor point
        center_to_point = points_expanded - centers_expanded  # (B, n, L, 2)

        # Project onto edge directions
        vec_ab_expanded = vec_ab.unsqueeze(2)  # (B, n, 1, 2)
        vec_ad_expanded = vec_ad.unsqueeze(2)  # (B, n, 1, 2)

        # Dot products for projection magnitudes
        dot_ab = torch.sum(center_to_point * vec_ab_expanded, dim=-1)  # (B, n, L)
        dot_ad = torch.sum(center_to_point * vec_ad_expanded, dim=-1)  # (B, n, L)

        # Edge norms
        norm_ab = torch.sqrt(torch.sum(vec_ab**2, dim=-1) + self.eps).unsqueeze(2)  # (B, n, 1)
        norm_ad = torch.sqrt(torch.sum(vec_ad**2, dim=-1) + self.eps).unsqueeze(2)  # (B, n, 1)

        # Minimum edge for normalization
        min_edge = torch.min(gt_dims, dim=-1)[0].unsqueeze(2)  # (B, n, 1)

        # Gaussian distribution computation (following paddle's exact formula)
        delta_x = dot_ab.pow(2) / (norm_ab.pow(3) * min_edge + self.eps)
        delta_y = dot_ad.pow(2) / (norm_ad.pow(3) * min_edge + self.eps)

        # Compute normalized Gaussian scores
        gaussian_scores = torch.exp(-0.5 * self.gaussian_factor * (delta_x + delta_y))

        # Compute refined scores with normalization (if enabled)
        if self.use_refined_scoring:
            sigma = min_edge / self.gaussian_factor
            refined_scores = gaussian_scores / (2 * math.pi * sigma + self.eps)
        else:
            refined_scores = gaussian_scores.clone()

        return gaussian_scores, refined_scores

    def _build_regression_ranges(self, device: torch.device, num_anchors_per_level: list[int]) -> torch.Tensor:
        """Build regression ranges for all anchor levels."""
        regress_ranges = []

        for level_idx, num_anchors in enumerate(num_anchors_per_level):
            if level_idx < self.num_levels:
                range_values = getattr(self, f'range_{level_idx}')
            else:
                range_values = getattr(self, f'range_{self.num_levels-1}')  # Use last range

            level_ranges = range_values.view(1, 1, 2).expand(1, num_anchors, -1)
            regress_ranges.append(level_ranges)

        return torch.cat(regress_ranges, dim=1)  # (1, L, 2)

    def _apply_range_constraints(
        self,
        anchor_points: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_rboxes: torch.Tensor,
        stride_tensor: torch.Tensor,
        regress_ranges: torch.Tensor,
    ) -> torch.Tensor:
        """Apply multi-level range constraints with rotated object adaptations.

        Follows the paddle implementation approach with both standard FCOS range
        constraints and additional constraints for rotated objects.
        """
        # Standard FCOS range constraint using axis-aligned boxes
        points_expanded = anchor_points.unsqueeze(0)  # (1, 1, L, 2)
        x1y1, x2y2 = gt_bboxes.unsqueeze(2).split(2, dim=-1)  # (B, n, 1, 2) each

        # Compute distances to box boundaries
        lt = points_expanded - x1y1  # left, top distances
        rb = x2y2 - points_expanded   # right, bottom distances
        ltrb = torch.cat([lt, rb], dim=-1)  # (B, n, L, 4)

        # Inside axis-aligned box constraint
        inside_constraint = torch.min(ltrb, dim=-1)[0] > self.eps

        # Range constraint based on maximum distance
        max_distance = torch.max(ltrb, dim=-1)[0]  # (B, n, L)

        regress_ranges_exp = regress_ranges.unsqueeze(0)  # (1, 1, L, 2)
        range_low, range_high = regress_ranges_exp[..., 0], regress_ranges_exp[..., 1]

        standard_range_mask = (max_distance >= range_low) & (max_distance <= range_high)

        # Additional constraint for rotated objects (from paddle implementation)
        stride_values = stride_tensor.transpose(1, 2)  # (1, 1, L)
        min_gt_edge = torch.min(gt_rboxes[..., 2:4], dim=-1)[0].unsqueeze(2)  # (B, n, 1)

        # Allow assignment for small objects relative to stride, even outside standard range
        rotated_constraint = ((min_gt_edge / stride_values) < 2.0) & (max_distance > range_high)

        # Combine constraints
        final_range_mask = inside_constraint & (standard_range_mask | rotated_constraint)
        return final_range_mask.float()

    def _create_assignments(
        self,
        best_gt_indices: torch.Tensor,
        best_scores: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_rboxes: torch.Tensor,
        bg_index: int,
        batch_size: int,
        num_max_boxes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create final label and box assignments."""
        # Create batch indices for gathering
        batch_indices = torch.arange(batch_size, dtype=best_gt_indices.dtype, device=best_gt_indices.device)
        batch_indices = batch_indices.unsqueeze(1)  # (B, 1)

        # Flatten indices for gathering
        flat_gt_indices = best_gt_indices + (batch_indices * num_max_boxes)

        # Gather labels and boxes
        flat_labels = gt_labels.flatten()  # (B*n,)
        flat_rboxes = gt_rboxes.view(-1, 5)  # (B*n, 5)

        assigned_labels = flat_labels[flat_gt_indices.flatten()].view(batch_size, -1)
        assigned_rboxes = flat_rboxes[flat_gt_indices.flatten()].view(batch_size, -1, 5)

        # Set background for low-score assignments
        background_mask = best_scores <= 0
        assigned_labels = torch.where(background_mask, bg_index, assigned_labels)

        return assigned_labels, assigned_rboxes

    def _create_iou_weighted_scores(
        self,
        assigned_labels: torch.Tensor,
        assigned_boxes: torch.Tensor,
        pred_boxes: torch.Tensor,
        bg_index: int,
    ) -> torch.Tensor:
        """Create IoU-weighted quality scores for loss computation.

        Always computes IoU scores between predictions and assigned GTs,
        as per the standard FCOS-R configuration where both classify_score
        and regress_weight use IoU-based weighting.

        Args:
            assigned_labels: (B, L) - Assigned class labels
            assigned_boxes: (B, L, 5) - Assigned GT rotated boxes
            pred_boxes: (B, L, 5) - Predicted rotated boxes
            bg_index: Background class index

        Returns:
            (B, L, C) - IoU-weighted one-hot quality scores
        """
        batch_size, num_anchors = assigned_labels.shape
        device = assigned_labels.device

        # Create one-hot encoding for classes
        one_hot_scores = torch.nn.functional.one_hot(assigned_labels, self.num_classes + 1).float()

        # Remove background class column
        if bg_index < self.num_classes:
            quality_scores = torch.cat([one_hot_scores[..., :bg_index], one_hot_scores[..., bg_index+1:]], dim=-1)
        else:
            quality_scores = one_hot_scores[..., :-1]

        # Compute IoU weights if predictions are available
        if pred_boxes is not None:
            iou_weights = self._compute_pred_gt_ious(pred_boxes, assigned_boxes, assigned_labels, bg_index)
            quality_scores = quality_scores * iou_weights.unsqueeze(-1)

        return quality_scores

    def _compute_pred_gt_ious(
        self,
        pred_boxes: torch.Tensor,
        assigned_boxes: torch.Tensor,
        assigned_labels: torch.Tensor,
        bg_index: int,
    ) -> torch.Tensor:
        """Compute IoU scores between predictions and assigned GTs.

        This method is isolated for easy removal/replacement if needed.
        Uses ProbIoU for accurate rotated box IoU computation.

        Args:
            pred_boxes: (B, L, 5) - Predicted rotated boxes
            assigned_boxes: (B, L, 5) - Assigned GT rotated boxes
            assigned_labels: (B, L) - Assigned labels for masking
            bg_index: Background class index

        Returns:
            (B, L) - IoU scores, zero for background assignments
        """
        batch_size, num_anchors = pred_boxes.shape[:2]

        # Flatten for batch IoU computation
        pred_flat = pred_boxes.view(-1, 5)
        assigned_flat = assigned_boxes.view(-1, 5)

        # Compute IoU using ProbIoU
        iou_scores = self.iou_calculator(pred_flat, assigned_flat)
        iou_scores = iou_scores.view(batch_size, num_anchors)

        # Set background assignments to zero IoU
        positive_mask = assigned_labels != bg_index
        iou_scores = iou_scores * positive_mask.float()

        # Clamp to valid range
        iou_scores = torch.clamp(iou_scores, 0.0, 1.0)

        return iou_scores
