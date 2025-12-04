# Modified from PaddleDetection (https://github.com/PaddlePaddle/PaddleDetection)
# Copyright (c) 2024 PaddlePaddle Authors. Apache 2.0 License.

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from rotated.assigners.calculators import HorizontalIoUCalculator, RotatedIoUCalculator
from rotated.boxes.utils import check_points_in_horizontal_boxes, check_points_in_rotated_boxes

if TYPE_CHECKING:
    from rotated.iou import IoUKwargs, IoUMethodName

__all__ = ["RotatedTaskAlignedAssigner", "TaskAlignedAssigner"]


class BaseTaskAlignedAssigner(nn.Module):
    """Base class for task-aligned assignment with generic logic.

    The assignment strategy:
    1. Compute IoU matrix between GT and predicted boxes
    2. Check spatial containment of anchors in GT boxes
    3. Compute alignment scores (cls_score^alpha * IoU^beta)
    4. Select top-k candidates per GT object
    5. Resolve assignment conflicts using IoU
    6. Generate final assignments with normalized scores

    Args:
        iou_calculator: Object that computes IoU between box pairs
        box_format: Format of boxes - 'rotated' for 5D boxes or 'horizontal' for 4D boxes
        topk: Number of top candidates to select per GT object
        alpha: Exponent for classification score in alignment computation
        beta: Exponent for IoU score in alignment computation
        eps: Small value to prevent division by zero

    Raises:
        ValueError: If box_format is not 'rotated' or 'horizontal'
    """

    def __init__(
        self,
        iou_calculator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        box_format: str = "rotated",
        topk: int = 13,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.iou_calculator = iou_calculator
        self.box_format = box_format
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        if box_format not in ["rotated", "horizontal"]:
            raise ValueError(f"box_format must be 'rotated' or 'horizontal', got {box_format}")

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform task-aligned assignment.

        Args:
            pred_scores: [B, L, C] - Predicted classification scores after sigmoid
            pred_boxes: [B, L, box_dim] - Predicted boxes (format depends on `box_format`)
            anchor_points: [1, L, 2] - Anchor point coordinates
            gt_labels: [B, N, 1] - Ground truth class labels
            gt_boxes: [B, N, box_dim] - Ground truth boxes (format depends on `box_format`)
            pad_gt_mask: [B, N, 1] - Valid ground truth mask
            bg_index: Background class index

        Returns:
            Tuple containing:
            - assigned_labels: [B, L] - Class labels for each anchor
            - assigned_boxes: [B, L, box_dim] - Assigned GT boxes per anchor
            - assigned_scores: [B, L, C] - Normalized quality scores
        """
        batch_size, num_anchors, num_classes = pred_scores.shape
        num_gts = gt_boxes.shape[1]

        # Handle empty GT case
        if num_gts == 0:
            return self._empty_assignment(batch_size, num_anchors, num_classes, bg_index, pred_scores.device)

        iou_matrix = self._compute_iou_matrix(gt_boxes, pred_boxes)
        spatial_mask = self._compute_spatial_mask(anchor_points, gt_boxes)
        class_scores = self._gather_class_scores(pred_scores, gt_labels)
        alignment_scores = class_scores.pow(self.alpha) * iou_matrix.pow(self.beta)
        topk_mask = self._select_topk(alignment_scores * spatial_mask.float(), pad_gt_mask)
        positive_mask = topk_mask * spatial_mask.float() * pad_gt_mask.float()
        positive_mask = self._resolve_conflicts(positive_mask, iou_matrix)

        # Generate assignments
        return self._create_assignments(
            positive_mask,
            gt_labels,
            gt_boxes,
            alignment_scores,
            iou_matrix,
            batch_size,
            num_anchors,
            num_classes,
            bg_index,
        )

    def _compute_iou_matrix(self, gt_boxes: torch.Tensor, pred_boxes: torch.Tensor) -> torch.Tensor:
        """Compute pairwise IoU matrix using the provided calculator.

        Args:
            gt_boxes: [B, N, box_dim] - Ground truth boxes
            pred_boxes: [B, L, box_dim] - Predicted boxes

        Returns:
            iou_matrix: [B, N, L] - IoU between each GT and predicted box pair
        """
        batch_size = gt_boxes.shape[0]

        # Process each batch separately to compute IoU matrices
        iou_matrices = []

        for batch_idx in range(batch_size):
            gt_batch = gt_boxes[batch_idx]  # [N, box_dim]
            pred_batch = pred_boxes[batch_idx]  # [L, box_dim]

            # Compute IoU matrix for this batch: [N, L]
            iou_matrix_batch = self.iou_calculator(gt_batch, pred_batch)
            iou_matrices.append(iou_matrix_batch)

        # Stack all batch IoU matrices: [B, N, L]
        iou_matrix = torch.stack(iou_matrices, dim=0)

        # Set IoU values > 1 + eps to 0 (treat as numerical errors), don't just clamp
        iou_matrix = torch.where(iou_matrix > 1.0 + self.eps, torch.zeros_like(iou_matrix), iou_matrix)

        return iou_matrix

    def _compute_spatial_mask(self, anchor_points: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        """Compute spatial containment mask using box format."""
        batch_size, *_ = gt_boxes.shape

        # Ensure anchor_points has the expected shape
        assert anchor_points.shape[0] == 1, f"Expected anchor_points shape [1, L, 2], got {anchor_points.shape}"

        # Choose spatial checking function based on box format
        if self.box_format == "rotated":
            spatial_check_fn = check_points_in_rotated_boxes
        else:  # horizontal
            spatial_check_fn = check_points_in_horizontal_boxes

        # Process each batch item
        spatial_masks = []
        for b in range(batch_size):
            mask = spatial_check_fn(anchor_points.squeeze(0), gt_boxes[b])  # [N, L]
            spatial_masks.append(mask)

        return torch.stack(spatial_masks, dim=0)  # [B, N, L]

    def _gather_class_scores(self, pred_scores: torch.Tensor, gt_labels: torch.Tensor) -> torch.Tensor:
        """Extract classification scores for ground truth classes.

        Args:
            pred_scores: [B, L, C] - Predicted class scores
            gt_labels: [B, N, 1] - Ground truth class labels

        Returns:
            class_scores: [B, N, L] - Classification scores for GT classes at each anchor
        """
        batch_size, num_preds, num_classes = pred_scores.shape
        num_gts = gt_labels.shape[1]

        # Clamp class indices to valid range to prevent index errors
        gt_labels_clamped = gt_labels.squeeze(-1).clamp(0, num_classes - 1)

        # Create index tensors for advanced indexing
        batch_idx = torch.arange(batch_size, device=pred_scores.device)[:, None, None]
        pred_idx = torch.arange(num_preds, device=pred_scores.device)[None, None, :]
        class_idx = gt_labels_clamped[:, :, None]

        # Expand indices to match output shape [B, N, L]
        batch_idx = batch_idx.expand(batch_size, num_gts, num_preds)
        pred_idx = pred_idx.expand(batch_size, num_gts, num_preds)
        class_idx = class_idx.expand(batch_size, num_gts, num_preds)

        # Index: pred_scores[batch, pred, class] -> [B, N, L]
        return pred_scores[batch_idx, pred_idx, class_idx]

    def _select_topk(self, scores: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Select top-k scoring anchors for each ground truth object."""
        batch_size, num_gts, num_anchors = scores.shape
        actual_topk = min(self.topk, num_anchors)

        # Get top-k indices for each GT object
        _, topk_indices = torch.topk(scores, k=actual_topk, dim=-1, largest=True, sorted=False)

        # Create binary mask for top-k selections
        topk_mask = torch.zeros_like(scores, dtype=torch.bool)

        # Index tensors for scatter operation
        batch_idx = torch.arange(batch_size, device=scores.device)[:, None, None]
        gt_idx = torch.arange(num_gts, device=scores.device)[None, :, None]

        batch_idx = batch_idx.expand(-1, num_gts, actual_topk)
        gt_idx = gt_idx.expand(batch_size, -1, actual_topk)

        # Set top-k positions to True
        topk_mask[batch_idx, gt_idx, topk_indices] = True

        # Apply validity mask
        valid_expanded = valid_mask.squeeze(-1).bool().unsqueeze(-1)
        return (topk_mask & valid_expanded).float()

    def _resolve_conflicts(self, mask: torch.Tensor, ious: torch.Tensor) -> torch.Tensor:
        """Resolve assignment conflicts when anchors match multiple GT objects."""
        # Identify anchors with conflicts
        anchor_counts = mask.sum(dim=1)  # [B, L]
        has_conflict = anchor_counts > 1

        if not has_conflict.any():
            return mask

        # Find GT with highest IoU for each anchor
        best_gt_indices = ious.argmax(dim=1)  # [B, L]

        # Create one-hot mask for best GT assignment
        batch_size, num_gts, num_anchors = mask.shape
        best_mask = torch.zeros_like(mask, dtype=torch.bool)

        batch_idx = torch.arange(batch_size, device=mask.device)[:, None]
        anchor_idx = torch.arange(num_anchors, device=mask.device)[None, :]

        best_mask[batch_idx, best_gt_indices, anchor_idx] = True

        # Apply conflict resolution only where conflicts exist
        conflict_expanded = has_conflict.unsqueeze(1)  # [B, 1, L]
        return torch.where(conflict_expanded, mask * best_mask.float(), mask)

    def _create_assignments(
        self,
        mask: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
        alignment_scores: torch.Tensor,
        ious: torch.Tensor,
        batch_size: int,
        num_anchors: int,
        num_classes: int,
        bg_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create final assignment tensors from assignment mask."""
        # Determine which anchors have assignments
        has_assignment = mask.sum(dim=1) > 0  # [B, L]
        assigned_gt_idx = mask.argmax(dim=1)  # [B, L]

        # Assign class labels
        batch_idx = torch.arange(batch_size, device=gt_labels.device)[:, None]
        assigned_labels = gt_labels[batch_idx, assigned_gt_idx].squeeze(-1)
        assigned_labels = torch.where(has_assignment, assigned_labels, torch.full_like(assigned_labels, bg_index))

        # Assign bounding boxes
        assigned_boxes = gt_boxes[batch_idx, assigned_gt_idx]

        # Create normalized quality scores
        assigned_scores = torch.zeros(batch_size, num_anchors, num_classes, device=mask.device)

        if has_assignment.any():
            # Step 1: Mask alignment scores with positive assignments
            masked_alignment = alignment_scores * mask  # [B, N, L]

            # Step 2: Get max alignment and max IoU per GT
            max_alignment_per_gt = masked_alignment.amax(dim=-1, keepdim=True)  # [B, N, 1]
            max_iou_per_gt = (ious * mask).amax(dim=-1, keepdim=True)  # [B, N, 1]

            # Step 3: Normalize alignment by max per GT, then scale by max IoU
            normalized_alignment = masked_alignment / (max_alignment_per_gt + self.eps) * max_iou_per_gt  # [B, N, L]

            # Step 4: Take max across GTs (dim=1) to get per-anchor normalization factor
            norm_factor = normalized_alignment.amax(dim=1)  # [B, L]

            # Assign scores to positive samples
            pos_indices = has_assignment.nonzero(as_tuple=False)
            if pos_indices.numel() > 0:
                batch_indices = pos_indices[:, 0]
                anchor_indices = pos_indices[:, 1]
                class_indices = assigned_labels[batch_indices, anchor_indices]

                assigned_scores[batch_indices, anchor_indices, class_indices] = norm_factor[
                    batch_indices, anchor_indices
                ]

        return assigned_labels, assigned_boxes, assigned_scores

    def _empty_assignment(
        self, batch_size: int, num_anchors: int, num_classes: int, bg_index: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create empty assignments when no GT objects are present."""
        assigned_labels = torch.full((batch_size, num_anchors), bg_index, dtype=torch.long, device=device)
        # Use dynamic box dimension based on format
        box_dim = 5 if self.box_format == "rotated" else 4
        assigned_boxes = torch.zeros((batch_size, num_anchors, box_dim), device=device)
        assigned_scores = torch.zeros((batch_size, num_anchors, num_classes), device=device)
        return assigned_labels, assigned_boxes, assigned_scores


class RotatedTaskAlignedAssigner(BaseTaskAlignedAssigner):
    """Task-aligned assigner specifically for rotated bounding boxes.

    Expects rotated boxes in format [cx, cy, w, h, angle] where:
    - cx, cy: Center coordinates in absolute pixels
    - w, h: Width and height in absolute pixels
    - angle: Rotation angle in radians

    Args:
        topk: Number of top candidates to select per GT object
        alpha: Exponent for classification score in alignment computation
        beta: Exponent for IoU score in alignment computation
        eps: Small value to prevent division by zero in assignment
        iou_method: Method name to compute Intersection Over Union
        iou_kwargs: Dictionary with parameters for the IoU method
    """

    def __init__(
        self,
        topk: int = 13,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
        iou_method: "IoUMethodName" = "prob_iou",
        iou_kwargs: "IoUKwargs" = None,
    ):
        super().__init__(
            iou_calculator=RotatedIoUCalculator(iou_method=iou_method, iou_kwargs=iou_kwargs),
            box_format="rotated",
            topk=topk,
            alpha=alpha,
            beta=beta,
            eps=eps,
        )


class TaskAlignedAssigner(BaseTaskAlignedAssigner):
    """Task-aligned assigner specifically for horizontal bounding boxes.

    Expects horizontal boxes in format [cx, cy, w, h] where:
    - cx, cy: Center coordinates in absolute pixels
    - w, h: Width and height in absolute pixels

    Args:
        topk: Number of top candidates to select per GT object
        alpha: Exponent for classification score in alignment computation
        beta: Exponent for IoU score in alignment computation
        eps: Small value to prevent division by zero in assignment
        iou_eps: Small constant for numerical stability in IoU computation
    """

    def __init__(
        self,
        topk: int = 13,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
        iou_eps: float = 1e-7,
    ):
        super().__init__(
            iou_calculator=HorizontalIoUCalculator(eps=iou_eps),
            box_format="horizontal",
            topk=topk,
            alpha=alpha,
            beta=beta,
            eps=eps,
        )
