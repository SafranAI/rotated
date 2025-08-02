"""Base task-aligned assigner with generic assignment logic."""

import torch
import torch.nn as nn


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
        spatial_checker: Object that checks point-in-box containment
        topk: Number of top candidates to select per GT object
        alpha: Exponent for classification score in alignment computation
        beta: Exponent for IoU score in alignment computation
        eps: Small value to prevent division by zero
    """

    def __init__(
        self, iou_calculator, spatial_checker, topk: int = 13, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9
    ):
        super().__init__()
        self.iou_calculator = iou_calculator
        self.spatial_checker = spatial_checker
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

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
            pred_boxes: [B, L, box_dim] - Predicted boxes (format depends on calculator)
            anchor_points: [1, L, 2] - Anchor point coordinates
            gt_labels: [B, N, 1] - Ground truth class labels
            gt_boxes: [B, N, box_dim] - Ground truth boxes (format depends on calculator)
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

        # Compute IoU matrix between all GT and predicted boxes
        iou_matrix = self._compute_iou_matrix(gt_boxes, pred_boxes)

        # Check spatial containment of anchors in GT boxes
        spatial_mask = self._compute_spatial_mask(anchor_points, gt_boxes)

        # Extract classification scores for GT classes
        class_scores = self._gather_class_scores(pred_scores, gt_labels)

        # Compute alignment metrics
        alignment_scores = class_scores.pow(self.alpha) * iou_matrix.pow(self.beta)

        # Select top-k candidates
        topk_mask = self._select_topk(alignment_scores * spatial_mask.float(), pad_gt_mask)

        # Create positive mask
        positive_mask = topk_mask * spatial_mask.float() * pad_gt_mask.float()

        # Resolve conflicts
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
        """Compute pairwise IoU matrix using the provided calculator."""
        batch_size, num_gts, box_dim = gt_boxes.shape
        num_preds = pred_boxes.shape[1]

        # Expand for pairwise computation
        gt_expanded = gt_boxes.unsqueeze(2).expand(-1, -1, num_preds, -1)
        pred_expanded = pred_boxes.unsqueeze(1).expand(-1, num_gts, -1, -1)

        # Flatten and compute IoU
        gt_flat = gt_expanded.reshape(-1, box_dim)
        pred_flat = pred_expanded.reshape(-1, box_dim)
        iou_flat = self.iou_calculator(gt_flat, pred_flat)

        return torch.clamp(iou_flat.view(batch_size, num_gts, num_preds), max=1.0)

    def _compute_spatial_mask(self, anchor_points: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        """Compute spatial containment mask using the provided checker."""
        batch_size, *_ = gt_boxes.shape

        # Process each batch item
        spatial_masks = []
        for b in range(batch_size):
            mask = self.spatial_checker(anchor_points.squeeze(0), gt_boxes[b])  # [N, L]
            spatial_masks.append(mask)

        return torch.stack(spatial_masks, dim=0)  # [B, N, L]

    def _gather_class_scores(self, pred_scores: torch.Tensor, gt_labels: torch.Tensor) -> torch.Tensor:
        """Extract classification scores for ground truth classes."""
        batch_size, num_preds, _ = pred_scores.shape
        num_gts = gt_labels.shape[1]

        # Create index tensors for advanced indexing
        batch_idx = torch.arange(batch_size, device=pred_scores.device)[:, None, None]
        pred_idx = torch.arange(num_preds, device=pred_scores.device)[None, None, :]
        class_idx = gt_labels.squeeze(-1)[:, :, None]

        # Expand indices to match output shape
        batch_idx = batch_idx.expand(batch_size, num_gts, num_preds)
        pred_idx = pred_idx.expand(batch_size, num_gts, num_preds)
        class_idx = class_idx.expand(batch_size, num_gts, num_preds)

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
            # Normalize alignment scores
            max_alignment_per_gt = (alignment_scores * mask).amax(dim=-1, keepdim=True)  # [B, N, 1]
            max_iou_per_gt = (ious * mask).amax(dim=-1, keepdim=True)  # [B, N, 1]

            # Compute normalization factor
            norm_factor = (alignment_scores * mask * max_iou_per_gt / (max_alignment_per_gt + self.eps)).amax(dim=1)

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
        # Use dynamic box dimension - will be reshaped by actual usage
        assigned_boxes = torch.zeros((batch_size, num_anchors, 5), device=device)
        assigned_scores = torch.zeros((batch_size, num_anchors, num_classes), device=device)
        return assigned_labels, assigned_boxes, assigned_scores


if __name__ == "__main__":
    print("Testing BaseTaskAlignedAssigner")

    from rotated.assigner.calculators import RotatedIoUCalculator
    from rotated.assigner.spatial import RotatedSpatialChecker

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create base assigner with rotated components
    assigner = BaseTaskAlignedAssigner(
        iou_calculator=RotatedIoUCalculator(), spatial_checker=RotatedSpatialChecker(), topk=9, alpha=1.0, beta=6.0
    ).to(device)

    # Test data
    B, L, C, N = 2, 50, 10, 3
    pred_scores = torch.rand(B, L, C, device=device)
    pred_boxes = torch.rand(B, L, 5, device=device) * 100
    anchor_points = torch.rand(1, L, 2, device=device) * 100
    gt_labels = torch.randint(0, C, (B, N, 1), device=device)
    gt_boxes = torch.rand(B, N, 5, device=device) * 100
    pad_gt_mask = torch.ones(B, N, 1, device=device)
    bg_index = C

    # Test normal case
    with torch.no_grad():
        labels, boxes, scores = assigner(
            pred_scores, pred_boxes, anchor_points, gt_labels, gt_boxes, pad_gt_mask, bg_index
        )

    print(f"Output shapes: labels{labels.shape}, boxes{boxes.shape}, scores{scores.shape}")

    positive_count = (labels < bg_index).sum().item()
    print(f"Positive assignments: {positive_count}/{B * L}")

    print("Base assigner test passed!")
