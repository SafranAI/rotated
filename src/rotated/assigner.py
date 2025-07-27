import torch
import torch.nn as nn

from rotated.iou.prob_iou import ProbIoU


class RotatedTaskAlignedAssigner(nn.Module):
    """Task-Aligned Assigner for Rotated Object Detection.

    Assigns ground truth objects to predicted anchors using task-aligned metrics
    for rotated bounding boxes. This assigner implements the task-aligned
    assignment strategy where the assignment quality is determined by both
    classification score and localization quality (IoU).

    Assignment Strategy:
    1. Spatial Filtering: Only anchors inside GT rotated boxes are candidates
    2. Quality Scoring: Combines classification and IoU scores with weights
    3. Top-K Selection: Selects top-k candidates per GT object
    4. Conflict Resolution: Resolves multi-assignment conflicts using IoU
    5. Score Normalization: Normalizes assigned scores for loss weighting

    The alignment score is computed as: cls_score^alpha * IoU^beta
    where alpha controls classification importance and beta controls
    localization importance in the assignment decision.

    Input Format:
    - pred_scores: (B, L, C) - Classification scores (post-sigmoid)
    - pred_bboxes: (B, L, 5) - Decoded rotated boxes [cx, cy, w, h, angle]
    - anchor_points: (1, L, 2) - Anchor point coordinates [x, y]
    - gt_labels: (B, N, 1) - Class labels as integers [0, num_classes-1]
    - gt_bboxes: (B, N, 5) - GT rotated boxes [cx, cy, w, h, angle]
    - pad_gt_mask: (B, N, 1) - Valid GT mask (1.0=valid, 0.0=padding)

    Output Format:
    - assigned_labels: (B, L) - Assigned class labels per anchor
    - assigned_bboxes: (B, L, 5) - Assigned GT boxes per anchor
    - assigned_scores: (B, L, C) - Normalized quality scores per anchor

    All inputs use absolute pixel coordinates. Rotated boxes follow the
    format [cx, cy, w, h, angle] where angle is in radians [0, π/2).

    Args:
        topk: Number of top candidates to select per GT object. Higher values
            increase positive samples but may include lower-quality assignments.
            Default: 13
        alpha: Exponent for classification score in alignment computation.
            Higher values prioritize classification quality. Default: 1.0
        beta: Exponent for IoU score in alignment computation. Higher values
            prioritize localization quality. Default: 6.0
        eps: Small value to prevent division by zero in normalization.
            Default: 1e-9

    Note:
        This assigner is designed for rotated object detection and uses
        rotated IoU computation for spatial relationships. The assignment
        quality considers both classification confidence and localization
        accuracy, leading to better training stability.

        All input tensors must be on the same device. The assigner expects
        valid coordinate ranges and properly formatted rotated boxes.
    """

    def __init__(self, topk: int = 13, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9):
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.iou_calculator = ProbIoU()

    @torch.no_grad()
    def forward(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        pad_gt_mask: torch.Tensor,
        bg_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform task-aligned assignment for rotated object detection.

        Assigns ground truth objects to anchor points based on task-aligned
        metrics combining classification and localization quality. The assignment
        process includes spatial filtering, quality scoring, top-k selection,
        and conflict resolution.

        Args:
            pred_scores: (B, L, C) - Predicted classification scores after sigmoid
            pred_bboxes: (B, L, 5) - Predicted rotated boxes [cx, cy, w, h, angle]
            anchor_points: (1, L, 2) - Anchor point coordinates in absolute pixels
            gt_labels: (B, N, 1) - Ground truth class labels [0, num_classes-1]
            gt_bboxes: (B, N, 5) - Ground truth rotated boxes [cx, cy, w, h, angle]
            pad_gt_mask: (B, N, 1) - Valid ground truth mask (1.0=valid, 0.0=pad)
            bg_index: Background class index (typically num_classes)

        Returns:
            Tuple containing:
            - assigned_labels: (B, L) - Class labels for each anchor
            - assigned_bboxes: (B, L, 5) - Assigned GT boxes per anchor
            - assigned_scores: (B, L, C) - Normalized quality scores

        Shape:
            B: Batch size
            L: Number of anchor points (total across all FPN levels)
            N: Maximum number of GT objects per image
            C: Number of classes
        """
        batch_size, num_anchors, num_classes = pred_scores.shape
        num_gts = gt_bboxes.shape[1]

        # Handle empty GT case
        if num_gts == 0:
            assigned_labels = torch.full(
                (batch_size, num_anchors), bg_index, dtype=torch.long, device=pred_scores.device
            )
            assigned_bboxes = torch.zeros(
                (batch_size, num_anchors, 5), dtype=pred_bboxes.dtype, device=pred_scores.device
            )
            assigned_scores = torch.zeros(
                (batch_size, num_anchors, num_classes), dtype=pred_scores.dtype, device=pred_scores.device
            )
            return assigned_labels, assigned_bboxes, assigned_scores

        # Compute IoU matrix between all GT and predicted boxes
        iou_matrix = self._compute_iou_matrix(gt_bboxes, pred_bboxes)

        # Check spatial containment of anchors in GT boxes
        spatial_mask = self._point_in_rotated_boxes(anchor_points, gt_bboxes)

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
            gt_bboxes,
            alignment_scores,
            iou_matrix,
            batch_size,
            num_anchors,
            num_classes,
            bg_index,
        )

    def _compute_iou_matrix(self, gt_bboxes: torch.Tensor, pred_bboxes: torch.Tensor) -> torch.Tensor:
        """Compute pairwise IoU matrix between GT and predicted rotated boxes.

        Args:
            gt_bboxes: (B, N, 5) - Ground truth rotated boxes
            pred_bboxes: (B, L, 5) - Predicted rotated boxes

        Returns:
            (B, N, L) - IoU matrix, clamped to [0, 1]
        """
        batch_size, num_gts, _ = gt_bboxes.shape
        num_preds = pred_bboxes.shape[1]

        # Expand for pairwise computation
        gt_expanded = gt_bboxes.unsqueeze(2).expand(-1, -1, num_preds, -1)
        pred_expanded = pred_bboxes.unsqueeze(1).expand(-1, num_gts, -1, -1)

        # Flatten and compute IoU
        gt_flat = gt_expanded.reshape(-1, 5)
        pred_flat = pred_expanded.reshape(-1, 5)
        iou_flat = self.iou_calculator(gt_flat, pred_flat)

        return torch.clamp(iou_flat.view(batch_size, num_gts, num_preds), max=1.0)

    def _point_in_rotated_boxes(self, points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Check containment of anchor points in rotated bounding boxes.

        Args:
            points: (1, L, 2) - Anchor point coordinates [x, y]
            boxes: (B, N, 5) - Rotated boxes [cx, cy, w, h, angle]

        Returns:
            (B, N, L) - Boolean mask where True indicates point is inside box
        """
        batch_size, num_boxes, _ = boxes.shape
        num_points = points.shape[1]

        # Expand tensors for pairwise computation
        points_exp = points.expand(batch_size, -1, -1).unsqueeze(1).expand(-1, num_boxes, -1, -1)
        boxes_exp = boxes.unsqueeze(2).expand(-1, -1, num_points, -1)

        # Extract parameters
        cx, cy, w, h, angle = boxes_exp.unbind(-1)
        px, py = points_exp.unbind(-1)

        # Transform to local coordinates
        dx, dy = px - cx, py - cy
        cos_a, sin_a = torch.cos(-angle), torch.sin(-angle)  # Negative angle for inverse transform
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a

        # Check bounds in local coordinate system
        return (torch.abs(local_x) <= w * 0.5) & (torch.abs(local_y) <= h * 0.5)

    def _gather_class_scores(self, pred_scores: torch.Tensor, gt_labels: torch.Tensor) -> torch.Tensor:
        """Extract classification scores for ground truth classes.

        Args:
            pred_scores: (B, L, C) - Predicted classification scores
            gt_labels: (B, N, 1) - Ground truth class labels

        Returns:
            (B, N, L) - Classification scores for GT classes
        """
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
        """Select top-k scoring anchors for each ground truth object.

        Args:
            scores: (B, N, L) - Alignment scores (cls^alpha * IoU^beta)
            valid_mask: (B, N, 1) - Valid GT object mask

        Returns:
            (B, N, L) - Binary mask indicating top-k selections
        """
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

        # Apply validity mask to filter out invalid GT objects
        valid_expanded = valid_mask.squeeze(-1).bool().unsqueeze(-1)
        return (topk_mask & valid_expanded).float()

    def _resolve_conflicts(self, mask: torch.Tensor, ious: torch.Tensor) -> torch.Tensor:
        """Resolve assignment conflicts when anchors match multiple GT objects.

        Args:
            mask: (B, N, L) - Current assignment mask
            ious: (B, N, L) - IoU matrix for conflict resolution

        Returns:
            (B, N, L) - Resolved assignment mask with conflicts removed
        """
        # Identify anchors with conflicts (assigned to multiple GTs)
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
        gt_bboxes: torch.Tensor,
        alignment_scores: torch.Tensor,
        ious: torch.Tensor,
        batch_size: int,
        num_anchors: int,
        num_classes: int,
        bg_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create final assignment tensors from assignment mask.

        Args:
            mask: (B, N, L) - Final assignment mask after conflict resolution
            gt_labels: (B, N, 1) - Ground truth class labels
            gt_bboxes: (B, N, 5) - Ground truth rotated boxes
            alignment_scores: (B, N, L) - Task-aligned scores
            ious: (B, N, L) - IoU matrix for normalization
            batch_size: Batch size
            num_anchors: Number of anchor points
            num_classes: Number of object classes
            bg_index: Background class index

        Returns:
            Tuple containing assigned labels, bboxes, and normalized scores
        """
        # Determine which anchors have assignments
        has_assignment = mask.sum(dim=1) > 0  # [B, L]
        assigned_gt_idx = mask.argmax(dim=1)  # [B, L] - which GT assigned to each anchor

        # Assign class labels
        batch_idx = torch.arange(batch_size, device=gt_labels.device)[:, None]
        assigned_labels = gt_labels[batch_idx, assigned_gt_idx].squeeze(-1)
        assigned_labels = torch.where(has_assignment, assigned_labels, torch.full_like(assigned_labels, bg_index))

        # Assign bounding boxes
        assigned_bboxes = gt_bboxes[batch_idx, assigned_gt_idx]

        # Create normalized quality scores for loss weighting
        assigned_scores = torch.zeros(batch_size, num_anchors, num_classes, device=mask.device)

        if has_assignment.any():
            # Normalize alignment scores by maximum values per GT
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

        return assigned_labels, assigned_bboxes, assigned_scores


# Simplified testing
if __name__ == "__main__":

    def test_basic_functionality():
        """Test basic functionality of RotatedTaskAlignedAssigner."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Testing on {device}")

        # Create assigner
        assigner = RotatedTaskAlignedAssigner(topk=9, alpha=1.0, beta=6.0).to(device)

        # Test data
        B, L, C, N = 2, 50, 10, 3
        pred_scores = torch.rand(B, L, C, device=device)
        pred_bboxes = torch.rand(B, L, 5, device=device) * 100
        anchor_points = torch.rand(1, L, 2, device=device) * 100
        gt_labels = torch.randint(0, C, (B, N, 1), device=device)
        gt_bboxes = torch.rand(B, N, 5, device=device) * 100
        pad_gt_mask = torch.ones(B, N, 1, device=device)
        bg_index = C

        # Test normal case
        with torch.no_grad():
            labels, bboxes, scores = assigner(
                pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, pad_gt_mask, bg_index
            )

        print(f"✓ Output shapes: labels{labels.shape}, bboxes{bboxes.shape}, scores{scores.shape}")

        positive_count = (labels < bg_index).sum().item()
        print(f"✓ Positive assignments: {positive_count}/{B * L}")

        # Test empty GT
        empty_labels, _, empty_scores = assigner(
            pred_scores,
            pred_bboxes,
            anchor_points,
            torch.zeros(B, 0, 1, dtype=torch.long, device=device),
            torch.zeros(B, 0, 5, device=device),
            torch.zeros(B, 0, 1, device=device),
            bg_index,
        )

        assert (empty_labels == bg_index).all(), "Empty GT should assign all to background"
        assert empty_scores.sum() == 0, "Empty GT should have zero scores"
        print("✓ Empty GT test passed")

        # Test with invalid GT
        partial_mask = torch.ones(B, N, 1, device=device)
        partial_mask[:, -1:] = 0  # Make last GT invalid

        partial_labels, _, _ = assigner(
            pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, partial_mask, bg_index
        )

        partial_positive = (partial_labels < bg_index).sum().item()
        print(f"✓ Partial GT test: {partial_positive} positive assignments")

        print("All tests passed!")

    test_basic_functionality()
