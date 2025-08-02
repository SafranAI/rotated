"""Rotated task-aligned assigner implementation."""

import torch

from rotated.assigners.base import BaseTaskAlignedAssigner
from rotated.assigners.calculators import RotatedIoUCalculator
from rotated.assigners.spatial import RotatedSpatialChecker


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
        iou_eps: Small constant for numerical stability in IoU computation
    """

    def __init__(
        self,
        topk: int = 13,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
        iou_eps: float = 1e-3,
    ):
        super().__init__(
            iou_calculator=RotatedIoUCalculator(eps=iou_eps),
            spatial_checker=RotatedSpatialChecker(),
            topk=topk,
            alpha=alpha,
            beta=beta,
            eps=eps,
        )


if __name__ == "__main__":
    print("Testing RotatedTaskAlignedAssigner")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create rotated assigner
    assigner = RotatedTaskAlignedAssigner(topk=9, alpha=1.0, beta=6.0).to(device)

    # Test data
    B, L, C, N = 2, 100, 15, 5
    pred_scores = torch.rand(B, L, C, device=device)
    pred_boxes = torch.rand(B, L, 5, device=device) * 100  # [cx, cy, w, h, angle]
    anchor_points = torch.rand(1, L, 2, device=device) * 100
    gt_labels = torch.randint(0, C, (B, N, 1), device=device)
    gt_boxes = torch.rand(B, N, 5, device=device) * 100  # [cx, cy, w, h, angle]
    pad_gt_mask = torch.ones(B, N, 1, device=device)
    bg_index = C

    # Test normal assignment
    with torch.no_grad():
        labels, boxes, scores = assigner(
            pred_scores, pred_boxes, anchor_points, gt_labels, gt_boxes, pad_gt_mask, bg_index
        )

    print(f"Normal case - Output shapes: labels{labels.shape}, boxes{boxes.shape}, scores{scores.shape}")

    positive_count = (labels < bg_index).sum().item()
    total_anchors = B * L
    print(f"Positive assignments: {positive_count}/{total_anchors} ({100 * positive_count / total_anchors:.1f}%)")

    # Test empty GT case
    empty_labels, empty_boxes, empty_scores = assigner(
        pred_scores,
        pred_boxes,
        anchor_points,
        torch.zeros(B, 0, 1, dtype=torch.long, device=device),
        torch.zeros(B, 0, 5, device=device),
        torch.zeros(B, 0, 1, device=device),
        bg_index,
    )

    assert (empty_labels == bg_index).all(), "Empty GT should assign all to background"
    assert empty_scores.sum() == 0, "Empty GT should have zero scores"
    print("Empty GT test passed")

    # Test with partial GT mask
    partial_mask = torch.ones(B, N, 1, device=device)
    partial_mask[:, -2:] = 0  # Make last 2 GTs invalid

    partial_labels, _, partial_scores = assigner(
        pred_scores, pred_boxes, anchor_points, gt_labels, gt_boxes, partial_mask, bg_index
    )

    partial_positive = (partial_labels < bg_index).sum().item()
    print(f"Partial GT test: {partial_positive} positive assignments (should be less than normal case)")

    # Verify score properties
    max_score_per_class = scores.max(dim=1)[0]  # [B, C]
    print(
        f"Max score per class range: [{max_score_per_class.min().item():.3f}, {max_score_per_class.max().item():.3f}]"
    )

    # Test class distribution
    unique_labels = torch.unique(labels[labels < bg_index])
    print(f"Assigned classes: {len(unique_labels)} unique classes out of {C}")

    # Test IoU quality of assignments
    if positive_count > 0:
        pos_mask = labels < bg_index
        pos_pred_boxes = pred_boxes[pos_mask]
        pos_assigned_boxes = boxes[pos_mask]

        if len(pos_pred_boxes) > 0:
            sample_ious = assigner.iou_calculator(pos_pred_boxes[:10], pos_assigned_boxes[:10])
            mean_iou = torch.diag(sample_ious).mean().item()
            print(f"Sample assignment IoU quality: {mean_iou:.3f}")

    print("All rotated assigner tests passed!")
