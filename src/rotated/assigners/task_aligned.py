import torch

from rotated.assigners.base import BaseTaskAlignedAssigner
from rotated.assigners.calculators import HorizontalIoUCalculator


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
            box_format='horizontal',
            topk=topk,
            alpha=alpha,
            beta=beta,
            eps=eps,
        )


if __name__ == "__main__":
    print("Testing TaskAlignedAssigner")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create horizontal assigner
    assigner = TaskAlignedAssigner(topk=9, alpha=1.0, beta=6.0).to(device)

    # Test data
    B, L, C, N = 2, 100, 15, 5
    pred_scores = torch.rand(B, L, C, device=device)
    pred_boxes = torch.rand(B, L, 4, device=device) * 100  # [cx, cy, w, h]
    anchor_points = torch.rand(1, L, 2, device=device) * 100
    gt_labels = torch.randint(0, C, (B, N, 1), device=device)
    gt_boxes = torch.rand(B, N, 4, device=device) * 100  # [cx, cy, w, h]
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
        torch.zeros(B, 0, 4, device=device),
        torch.zeros(B, 0, 1, device=device),
        bg_index,
    )

    assert (empty_labels == bg_index).all(), "Empty GT should assign all to background"
    assert empty_scores.sum() == 0, "Empty GT should have zero scores"
    print("Empty GT test passed")

    # Test with partial GT mask
    partial_mask = torch.ones(B, N, 1, device=device)
    partial_mask[:, -1:] = 0  # Make last GT invalid

    partial_labels, _, partial_scores = assigner(
        pred_scores, pred_boxes, anchor_points, gt_labels, gt_boxes, partial_mask, bg_index
    )

    partial_positive = (partial_labels < bg_index).sum().item()
    print(f"Partial GT test: {partial_positive} positive assignments")

    # Verify score properties
    max_score_per_class = scores.max(dim=1)[0]  # [B, C]
    print(
        f"Max score per class range: [{max_score_per_class.min().item():.3f}, {max_score_per_class.max().item():.3f}]"
    )

    # Test class distribution
    unique_labels = torch.unique(labels[labels < bg_index])
    print(f"Assigned classes: {len(unique_labels)} unique classes out of {C}")

    print("All horizontal assigner tests passed!")
