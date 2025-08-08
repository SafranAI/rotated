import torch

from rotated.assigners.task_aligned import TaskAlignedAssigner


def test_task_aligned_assigner_forward():
    """Test TaskAlignedAssigner forward pass."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assigner = TaskAlignedAssigner(topk=9, alpha=1.0, beta=6.0).to(device)

    # Test data
    batch_size, num_anchors, num_classes, num_gts = 2, 100, 15, 5
    pred_scores = torch.rand(batch_size, num_anchors, num_classes, device=device)
    pred_boxes = torch.rand(batch_size, num_anchors, 4, device=device) * 100  # [cx, cy, w, h]
    anchor_points = torch.rand(1, num_anchors, 2, device=device) * 100
    gt_labels = torch.randint(0, num_classes, (batch_size, num_gts, 1), device=device)
    gt_boxes = torch.rand(batch_size, num_gts, 4, device=device) * 100  # [cx, cy, w, h]
    pad_gt_mask = torch.ones(batch_size, num_gts, 1, device=device)
    bg_index = num_classes

    with torch.no_grad():
        assigned_labels, assigned_boxes, assigned_scores = assigner(
            pred_scores, pred_boxes, anchor_points, gt_labels, gt_boxes, pad_gt_mask, bg_index
        )

    # Verify output shapes
    assert assigned_labels.shape == (batch_size, num_anchors)
    assert assigned_boxes.shape == (batch_size, num_anchors, 4)
    assert assigned_scores.shape == (batch_size, num_anchors, num_classes)

    # Verify some positive assignments exist
    positive_count = (assigned_labels < bg_index).sum().item()
    assert positive_count > 0
