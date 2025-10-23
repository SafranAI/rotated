import torch

from rotated.assigners.calculators import HorizontalIoUCalculator, RotatedIoUCalculator


def test_rotated_iou_calculator():
    """Test rotated IoU calculator basic functionality."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calc = RotatedIoUCalculator()

    # Test basic pairwise computation
    boxes1 = torch.randn(2, 5, device=device)
    boxes2 = torch.randn(3, 5, device=device)
    iou_matrix = calc(boxes1, boxes2)
    assert iou_matrix.shape == (2, 3)

    # Test empty cases
    empty_boxes = torch.empty(0, 5, device=device)
    empty_iou = calc(empty_boxes, boxes2)
    assert empty_iou.shape == (0, 3)

    empty_iou2 = calc(boxes1, empty_boxes)
    assert empty_iou2.shape == (2, 0)

def test_horizontal_iou_calculator():
    """Test horizontal IoU calculator basic functionality."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calc = HorizontalIoUCalculator()

    # Test basic pairwise computation
    boxes1 = torch.randn(3, 4, device=device)
    boxes2 = torch.randn(2, 4, device=device)
    iou_matrix = calc(boxes1, boxes2)
    assert iou_matrix.shape == (3, 2)

    # Test empty cases
    empty_boxes = torch.empty(0, 4, device=device)
    empty_iou = calc(empty_boxes, boxes2)
    assert empty_iou.shape == (0, 2)

    empty_iou2 = calc(boxes1, empty_boxes)
    assert empty_iou2.shape == (3, 0)
