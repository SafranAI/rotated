import pytest
import torch

from rotated.boxes.nms import NMS


@pytest.fixture()
def nms():
    return NMS()


def test_rotated_nms_suppresses_overlapping_boxes(nms):
    """Test that overlapping boxes are properly suppressed."""
    # Two highly overlapping boxes + one separate box
    boxes = torch.tensor(
        [
            [100.0, 100.0, 50.0, 30.0, 0.0],  # Box 1
            [102.0, 102.0, 52.0, 32.0, 0.0],  # Box 2 - overlaps with Box 1
            [300.0, 300.0, 40.0, 25.0, 0.0],  # Box 3 - separate
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])  # Box 1 has highest score

    keep = nms.rotated_nms(boxes, scores, iou_threshold=0.3)

    # Should keep exactly 2 boxes (highest scoring overlapping + separate)
    assert len(keep) == 2
    # Should keep box 0 (highest score) and box 2 (separate)
    assert 0 in keep
    assert 2 in keep
    assert 1 not in keep  # Overlapping box with lower score should be suppressed


def test_rotated_nms_preserves_non_overlapping(nms):
    """Test that non-overlapping boxes are all preserved."""
    # Three well-separated boxes
    boxes = torch.tensor(
        [
            [100.0, 100.0, 30.0, 20.0, 0.0],
            [200.0, 200.0, 30.0, 20.0, 0.0],
            [300.0, 300.0, 30.0, 20.0, 0.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])

    keep = nms.rotated_nms(boxes, scores, iou_threshold=0.5)

    # All boxes should be kept since they don't overlap
    assert len(keep) == 3
    assert torch.equal(keep, torch.tensor([0, 1, 2]))  # Sorted by score


def test_multiclass_nms_preserves_different_classes(nms):
    """Test that overlapping boxes from different classes are preserved."""
    # Two overlapping boxes but different classes
    boxes = torch.tensor(
        [
            [100.0, 100.0, 50.0, 30.0, 0.0],
            [102.0, 102.0, 52.0, 32.0, 0.0],  # Overlaps with first
        ]
    )
    scores = torch.tensor([0.9, 0.8])
    labels = torch.tensor([0, 1])  # Different classes

    keep = nms(boxes, scores, labels, iou_threshold=0.5)

    # Both should be kept despite overlap (different classes)
    assert len(keep) == 2
    assert 0 in keep
    assert 1 in keep


def test_multiclass_nms_suppresses_same_class(nms):
    """Test that overlapping boxes from same class are suppressed."""
    # Two overlapping boxes, same class
    boxes = torch.tensor(
        [
            [100.0, 100.0, 50.0, 30.0, 0.0],
            [102.0, 102.0, 52.0, 32.0, 0.0],  # Overlaps with first
        ]
    )
    scores = torch.tensor([0.9, 0.8])
    labels = torch.tensor([0, 0])  # Same class

    keep = nms(boxes, scores, labels, iou_threshold=0.5)

    # Only highest scoring box should be kept
    assert len(keep) == 1
    assert keep[0] == 0  # Higher score


def test_batched_nms_handles_different_scenarios(nms):
    """Test batched NMS with different scenarios per batch."""
    # Batch 1: 3 non-overlapping boxes
    # Batch 2: 2 overlapping boxes (one will be suppressed) + 1 separate
    boxes = torch.tensor(
        [
            # Batch 1: all separate
            [[100.0, 100.0, 30.0, 20.0, 0.0], [200.0, 200.0, 30.0, 20.0, 0.0], [300.0, 300.0, 30.0, 20.0, 0.0]],
            # Batch 2: first two overlap, third separate
            [
                [100.0, 100.0, 50.0, 30.0, 0.0],
                [102.0, 102.0, 52.0, 32.0, 0.0],  # Overlaps with first
                [300.0, 300.0, 30.0, 20.0, 0.0],
            ],
        ]
    )
    scores = torch.tensor(
        [
            [0.9, 0.8, 0.7],  # Batch 1: all different scores
            [0.9, 0.8, 0.6],  # Batch 2: overlapping boxes have different scores
        ]
    )
    labels = torch.tensor(
        [
            [0, 1, 2],  # All different classes
            [0, 0, 1],  # First two same class, third different
        ]
    )

    keep = nms.batched_multiclass_rotated_nms(boxes, scores, labels, 0.5, max_output_per_batch=5)

    # Batch 1: all 3 boxes should be kept (non-overlapping)
    batch1_valid = keep[0][keep[0] >= 0]
    assert len(batch1_valid) == 3

    # Batch 2: 2 boxes should be kept (one overlapping suppressed, one separate kept)
    batch2_valid = keep[1][keep[1] >= 0]
    assert len(batch2_valid) == 2


@pytest.mark.parametrize(
    "argument, match",
    [
        ({"iou_method": "bad_method"}, "Unknown IoU method"),
        ({"iou_kwargs": {"bad_param": 1.0}}, "Parameter 'bad_param' is not supported"),
    ],
)
def test_wrong_iou_params(argument, match):
    """Test that error is raised when bad IoU params are passed to NMS"""
    with pytest.raises(ValueError, match=match):
        NMS(**argument)


def test_correct_nms_params():
    nms = NMS(nms_thresh=0.1, iou_method="precise_rotated_iou", iou_kwargs={"eps": 1e-3})
    assert nms.nms_thresh == 0.1
    assert nms.iou_calculator.__class__.__name__ == "PreciseRotatedIoU"
    assert nms.iou_calculator.eps == 1e-3


def test_torchscript_compatibility(nms):
    """Test TorchScript compilation works correctly."""
    boxes = torch.tensor(
        [
            [100.0, 100.0, 50.0, 30.0, 0.0],
            [200.0, 200.0, 30.0, 20.0, 0.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8])
    labels = torch.tensor([0, 1])

    # Test scripting
    scripted_fn = torch.jit.script(nms)
    scripted_result = scripted_fn(boxes, scores, labels, 0.5)

    # Test eager mode
    eager_result = nms.forward(boxes, scores, labels, 0.5)

    # Results should be identical
    assert torch.allclose(scripted_result, eager_result)
