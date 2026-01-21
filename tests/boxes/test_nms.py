"""Comprehensive NMS tests including TorchScript compatibility."""

from typing import TypeAlias

import pytest
import torch

from rotated.boxes.nms import NMS

BoxesScores: TypeAlias = tuple[torch.Tensor, torch.Tensor]
SampleData: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@pytest.fixture
def nms() -> NMS:
    return NMS()


@pytest.fixture
def overlapping_boxes() -> BoxesScores:
    """Two overlapping boxes + one separate box."""
    boxes = torch.tensor(
        [
            [100.0, 100.0, 50.0, 30.0, 0.0],  # Box 1
            [102.0, 102.0, 52.0, 32.0, 0.0],  # Box 2 - overlaps with Box 1
            [300.0, 300.0, 40.0, 25.0, 0.0],  # Box 3 - separate
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    return boxes, scores


@pytest.fixture
def non_overlapping_boxes() -> BoxesScores:
    """Three well-separated boxes."""
    boxes = torch.tensor(
        [
            [100.0, 100.0, 30.0, 20.0, 0.0],
            [200.0, 200.0, 30.0, 20.0, 0.0],
            [300.0, 300.0, 30.0, 20.0, 0.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    return boxes, scores


@pytest.fixture
def two_overlapping_boxes() -> BoxesScores:
    """Two overlapping boxes only."""
    boxes = torch.tensor(
        [
            [100.0, 100.0, 50.0, 30.0, 0.0],
            [102.0, 102.0, 52.0, 32.0, 0.0],  # Overlaps with first
        ]
    )
    scores = torch.tensor([0.9, 0.8])
    return boxes, scores


@pytest.fixture
def sample_data() -> SampleData:
    """Sample detection data for multi-class NMS."""
    boxes = torch.tensor(
        [
            [100.0, 100.0, 50.0, 30.0, 0.0],
            [105.0, 105.0, 50.0, 30.0, 0.0],  # Overlaps with first
            [200.0, 200.0, 30.0, 20.0, 0.0],
            [150.0, 150.0, 40.0, 25.0, 0.5],
        ]
    )
    scores = torch.tensor([0.9, 0.85, 0.8, 0.7])
    labels = torch.tensor([0, 0, 1, 0])
    return boxes, scores, labels


def test_rotated_nms_suppresses_overlapping_boxes(nms: NMS, overlapping_boxes: BoxesScores):
    """Test that overlapping boxes are properly suppressed."""
    boxes, scores = overlapping_boxes

    keep = nms.rotated_nms(boxes, scores, iou_threshold=0.3)

    # Should keep exactly 2 boxes (highest scoring overlapping + separate)
    assert len(keep) == 2
    # Should keep box 0 (highest score) and box 2 (separate)
    assert 0 in keep
    assert 2 in keep
    assert 1 not in keep  # Overlapping box with lower score should be suppressed


def test_rotated_nms_preserves_non_overlapping(nms: NMS, non_overlapping_boxes: BoxesScores):
    """Test that non-overlapping boxes are all preserved."""
    boxes, scores = non_overlapping_boxes

    keep = nms.rotated_nms(boxes, scores, iou_threshold=0.5)

    # All boxes should be kept since they don't overlap
    assert len(keep) == 3
    assert torch.equal(keep, torch.tensor([0, 1, 2]))  # Sorted by score


def test_multiclass_nms_preserves_different_classes(nms: NMS, two_overlapping_boxes: BoxesScores):
    """Test that overlapping boxes from different classes are preserved."""
    boxes, scores = two_overlapping_boxes
    labels = torch.tensor([0, 1])  # Different classes

    keep = nms(boxes, scores, labels, iou_threshold=0.5)

    # Both should be kept despite overlap (different classes)
    assert len(keep) == 2
    assert 0 in keep
    assert 1 in keep


def test_multiclass_nms_suppresses_same_class(nms: NMS, two_overlapping_boxes: BoxesScores):
    """Test that overlapping boxes from same class are suppressed."""
    boxes, scores = two_overlapping_boxes
    labels = torch.tensor([0, 0])  # Same class

    keep = nms(boxes, scores, labels, iou_threshold=0.5)

    # Only highest scoring box should be kept
    assert len(keep) == 1
    assert keep[0] == 0  # Higher score


def test_batched_nms_handles_different_scenarios(nms: NMS):
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


def test_correct_nms_params():
    """Test that NMS initializes with correct parameters."""
    nms = NMS(nms_thresh=0.1, nms_mode="vectorized", n_samples=50, eps=1e-3)
    assert nms.nms_thresh == 0.1
    assert nms.nms_mode == "vectorized"
    assert nms.n_samples == 50
    assert nms.eps == 1e-3


@pytest.mark.parametrize("nms_mode", ["sequential", "vectorized", "fast"])
def test_nms_modes(nms_mode: str, overlapping_boxes: BoxesScores):
    """Test that all three NMS modes work correctly."""
    nms_instance = NMS(nms_mode=nms_mode)
    boxes, scores = overlapping_boxes

    keep = nms_instance.rotated_nms(boxes, scores, iou_threshold=0.3)

    assert len(keep) == 2
    assert 0 in keep
    assert 2 in keep


# NOTE: We can only test for scripting since those methods are getting float inputs
# which cannot be traced


def test_script_nms(nms: NMS, sample_data: SampleData):
    """Test scripting NMS forward method."""
    boxes, scores, labels = sample_data

    # Script the module
    scripted_fn = torch.jit.script(nms)

    # Test with original data
    scripted_result = scripted_fn(boxes, scores, labels, 0.5)
    eager_result = nms.forward(boxes, scores, labels, 0.5)

    assert torch.equal(scripted_result, eager_result)

    # Test with different IoU threshold
    scripted_result = scripted_fn(boxes, scores, labels, 0.7)
    eager_result = nms.forward(boxes, scores, labels, 0.7)

    assert torch.equal(scripted_result, eager_result)


def test_script_nms_dynamic_input_sizes(nms: NMS):
    """Test scripted NMS with different input sizes (N)."""
    scripted_fn = torch.jit.script(nms)

    # Test with various input sizes
    for n_boxes in [1, 2, 5, 10]:
        boxes = torch.randn(n_boxes, 5).abs() * 100
        boxes[:, 4] = torch.rand(n_boxes) * 3.14159
        scores = torch.rand(n_boxes)
        labels = torch.randint(0, 3, (n_boxes,))

        scripted_result = scripted_fn(boxes, scores, labels, 0.5)
        eager_result = nms.forward(boxes, scores, labels, 0.5)

        assert torch.equal(scripted_result, eager_result)


@pytest.mark.parametrize("nms_mode", ["sequential", "vectorized", "fast"])
def test_script_nms_all_modes(nms_mode: str, sample_data: SampleData):
    """Test scripting NMS with all three modes."""
    nms_instance = NMS(nms_thresh=0.5, nms_mode=nms_mode, n_samples=40, eps=1e-7)
    boxes, scores, labels = sample_data

    # Script the module
    scripted_fn = torch.jit.script(nms_instance)

    # Test scripted version
    scripted_result = scripted_fn(boxes, scores, labels, 0.5)
    eager_result = nms_instance.forward(boxes, scores, labels, 0.5)

    assert torch.equal(scripted_result, eager_result)


def test_nms_multiclass_suppression(nms: NMS):
    """Test that NMS correctly handles multi-class suppression."""
    # Create boxes from different classes that overlap spatially
    boxes = torch.tensor(
        [
            [100.0, 100.0, 50.0, 30.0, 0.0],
            [105.0, 105.0, 50.0, 30.0, 0.0],  # Overlaps with first, same class
            [102.0, 102.0, 50.0, 30.0, 0.0],  # Overlaps with first, different class
        ]
    )
    scores = torch.tensor([0.9, 0.85, 0.8])
    labels = torch.tensor([0, 0, 1])  # First two same class, third different

    scripted_fn = torch.jit.script(nms)

    result = scripted_fn(boxes, scores, labels, 0.5)

    # Should keep at least 2 boxes (one from each class)
    assert result.numel() >= 2
