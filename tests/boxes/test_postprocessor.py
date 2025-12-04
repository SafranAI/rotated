import pytest
import torch

from rotated.nn.postprocessor import DetectionPostProcessor


@pytest.fixture()
def postprocessor():
    return DetectionPostProcessor()


def test_rotated_nms_suppresses_overlapping_boxes(postprocessor):
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

    keep = postprocessor.rotated_nms(boxes, scores, iou_threshold=0.3)

    # Should keep exactly 2 boxes (highest scoring overlapping + separate)
    assert len(keep) == 2
    # Should keep box 0 (highest score) and box 2 (separate)
    assert 0 in keep
    assert 2 in keep
    assert 1 not in keep  # Overlapping box with lower score should be suppressed


def test_rotated_nms_preserves_non_overlapping(postprocessor):
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

    keep = postprocessor.rotated_nms(boxes, scores, iou_threshold=0.5)

    # All boxes should be kept since they don't overlap
    assert len(keep) == 3
    assert torch.equal(keep, torch.tensor([0, 1, 2]))  # Sorted by score


def test_multiclass_nms_preserves_different_classes(postprocessor):
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

    keep = postprocessor.multiclass_rotated_nms(boxes, scores, labels, iou_threshold=0.5)

    # Both should be kept despite overlap (different classes)
    assert len(keep) == 2
    assert 0 in keep
    assert 1 in keep


def test_multiclass_nms_suppresses_same_class(postprocessor):
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

    keep = postprocessor.multiclass_rotated_nms(boxes, scores, labels, iou_threshold=0.5)

    # Only highest scoring box should be kept
    assert len(keep) == 1
    assert keep[0] == 0  # Higher score


def test_batched_nms_handles_different_scenarios(postprocessor):
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

    keep = postprocessor.batched_multiclass_rotated_nms(boxes, scores, labels, 0.5, max_output_per_batch=5)

    # Batch 1: all 3 boxes should be kept (non-overlapping)
    batch1_valid = keep[0][keep[0] >= 0]
    assert len(batch1_valid) == 3

    # Batch 2: 2 boxes should be kept (one overlapping suppressed, one separate kept)
    batch2_valid = keep[1][keep[1] >= 0]
    assert len(batch2_valid) == 2


def test_postprocess_score_filtering():
    """Test postprocess filters out low-confidence detections."""
    boxes = torch.tensor(
        [
            [100.0, 100.0, 30.0, 20.0, 0.0],
            [200.0, 200.0, 30.0, 20.0, 0.0],
            [300.0, 300.0, 30.0, 20.0, 0.0],
        ]
    )
    scores = torch.tensor([0.9, 0.03, 0.7])  # Middle score below threshold
    labels = torch.tensor([0, 1, 2])

    postprocessor = DetectionPostProcessor(detections_per_img=5)

    _, result_scores, _ = postprocessor.forward(boxes, scores, labels)

    # Should keep only 2 boxes (scores 0.9 and 0.7)
    valid_mask = result_scores > 0
    num_valid = valid_mask.sum().item()
    assert num_valid == 2

    # Verify kept scores are above threshold
    valid_scores = result_scores[valid_mask]
    assert torch.all(valid_scores >= 0.05)


def test_postprocess_topk_candidates():
    """Test postprocess limits detections with topk_candidates."""
    # Create 5 boxes with decreasing scores
    boxes = torch.tensor(
        [
            [100.0, 100.0, 30.0, 20.0, 0.0],
            [200.0, 200.0, 30.0, 20.0, 0.0],
            [300.0, 300.0, 30.0, 20.0, 0.0],
            [400.0, 400.0, 30.0, 20.0, 0.0],
            [500.0, 500.0, 30.0, 20.0, 0.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])  # All above threshold
    labels = torch.tensor([0, 1, 2, 3, 4])  # All different classes

    postprocessor = DetectionPostProcessor(topk_candidates=3)
    _, result_scores, _ = postprocessor.forward(boxes, scores, labels)

    # Should keep only top 3 by score (topk_candidates=3)
    valid_mask = result_scores > 0
    num_valid = valid_mask.sum().item()
    assert num_valid == 3

    # Verify they are the highest scoring ones
    valid_scores = result_scores[valid_mask]
    expected_scores = torch.tensor([0.9, 0.8, 0.7])
    assert torch.allclose(valid_scores.sort(descending=True)[0], expected_scores)


def test_postprocess_nms_suppression():
    """Test postprocess applies NMS to overlapping boxes."""
    boxes = torch.tensor(
        [
            [100.0, 100.0, 50.0, 30.0, 0.0],  # High score
            [102.0, 102.0, 52.0, 32.0, 0.0],  # Lower score, overlaps with first
            [200.0, 200.0, 30.0, 20.0, 0.0],  # Separate box
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.6])
    labels = torch.tensor([0, 0, 1])  # First two same class

    postprocessor = DetectionPostProcessor(detections_per_img=5, topk_candidates=10)
    _, result_scores, _ = postprocessor.forward(boxes, scores, labels)

    # Should suppress overlapping box, keep 2 total
    valid_mask = result_scores > 0
    num_valid = valid_mask.sum().item()
    assert num_valid == 2

    # Should keep highest scoring box from overlapping pair + separate box
    valid_scores = result_scores[valid_mask]
    assert 0.9 in valid_scores  # Highest scoring overlapping box
    assert 0.6 in valid_scores  # Separate box
    assert 0.8 not in valid_scores  # Suppressed box


def test_postprocess_batched_input():
    """Test postprocess handles batched input correctly."""
    # Create simple test cases for each batch
    boxes1 = torch.tensor(
        [
            [100.0, 100.0, 30.0, 20.0, 0.0],
            [200.0, 200.0, 30.0, 20.0, 0.0],
        ]
    )
    boxes2 = torch.tensor(
        [
            [100.0, 100.0, 30.0, 20.0, 0.0],
            [200.0, 200.0, 30.0, 20.0, 0.0],
        ]
    )

    batch_boxes = torch.stack([boxes1, boxes2])
    batch_scores = torch.tensor([[0.9, 0.8], [0.9, 0.02]])  # Second batch has low score
    batch_labels = torch.tensor([[0, 1], [0, 1]])

    postprocessor = DetectionPostProcessor(detections_per_img=3, topk_candidates=3)
    result_boxes, result_scores, result_labels = postprocessor.forward(batch_boxes, batch_scores, batch_labels)

    # Check output shapes
    assert result_boxes.shape == (2, 3, 5)
    assert result_scores.shape == (2, 3)
    assert result_labels.shape == (2, 3)

    # Batch 1 should have 2 valid detections
    batch1_valid = (result_scores[0] > 0).sum().item()
    assert batch1_valid == 2

    # Batch 2 should have 1 valid detection (score filtering)
    batch2_valid = (result_scores[1] > 0).sum().item()
    assert batch2_valid == 1


def test_postprocess_empty_input():
    """Test postprocess handles empty input correctly."""
    empty_boxes = torch.empty(0, 5)
    empty_scores = torch.empty(0)
    empty_labels = torch.empty(0, dtype=torch.long)

    postprocessor = DetectionPostProcessor(detections_per_img=3, topk_candidates=5)
    result_boxes, result_scores, result_labels = postprocessor.forward(empty_boxes, empty_scores, empty_labels)

    # Should return properly shaped tensors with padding
    assert result_boxes.shape == (3, 5)
    assert result_scores.shape == (3,)
    assert result_labels.shape == (3,)

    # All should be padding values
    assert torch.all(result_scores == 0)
    assert torch.all(result_labels == -1)


def test_torchscript_compatibility(postprocessor):
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
    scripted_fn = torch.jit.script(postprocessor)
    scripted_result = scripted_fn(boxes, scores, labels)

    # Test eager mode
    eager_result = postprocessor.forward(boxes, scores, labels)

    # Results should be identical
    assert torch.allclose(scripted_result[0], eager_result[0])
    assert torch.equal(scripted_result[1], eager_result[1])
    assert torch.allclose(scripted_result[2], eager_result[2])


def test_postprocess_invalid_input_dimensions(postprocessor):
    """Test postprocess raises error for invalid input dimensions."""
    invalid_boxes = torch.rand(2, 3, 4, 5)  # 4D tensor
    scores = torch.rand(2, 3, 4)
    labels = torch.randint(0, 2, (2, 3, 4))

    try:
        postprocessor.forward(invalid_boxes, scores, labels)
    except ValueError as e:
        assert "Expected 2D or 3D input" in str(e)
    else:
        raise AssertionError("Expected ValueError for invalid input dimensions")
