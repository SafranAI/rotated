from pathlib import Path
from typing import TypeAlias

import pytest
import torch

from rotated.nn.postprocessor import DetectionPostProcessor


def test_postprocess_score_filtering():
    """Test postprocess filters out low-confidence detections."""
    boxes = torch.tensor(
        [
            [
                [100.0, 100.0, 30.0, 20.0, 0.0],
                [200.0, 200.0, 30.0, 20.0, 0.0],
                [300.0, 300.0, 30.0, 20.0, 0.0],
            ]
        ]
    )
    scores = torch.tensor([[0.9, 0.03, 0.7]])  # Middle score below threshold
    labels = torch.tensor([[0, 1, 2]])

    postprocessor = DetectionPostProcessor(detections_per_img=5, score_thresh=0.1)

    _, result_scores, _ = postprocessor(boxes, scores, labels)

    # Should keep only 2 boxes (scores 0.9 and 0.7)
    valid_mask = result_scores[0] > 0
    num_valid = valid_mask.sum().item()
    assert num_valid == 2

    # Verify kept scores are above threshold
    valid_scores = result_scores[0][valid_mask]
    assert torch.all(valid_scores >= 0.1)


def test_postprocess_topk_candidates():
    """Test postprocess limits detections with topk_candidates."""
    # Create 5 boxes with decreasing scores
    boxes = torch.tensor(
        [
            [
                [100.0, 100.0, 30.0, 20.0, 0.0],
                [200.0, 200.0, 30.0, 20.0, 0.0],
                [300.0, 300.0, 30.0, 20.0, 0.0],
                [400.0, 400.0, 30.0, 20.0, 0.0],
                [500.0, 500.0, 30.0, 20.0, 0.0],
            ]
        ]
    )
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5]])  # All above threshold
    labels = torch.tensor([[0, 1, 2, 3, 4]])  # All different classes

    postprocessor = DetectionPostProcessor(topk_candidates=3)
    _, result_scores, _ = postprocessor(boxes, scores, labels)

    # Should keep only top 3 by score (topk_candidates=3)
    valid_mask = result_scores[0] > 0
    num_valid = valid_mask.sum().item()
    assert num_valid == 3

    # Verify they are the highest scoring ones
    valid_scores = result_scores[0][valid_mask]
    expected_scores = torch.tensor([0.9, 0.8, 0.7])
    assert torch.allclose(valid_scores.sort(descending=True)[0], expected_scores)


def test_postprocess_nms_suppression():
    """Test postprocess applies NMS to overlapping boxes."""
    boxes = torch.tensor(
        [
            [
                [100.0, 100.0, 50.0, 30.0, 0.0],  # High score
                [102.0, 102.0, 52.0, 32.0, 0.0],  # Lower score, overlaps with first
                [200.0, 200.0, 30.0, 20.0, 0.0],  # Separate box
            ]
        ]
    )
    scores = torch.tensor([[0.9, 0.8, 0.6]])
    labels = torch.tensor([[0, 0, 1]])  # First two same class

    postprocessor = DetectionPostProcessor(detections_per_img=5, topk_candidates=10)
    _, result_scores, _ = postprocessor(boxes, scores, labels)

    # Should suppress overlapping box, keep 2 total
    valid_mask = result_scores[0] > 0
    num_valid = valid_mask.sum().item()
    assert num_valid == 2

    # Should keep highest scoring box from overlapping pair + separate box
    valid_scores = result_scores[0][valid_mask]
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
    result_boxes, result_scores, result_labels = postprocessor(batch_boxes, batch_scores, batch_labels)

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
    empty_boxes = torch.empty(1, 0, 5)  # Batched: [1, 0, 5]
    empty_scores = torch.empty(1, 0)
    empty_labels = torch.empty(1, 0, dtype=torch.long)

    postprocessor = DetectionPostProcessor(detections_per_img=3, topk_candidates=5)
    result_boxes, result_scores, result_labels = postprocessor(empty_boxes, empty_scores, empty_labels)

    # Should return properly shaped tensors with padding
    assert result_boxes.shape == (1, 3, 5)
    assert result_scores.shape == (1, 3)
    assert result_labels.shape == (1, 3)

    # All should be padding values
    assert torch.all(result_scores == 0)
    assert torch.all(result_labels == -1)


def test_postprocess_invalid_input_dimensions():
    """Test postprocess raises error for invalid input dimensions."""
    invalid_boxes = torch.rand(2, 3, 4, 5)  # 4D tensor
    scores = torch.rand(2, 3, 4)
    labels = torch.randint(0, 2, (2, 3, 4))
    postprocessor = DetectionPostProcessor()

    with pytest.raises(ValueError, match="Expected 3D batched input"):
        postprocessor(invalid_boxes, scores, labels)

    # Also test 2D input (single sample without batch dim)
    invalid_boxes_2d = torch.rand(3, 5)
    scores_2d = torch.rand(3)
    labels_2d = torch.randint(0, 2, (3,))

    with pytest.raises(ValueError, match="Expected 3D batched input"):
        postprocessor(invalid_boxes_2d, scores_2d, labels_2d)


@pytest.fixture
def postprocessor() -> DetectionPostProcessor:
    """Create a postprocessor instance."""
    return DetectionPostProcessor(score_thresh=0.05, nms_thresh=0.5, detections_per_img=100, topk_candidates=200)


SampleData: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@pytest.fixture
def sample_batched_data() -> SampleData:
    """Create sample batched detection data."""
    boxes = torch.tensor(
        [
            [
                [100.0, 100.0, 50.0, 30.0, 0.0],
                [200.0, 200.0, 30.0, 20.0, 0.0],
                [150.0, 150.0, 40.0, 25.0, 0.5],
            ],
            [
                [50.0, 50.0, 20.0, 15.0, 0.0],
                [250.0, 250.0, 35.0, 25.0, 0.3],
                [180.0, 180.0, 45.0, 30.0, 0.8],
            ],
        ]
    )
    scores = torch.tensor(
        [
            [0.9, 0.8, 0.7],
            [0.85, 0.75, 0.65],
        ]
    )
    labels = torch.tensor(
        [
            [0, 1, 0],
            [1, 0, 2],
        ]
    )
    return boxes, scores, labels


def test_trace_postprocessor_dynamic_batch_size(postprocessor: DetectionPostProcessor, sample_batched_data: SampleData):
    """Test tracing with batch_size=2, then inference with different batch sizes."""
    boxes, scores, labels = sample_batched_data

    # Trace with batch_size=2
    traced_fn = torch.jit.trace(postprocessor, example_inputs=(boxes, scores, labels))

    # Test with batch_size=1 (smaller)
    boxes_b1 = boxes[:1]
    scores_b1 = scores[:1]
    labels_b1 = labels[:1]

    traced_result = traced_fn(boxes_b1, scores_b1, labels_b1)
    eager_result = postprocessor(boxes_b1, scores_b1, labels_b1)

    assert traced_result[0].shape[0] == 1
    assert torch.allclose(traced_result[0], eager_result[0])
    assert torch.allclose(traced_result[1], eager_result[1])
    assert torch.equal(traced_result[2], eager_result[2])

    # Test with batch_size=4 (larger)
    boxes_b4 = boxes.repeat(2, 1, 1)
    scores_b4 = scores.repeat(2, 1)
    labels_b4 = labels.repeat(2, 1)

    traced_result = traced_fn(boxes_b4, scores_b4, labels_b4)
    eager_result = postprocessor(boxes_b4, scores_b4, labels_b4)

    assert traced_result[0].shape[0] == 4
    assert torch.allclose(traced_result[0], eager_result[0])
    assert torch.allclose(traced_result[1], eager_result[1])
    assert torch.equal(traced_result[2], eager_result[2])


def test_script_postprocessor(postprocessor: DetectionPostProcessor, sample_batched_data: SampleData):
    """Test scripting the postprocessor."""
    boxes, scores, labels = sample_batched_data

    # Script the model
    scripted_fn = torch.jit.script(postprocessor)

    # Test with original batch size
    scripted_result = scripted_fn(boxes, scores, labels)
    eager_result = postprocessor(boxes, scores, labels)

    assert torch.allclose(scripted_result[0], eager_result[0])
    assert torch.allclose(scripted_result[1], eager_result[1])
    assert torch.equal(scripted_result[2], eager_result[2])

    # Test with different batch size
    boxes_b1 = boxes[:1]
    scores_b1 = scores[:1]
    labels_b1 = labels[:1]

    scripted_result = scripted_fn(boxes_b1, scores_b1, labels_b1)
    eager_result = postprocessor(boxes_b1, scores_b1, labels_b1)

    assert scripted_result[0].shape[0] == 1
    assert torch.allclose(scripted_result[0], eager_result[0])


def test_trace_postprocessor_save_and_load(
    postprocessor: DetectionPostProcessor, sample_batched_data: SampleData, tmp_path: Path
):
    """Test saving and loading traced postprocessor with different batch size."""
    boxes, scores, labels = sample_batched_data

    # Trace with batch_size=2 and save
    traced_fn = torch.jit.trace(postprocessor, example_inputs=(boxes, scores, labels))
    save_path = tmp_path / "traced_postprocessor.pt"
    torch.jit.save(traced_fn, save_path)

    # Load and test with batch_size=1
    loaded_fn = torch.jit.load(save_path)
    boxes_b1 = boxes[:1]
    scores_b1 = scores[:1]
    labels_b1 = labels[:1]

    loaded_result = loaded_fn(boxes_b1, scores_b1, labels_b1)
    eager_result = postprocessor(boxes_b1, scores_b1, labels_b1)

    assert torch.allclose(loaded_result[0], eager_result[0])
    assert torch.allclose(loaded_result[1], eager_result[1])
    assert torch.equal(loaded_result[2], eager_result[2])


def test_postprocessor_input_validation(postprocessor: DetectionPostProcessor):
    """Test that input validation works correctly."""
    # Test with 2D input (should fail)
    boxes_2d = torch.randn(10, 5)
    scores_2d = torch.randn(10)
    labels_2d = torch.randint(0, 5, (10,))

    with pytest.raises(ValueError, match="Expected 3D batched input"):
        postprocessor(boxes_2d, scores_2d, labels_2d)
