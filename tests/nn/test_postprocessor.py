import torch

from rotated.nn.postprocessor import DetectionPostProcessor


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

    _, result_scores, _ = postprocessor(boxes, scores, labels)

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
    _, result_scores, _ = postprocessor(boxes, scores, labels)

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
    _, result_scores, _ = postprocessor(boxes, scores, labels)

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
    empty_boxes = torch.empty(0, 5)
    empty_scores = torch.empty(0)
    empty_labels = torch.empty(0, dtype=torch.long)

    postprocessor = DetectionPostProcessor(detections_per_img=3, topk_candidates=5)
    result_boxes, result_scores, result_labels = postprocessor(empty_boxes, empty_scores, empty_labels)

    # Should return properly shaped tensors with padding
    assert result_boxes.shape == (3, 5)
    assert result_scores.shape == (3,)
    assert result_labels.shape == (3,)

    # All should be padding values
    assert torch.all(result_scores == 0)
    assert torch.all(result_labels == -1)


def test_torchscript_compatibility():
    """Test TorchScript compilation works correctly."""
    boxes = torch.tensor(
        [
            [100.0, 100.0, 50.0, 30.0, 0.0],
            [200.0, 200.0, 30.0, 20.0, 0.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8])
    labels = torch.tensor([0, 1])
    postprocessor = DetectionPostProcessor()

    # Test scripting
    scripted_fn = torch.jit.script(postprocessor)
    scripted_result = scripted_fn(boxes, scores, labels)

    # Test eager mode
    eager_result = postprocessor(boxes, scores, labels)

    # Results should be identical
    assert torch.allclose(scripted_result[0], eager_result[0])
    assert torch.equal(scripted_result[1], eager_result[1])
    assert torch.allclose(scripted_result[2], eager_result[2])


def test_postprocess_invalid_input_dimensions():
    """Test postprocess raises error for invalid input dimensions."""
    invalid_boxes = torch.rand(2, 3, 4, 5)  # 4D tensor
    scores = torch.rand(2, 3, 4)
    labels = torch.randint(0, 2, (2, 3, 4))
    postprocessor = DetectionPostProcessor()

    try:
        postprocessor(invalid_boxes, scores, labels)
    except ValueError as e:
        assert "Expected 2D or 3D input" in str(e)
    else:
        raise AssertionError("Expected ValueError for invalid input dimensions")
