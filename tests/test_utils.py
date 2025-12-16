"""Tests for check_is_valid_box function."""

import pytest
import torch

from src.rotated.boxes.utils import check_is_valid_box


@pytest.mark.parametrize(
    "boxes, expected",
    [
        # Valid boxes
        (torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.0]]), torch.tensor([True])),
        (torch.tensor([[1.0, 2.0, 3.0, 4.0]]), torch.tensor([True])),
        # Invalid due to small size
        (torch.tensor([[1.0, 2.0, 1e-7, 4.0, 0.0]]), torch.tensor([False])),
        (torch.tensor([[1.0, 2.0, 3.0, 1e-7, 0.0]]), torch.tensor([False])),
        # Invalid due to NaN
        (torch.tensor([[1.0, 2.0, float("nan"), 4.0, 0.0]]), torch.tensor([False])),
        # Invalid due to inf
        (torch.tensor([[1.0, 2.0, 3.0, float("inf"), 0.0]]), torch.tensor([False])),
        # Mixed valid and invalid
        (
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.0], [1.0, 2.0, 1e-7, 4.0, 0.0], [1.0, 2.0, 3.0, float("nan"), 0.0]]),
            torch.tensor([True, False, False]),
        ),
    ],
    ids=[
        "valid_rotated_box",
        "valid_horizontal_box",
        "invalid_small_width",
        "invalid_small_height",
        "invalid_nan_value",
        "invalid_inf_value",
        "mixed_valid_invalid",
    ],
)
def test_check_is_valid_box(boxes, expected):
    """Test check_is_valid_box with various inputs."""
    result = check_is_valid_box(boxes)
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"


def test_check_is_valid_box_custom_threshold():
    """Test check_is_valid_box with custom min_box_size."""
    boxes = torch.tensor([[1.0, 2.0, 0.1, 4.0, 0.0]])

    # Should be valid with default threshold
    assert check_is_valid_box(boxes).item() is True

    # Should be invalid with higher threshold
    assert check_is_valid_box(boxes, min_box_size=0.2).item() is False
