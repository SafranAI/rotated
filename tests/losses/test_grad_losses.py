import math
from typing import TypeAlias

import pytest
import torch

from rotated.losses.mgiou import MGIoU2DLoss
from rotated.losses.prob_iou import ProbIoULoss


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


LossCls: TypeAlias = ProbIoULoss | MGIoU2DLoss


@pytest.fixture(scope="function", params=[ProbIoULoss, MGIoU2DLoss])
def loss_fn(device: torch.device, request) -> LossCls:
    loss_cls = request.param
    return loss_cls().to(device)


@pytest.fixture
def loss_fn_fast(device) -> MGIoU2DLoss:
    return MGIoU2DLoss(fast_mode=True).to(device)


def test_differentiable_loss_identical_boxes(loss_fn: LossCls, device: torch.device):
    """Test loss with identical boxes (should give loss ≈ 0)."""
    pred_boxes = torch.tensor(
        [
            [50.0, 50.0, 30.0, 20.0, 0.0],
            [100.0, 100.0, 40.0, 40.0, math.pi / 4],
            [25.0, 75.0, 10.0, 15.0, math.pi / 2],
        ],
        device=device,
        requires_grad=True,
    )
    target_boxes = pred_boxes.detach().clone()

    loss = loss_fn(pred_boxes, target_boxes)

    atol, rtol = 1e-6, 1e-6
    if isinstance(loss_fn, ProbIoULoss):
        # need to relax constraint for ProbIoULoss
        atol, rtol = 1e-3, 1e-6
    torch.testing.assert_close(loss, torch.tensor(0.0, device=device), atol=atol, rtol=rtol)
    assert loss.requires_grad, "Loss should require gradients"

    loss.backward()
    assert pred_boxes.grad is not None, "Gradients should be computed"


def test_differentiable_loss_non_overlapping_boxes(loss_fn: LossCls, device: torch.device):
    """Test loss with non-overlapping boxes (should give loss ≈ 0.5)."""
    pred_boxes = torch.tensor(
        [
            [10.0, 10.0, 5.0, 5.0, 0.0],
            [20.0, 20.0, 8.0, 8.0, 0.0],
        ],
        device=device,
        requires_grad=True,
    )
    target_boxes = torch.tensor(
        [
            [100.0, 100.0, 5.0, 5.0, 0.0],
            [200.0, 200.0, 8.0, 8.0, 0.0],
        ],
        device=device,
    )

    loss = loss_fn(pred_boxes, target_boxes)

    expected_loss = 1.0
    if isinstance(loss_fn, MGIoU2DLoss):
        # max loss value for MGIoU2DLoss is 0.5
        expected_loss = 0.5
    torch.testing.assert_close(loss, torch.tensor(expected_loss, device=device))
    assert loss.requires_grad, "Loss should require gradients"

    loss.backward()
    assert pred_boxes.grad is not None, "Gradients should be computed"


def test_differentiable_loss_partial_overlap(loss_fn: LossCls, device: torch.device):
    """Test loss with partially overlapping boxes."""
    pred_boxes = torch.tensor(
        [
            [50.0, 50.0, 30.0, 20.0, 0.0],
            [100.0, 100.0, 40.0, 40.0, 0.0],
        ],
        device=device,
        requires_grad=True,
    )
    target_boxes = torch.tensor(
        [
            [55.0, 50.0, 30.0, 20.0, 0.0],  # Slight horizontal offset
            [105.0, 105.0, 40.0, 40.0, math.pi / 6],  # Offset + rotated
        ],
        device=device,
    )

    loss = loss_fn(pred_boxes, target_boxes)

    assert 0.0 < loss.item() < 0.5
    assert loss.requires_grad, "Loss should require gradients"

    loss.backward()
    assert pred_boxes.grad is not None, "Gradients should be computed"


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_differentiable_loss_reduction_mean(device: torch.device, reduction: str):
    """Test loss with mean reduction."""
    loss_fn = MGIoU2DLoss(reduction=reduction).to(device)

    pred_boxes = torch.tensor(
        [
            [50.0, 50.0, 30.0, 20.0, 0.0],
            [100.0, 100.0, 40.0, 40.0, 0.0],
        ],
        device=device,
    )
    target_boxes = torch.tensor(
        [
            [55.0, 50.0, 30.0, 20.0, 0.0],
            [110.0, 110.0, 40.0, 40.0, 0.0],
        ],
        device=device,
    )

    loss = loss_fn(pred_boxes, target_boxes)
    assert loss.dim() == 0, "Mean reduction should return scalar"


@pytest.mark.parametrize("loss_cls", [MGIoU2DLoss, ProbIoULoss])
def test_differentiable_loss_reduction_none(loss_cls, device: torch.device):
    """Test loss without reduction (per-sample losses)."""
    loss_fn = loss_cls(reduction="none")

    pred_boxes = torch.tensor(
        [
            [50.0, 50.0, 30.0, 20.0, 0.0],
            [100.0, 100.0, 40.0, 35.0, 0.0],
            [25.0, 25.0, 15.0, 20.0, 0.0],
        ],
        device=device,
    )
    target_boxes = torch.tensor(
        [
            [55.0, 50.0, 30.0, 20.0, 0.0],
            [110.0, 110.0, 40.0, 35.0, 0.0],
            [30.0, 30.0, 15.0, 20.0, 0.0],
        ],
        device=device,
    )

    loss = loss_fn(pred_boxes, target_boxes)
    assert loss.shape == (3,), f"Expected shape (3,), got {loss.shape}"
    assert loss.min() >= 0.0
    assert loss.max() <= 1.0


def test_differentiable_loss_rotated_boxes(loss_fn: LossCls, device: torch.device):
    """Test loss with various rotations."""
    pred_boxes = torch.tensor(
        [
            [50.0, 50.0, 30.0, 20.0, 0.0],  # Horizontal
            [50.0, 50.0, 30.0, 20.0, math.pi / 4],  # 45 degrees
            [50.0, 50.0, 30.0, 20.0, math.pi / 2],  # 90 degrees
            [50.0, 50.0, 30.0, 20.0, math.pi],  # 180 degrees
        ],
        device=device,
        requires_grad=True,
    )
    target_boxes = torch.tensor(
        [
            [50.0, 50.0, 30.0, 20.0, 0.0],  # Same orientation
            [50.0, 50.0, 30.0, 20.0, math.pi / 6],  # Slight rotation diff
            [50.0, 50.0, 30.0, 20.0, 0.0],  # 90 degree difference
            [50.0, 50.0, 30.0, 20.0, 0.0],  # 180 degree difference
        ],
        device=device,
    )

    loss = loss_fn(pred_boxes, target_boxes)

    assert 0.0 < loss.item() <= 1.0, f"Expected non-zero loss for rotated boxes, got {loss.item()}"
    assert loss.requires_grad, "Loss should require gradients"

    loss.backward()
    assert pred_boxes.grad is not None, "Gradients should be computed"


def test_differentiable_loss_zero_dimensions(loss_fn: LossCls, device: torch.device):
    """Test loss with degenerate boxes (zero dimensions)."""
    pred_boxes = torch.tensor(
        [
            [50.0, 50.0, 0.0, 0.0, 0.0],  # Point
            [100.0, 100.0, 30.0, 0.0, 0.0],  # Line (zero height)
        ],
        device=device,
    )
    target_boxes = torch.tensor(
        [
            [50.0, 50.0, 10.0, 10.0, 0.0],
            [100.0, 100.0, 30.0, 20.0, 0.0],
        ],
        device=device,
    )

    loss = loss_fn(pred_boxes, target_boxes)

    assert not torch.isnan(loss), "Loss should not be NaN for degenerate boxes"
    assert 0.0 < loss.item() <= 1.0


def test_differentiable_loss_all_zero_target(loss_fn: LossCls, device: torch.device):
    """Test loss when target has all zeros (fallback to ProbIoU)."""
    pred_boxes = torch.tensor(
        [
            [50.0, 50.0, 30.0, 20.0, 0.0],
            [100.0, 100.0, 40.0, 40.0, 0.0],
        ],
        device=device,
    )
    target_boxes = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],  # All zeros
            [100.0, 100.0, 40.0, 40.0, 0.0],
        ],
        device=device,
    )

    loss = loss_fn(pred_boxes, target_boxes)

    assert not torch.isnan(loss), "Loss should not be NaN for all-zero targets"
    assert 0.0 < loss.item() <= 1.0, "Loss should be non-negative"


def test_differentiable_loss_single_box(loss_fn: LossCls, device: torch.device):
    """Test loss with a single box pair."""
    pred_boxes = torch.tensor(
        [[50.0, 50.0, 30.0, 20.0, math.pi / 6]],
        device=device,
        requires_grad=True,
    )
    target_boxes = torch.tensor(
        [[52.0, 51.0, 32.0, 21.0, math.pi / 5]],
        device=device,
    )

    loss = loss_fn(pred_boxes, target_boxes)

    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.requires_grad, "Loss should require gradients"

    loss.backward()
    assert pred_boxes.grad is not None, "Gradients should be computed"
    assert pred_boxes.grad.shape == pred_boxes.shape, "Gradient shape should match input"


def test_differentiable_loss_batch_processing(loss_fn: LossCls, device: torch.device):
    """Test loss with larger batch to verify batch processing."""
    batch_size = 100
    pred_boxes = torch.rand(batch_size, 5, device=device)
    pred_boxes[:, 2:4] = pred_boxes[:, 2:4] * 50 + 10  # Width/height in [10, 60]
    pred_boxes[:, :2] = pred_boxes[:, :2] * 100  # Center in [0, 100]
    pred_boxes[:, 4] = pred_boxes[:, 4] * math.pi / 2.0  # Angle in [0, π/2]
    pred_boxes.requires_grad_(True)

    target_boxes = torch.rand(batch_size, 5, device=device)
    target_boxes[:, 2:4] = target_boxes[:, 2:4] * 50 + 10
    target_boxes[:, :2] = target_boxes[:, :2] * 100
    target_boxes[:, 4] = target_boxes[:, 4] * math.pi / 2.0

    loss = loss_fn(pred_boxes, target_boxes)

    assert loss.dim() == 0, "Loss should be scalar"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss.requires_grad, "Loss should require gradients"

    loss.backward()
    assert pred_boxes.grad is not None, "Gradients should be computed"


def test_mgiou_loss_fast_mode_consistency(device: torch.device):
    """Test that fast mode gives similar results to regular mode."""
    loss_fn_regular = MGIoU2DLoss(fast_mode=False).to(device)
    loss_fn_fast = MGIoU2DLoss(fast_mode=True).to(device)

    pred_boxes = torch.tensor(
        [
            [50.0, 50.0, 30.0, 20.0, 0.0],
            [100.0, 100.0, 40.0, 40.0, math.pi / 4],
            [25.0, 75.0, 10.0, 15.0, math.pi / 2],
        ],
        device=device,
    )
    target_boxes = torch.tensor(
        [
            [52.0, 51.0, 32.0, 21.0, 0.1],
            [105.0, 105.0, 42.0, 42.0, math.pi / 5],
            [27.0, 77.0, 11.0, 16.0, math.pi / 2.5],
        ],
        device=device,
    )

    loss_regular = loss_fn_regular(pred_boxes, target_boxes)
    loss_fast = loss_fn_fast(pred_boxes, target_boxes)
    # both loss should be close
    torch.testing.assert_close(loss_regular.item(), loss_fast.item())


@pytest.mark.parametrize("loss_cls", [MGIoU2DLoss, ProbIoULoss])
def test_differentiable_loss_eps_parameter(device: torch.device, loss_cls: LossCls):
    """Test that eps parameter prevents division by zero."""
    loss_fn_small_eps = loss_cls(eps=1e-9).to(device)
    loss_fn_large_eps = loss_cls(eps=1e-3).to(device)

    # Create boxes with very small dimensions that might cause numerical issues
    pred_boxes = torch.tensor(
        [[50.0, 50.0, 1e-5, 1e-5, 0.0]],
        device=device,
    )
    target_boxes = torch.tensor(
        [[50.0, 50.0, 1e-5, 1e-5, 0.0]],
        device=device,
    )

    loss_small = loss_fn_small_eps(pred_boxes, target_boxes)
    loss_large = loss_fn_large_eps(pred_boxes, target_boxes)

    assert not torch.isnan(loss_small), "Loss should not be NaN with small eps"
    assert not torch.isnan(loss_large), "Loss should not be NaN with large eps"


def test_differentiable_loss_gradient_magnitude(loss_fn: LossCls, device: torch.device):
    """Test that gradients have reasonable magnitude."""
    pred_boxes = torch.tensor(
        [
            [50.0, 50.0, 30.0, 20.0, 0.0],
            [100.0, 100.0, 40.0, 40.0, math.pi / 4],
        ],
        device=device,
        requires_grad=True,
    )
    target_boxes = torch.tensor(
        [
            [55.0, 52.0, 32.0, 22.0, 0.1],
            [105.0, 103.0, 42.0, 38.0, math.pi / 5],
        ],
        device=device,
    )

    loss = loss_fn(pred_boxes, target_boxes)
    loss.backward()

    grad_norm = pred_boxes.grad.norm().item()
    assert grad_norm > 0, "Gradients should be non-zero"
    assert grad_norm < 100, f"Gradient norm too large: {grad_norm}"
    assert not torch.isnan(pred_boxes.grad).any(), "Gradients should not contain NaN"


def test_differentiable_loss_different_sizes(loss_fn: LossCls, device: torch.device):
    """Test loss with boxes of very different sizes."""
    pred_boxes = torch.tensor(
        [
            [50.0, 50.0, 5.0, 5.0, 0.0],  # Small box
            [100.0, 100.0, 100.0, 100.0, 0.0],  # Large box
        ],
        device=device,
        requires_grad=True,
    )
    target_boxes = torch.tensor(
        [
            [50.0, 50.0, 50.0, 50.0, 0.0],  # Much larger
            [100.0, 100.0, 10.0, 10.0, 0.0],  # Much smaller
        ],
        device=device,
    )

    loss = loss_fn(pred_boxes, target_boxes)

    assert not torch.isnan(loss), "Loss should not be NaN for different sized boxes"
    assert 0.0 < loss.item() <= 1.0

    loss.backward()
    assert pred_boxes.grad is not None, "Gradients should be computed"


def test_differentiable_loss_extreme_rotations(loss_fn: LossCls, device: torch.device):
    """Test loss with extreme rotation values."""
    pred_boxes = torch.tensor(
        [
            [50.0, 50.0, 30.0, 20.0, 10 * math.pi],  # Multiple rotations
            [100.0, 100.0, 40.0, 40.0, -5 * math.pi],  # Negative multiple rotations
        ],
        device=device,
        requires_grad=True,
    )
    target_boxes = torch.tensor(
        [
            [50.0, 50.0, 30.0, 20.0, 0.0],
            [100.0, 100.0, 40.0, 40.0, math.pi / 4],
        ],
        device=device,
    )

    loss = loss_fn(pred_boxes, target_boxes)

    assert not torch.isnan(loss), "Loss should not be NaN for extreme rotations"
    assert 0.0 < loss.item() < 1.0

    loss.backward()
    assert not torch.isnan(pred_boxes.grad).any(), "Gradients should not contain NaN"


def test_differentiable_loss_square_boxes(loss_fn: LossCls, device: torch.device):
    """Test loss with square boxes (equal width and height)."""
    pred_boxes = torch.tensor(
        [
            [50.0, 50.0, 20.0, 20.0, 0.0],
            [100.0, 100.0, 30.0, 30.0, math.pi / 4],
            [150.0, 150.0, 40.0, 40.0, math.pi / 2],
        ],
        device=device,
        requires_grad=True,
    )
    target_boxes = torch.tensor(
        [
            [50.0, 50.0, 20.0, 20.0, math.pi / 8],  # Rotated square
            [100.0, 100.0, 30.0, 30.0, math.pi / 3],
            [150.0, 150.0, 40.0, 40.0, 0.0],
        ],
        device=device,
    )

    loss = loss_fn(pred_boxes, target_boxes)

    assert not torch.isnan(loss), "Loss should not be NaN for square boxes"
    assert loss.requires_grad, "Loss should require gradients"

    loss.backward()
    assert pred_boxes.grad is not None, "Gradients should be computed"
