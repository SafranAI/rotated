import pytest
import torch

from rotated.iou.approx_sdf import _gradient_central_diff_dim1, compute_rotated_iou_approx_sdf


class TestGradientCentralDiffDim1:
    """Test cases for the ONNX-compatible gradient function."""

    @pytest.mark.parametrize(
        "tensor_values",
        [
            [[0, 1, 2, 3]],
            [[1, 2]],
            [[1.5, 3.7, 5.2, 2.1]],
            [[10, 20, 30, 40, 50]],
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
        ],
    )
    def test_basic_2d_cases(self, tensor_values):
        """Test basic 2D tensor cases against torch.gradient."""
        input_tensor = torch.tensor(tensor_values, dtype=torch.float32)
        expected = torch.gradient(input_tensor, dim=1)[0]
        result = _gradient_central_diff_dim1(input_tensor)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_3d_tensors(self):
        """Test 3D tensors like those used in actual IoU computation."""
        # Test case similar to actual usage: (n, n_samples, 4)
        input_tensor = torch.randn(3, 10, 4, dtype=torch.float32)
        expected = torch.gradient(input_tensor, dim=1)[0]
        result = _gradient_central_diff_dim1(input_tensor)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_edge_cases(self):
        """Test edge cases like single element and small tensors."""
        # Single element along dim=1
        input_single = torch.tensor([[[1.0]]], dtype=torch.float32)
        result_single = _gradient_central_diff_dim1(input_single)
        expected_single = torch.zeros_like(input_single)
        assert torch.allclose(result_single, expected_single)

        # Two elements along dim=1 - torch.gradient fails here, so we test our function directly
        input_two = torch.tensor([[1.0, 2.0]], dtype=torch.float32)  # Shape: [1, 2]
        result_two = _gradient_central_diff_dim1(input_two)
        # For two elements: [2-1, 2-1] = [1, 1]
        expected_two = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        assert torch.allclose(result_two, expected_two)

    def test_torchscript_compatibility(self):
        """Test that the function can be compiled with TorchScript."""
        input_tensor = torch.randn(2, 5, 3, dtype=torch.float32)

        # This should not raise an exception
        scripted_fn = torch.jit.script(_gradient_central_diff_dim1)

        # Test that scripted version produces same results
        expected = _gradient_central_diff_dim1(input_tensor)
        result = scripted_fn(input_tensor)
        assert torch.allclose(result, expected)

    def test_numerical_stability(self):
        """Test numerical stability with very small and very large values."""
        # Very small values - shape [1, 3] for dim=1
        small_tensor = torch.tensor([[1e-8, 2e-8, 3e-8]], dtype=torch.float32)
        result_small = _gradient_central_diff_dim1(small_tensor)
        expected_small = torch.gradient(small_tensor, dim=1)[0]
        assert torch.allclose(result_small, expected_small, atol=1e-10)

        # Very large values - shape [1, 3] for dim=1
        large_tensor = torch.tensor([[1e8, 2e8, 3e8]], dtype=torch.float32)
        result_large = _gradient_central_diff_dim1(large_tensor)
        expected_large = torch.gradient(large_tensor, dim=1)[0]
        assert torch.allclose(result_large, expected_large, rtol=1e-5)


class TestComputeRotatedIoUApproxSDF:
    """Test cases for the full IoU computation function."""

    def test_perfect_overlap(self):
        """Test IoU computation with perfectly overlapping boxes."""
        pred_boxes = torch.tensor(
            [
                [50.0, 50.0, 100.0, 80.0, 0.0],
                [150.0, 150.0, 80.0, 60.0, 45.0],
            ],
            dtype=torch.float32,
        )

        target_boxes = pred_boxes.clone()

        ious = compute_rotated_iou_approx_sdf(pred_boxes, target_boxes)
        expected = torch.ones(2, dtype=torch.float32)

        assert torch.allclose(ious, expected, atol=1e-4)

    def test_no_overlap(self):
        """Test IoU computation with non-overlapping boxes."""
        pred_boxes = torch.tensor(
            [
                [10.0, 10.0, 20.0, 20.0, 0.0],
                [300.0, 300.0, 40.0, 40.0, 0.0],
            ],
            dtype=torch.float32,
        )

        target_boxes = torch.tensor(
            [
                [200.0, 200.0, 30.0, 30.0, 0.0],
                [400.0, 400.0, 50.0, 50.0, 0.0],
            ],
            dtype=torch.float32,
        )

        ious = compute_rotated_iou_approx_sdf(pred_boxes, target_boxes)
        expected = torch.zeros(2, dtype=torch.float32)

        assert torch.allclose(ious, expected, atol=1e-4)

    def test_partial_overlap(self):
        """Test IoU computation with partially overlapping boxes."""
        pred_boxes = torch.tensor(
            [
                [50.0, 50.0, 100.0, 80.0, 0.0],
            ],
            dtype=torch.float32,
        )

        target_boxes = torch.tensor(
            [
                [70.0, 70.0, 100.0, 80.0, 0.0],
            ],
            dtype=torch.float32,
        )

        ious = compute_rotated_iou_approx_sdf(pred_boxes, target_boxes)

        # Should be between 0 and 1 (not perfect overlap, not no overlap)
        assert 0 < ious[0] < 1

    def test_empty_input(self):
        """Test IoU computation with empty input tensors."""
        pred_boxes = torch.empty(0, 5, dtype=torch.float32)
        target_boxes = torch.empty(0, 5, dtype=torch.float32)

        ious = compute_rotated_iou_approx_sdf(pred_boxes, target_boxes)

        assert ious.shape == (0,)
        assert ious.dtype == pred_boxes.dtype

    def test_torchscript_compatibility(self):
        """Test that the IoU function can be compiled with TorchScript."""
        pred_boxes = torch.randn(3, 5, dtype=torch.float32)
        target_boxes = torch.randn(3, 5, dtype=torch.float32)

        # This should not raise an exception
        scripted_iou = torch.jit.script(compute_rotated_iou_approx_sdf)

        # Test that scripted version produces same results
        expected = compute_rotated_iou_approx_sdf(pred_boxes, target_boxes)
        result = scripted_iou(pred_boxes, target_boxes)
        assert torch.allclose(result, expected)

    def test_different_n_samples(self):
        """Test IoU computation with different numbers of samples."""
        pred_boxes = torch.tensor(
            [
                [50.0, 50.0, 100.0, 80.0, 30.0],
            ],
            dtype=torch.float32,
        )

        target_boxes = pred_boxes.clone()

        # Test with different n_samples values
        for n_samples in [10, 20, 40, 80]:
            ious = compute_rotated_iou_approx_sdf(pred_boxes, target_boxes, n_samples=n_samples)
            # Should be close to 1.0 for perfect overlap
            assert torch.allclose(ious, torch.ones(1), atol=0.05)

    def test_rotated_boxes(self):
        """Test IoU computation with rotated boxes."""
        pred_boxes = torch.tensor(
            [
                [100.0, 100.0, 80.0, 60.0, 0.0],
                [100.0, 100.0, 80.0, 60.0, 45.0],
                [100.0, 100.0, 80.0, 60.0, 90.0],
            ],
            dtype=torch.float32,
        )

        target_boxes = torch.tensor(
            [
                [100.0, 100.0, 80.0, 60.0, 0.0],
                [100.0, 100.0, 80.0, 60.0, 45.0],
                [100.0, 100.0, 80.0, 60.0, 90.0],
            ],
            dtype=torch.float32,
        )

        ious = compute_rotated_iou_approx_sdf(pred_boxes, target_boxes)
        expected = torch.ones(3, dtype=torch.float32)

        assert torch.allclose(ious, expected, atol=1e-4)
