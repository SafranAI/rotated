"""IoU Comparison Tool for Rotated Bounding Boxes.

Compares different IoU implementations including probabilistic, approximation,
precise geometric, and optional Shapely-based methods.
"""

import math

import torch

try:
    from shapely.affinity import rotate, translate
    from shapely.geometry import Polygon

    SHAPELY_AVAILABLE = True
except ImportError:
    Polygon = None
    rotate = None
    translate = None
    SHAPELY_AVAILABLE = False

from rotated.iou.approx_iou import ApproxRotatedIoU
from rotated.iou.precise_iou import PreciseRotatedIoU
from rotated.iou.prob_iou import ProbIoU
from rotated.losses.prob_iou import ProbIoULoss


class ShapelyIoU:
    """IoU computation using Shapely library for comparison."""

    def __init__(self):
        if not SHAPELY_AVAILABLE:
            raise ImportError("Shapely is required for ShapelyIoU. Install with: pip install shapely")

    def __call__(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU using Shapely.

        Args:
            pred_boxes: [N, 5] - (x, y, w, h, angle)
            target_boxes: [N, 5] - (x, y, w, h, angle)

        Returns:
            IoU values [N]
        """
        batch_size = pred_boxes.shape[0]
        ious = torch.zeros(batch_size, device=pred_boxes.device, dtype=pred_boxes.dtype)

        for idx in range(batch_size):
            pred_poly = self._box_to_shapely_polygon(pred_boxes[idx])
            target_poly = self._box_to_shapely_polygon(target_boxes[idx])

            try:
                intersection_area = pred_poly.intersection(target_poly).area
                union_area = pred_poly.union(target_poly).area

                if union_area > 0:
                    ious[idx] = intersection_area / union_area
                else:
                    ious[idx] = 0.0
            except Exception:
                # Handle degenerate cases
                ious[idx] = 0.0

        return ious

    def _box_to_shapely_polygon(self, box: torch.Tensor):
        """Convert a rotated box to Shapely polygon.

        Args:
            box: [5] - (x, y, w, h, angle)

        Returns:
            Shapely Polygon object
        """
        center_x, center_y, width, height, angle = box.cpu().numpy()

        # Create rectangle centered at origin
        half_width, half_height = width / 2, height / 2
        coords = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        ]

        # Create polygon
        poly = Polygon(coords)

        # Rotate around center
        poly = rotate(poly, math.degrees(angle), origin="centroid")

        # Translate to final position
        poly = translate(poly, center_x, center_y)

        return poly


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing Rotated IoU Implementations")
    print("=" * 50)

    # Create calculators
    prob_iou = ProbIoU()
    approx_iou = ApproxRotatedIoU()
    precise_iou = PreciseRotatedIoU()

    if SHAPELY_AVAILABLE:
        shapely_iou = ShapelyIoU()
        print("✓ Shapely available for comparison")
    else:
        shapely_iou = None
        print("⚠ Shapely not available - install with: pip install shapely")

    # Test cases
    pred_boxes = torch.tensor(
        [
            [50, 50, 30, 20, 0.0],  # Horizontal rectangle
            [50, 50, 30, 20, math.pi / 4],  # 45-degree rotation
            [50, 50, 20, 30, math.pi / 2],  # 90-degree rotation
            [50, 50, 10, 10, 0.0],  # Small square
            [50, 50, 40, 40, math.pi / 6],  # Large rotated square
        ],
        device=device,
        dtype=torch.float32,
    )

    target_boxes = torch.tensor(
        [
            [50, 50, 30, 20, 0.0],  # Identical
            [55, 55, 30, 20, math.pi / 4],  # Slightly offset
            [50, 50, 20, 30, 0.0],  # Different rotation
            [50, 50, 15, 15, 0.0],  # Different size
            [60, 60, 40, 40, math.pi / 6],  # Translated
        ],
        device=device,
        dtype=torch.float32,
    )

    print(f"\nTesting {pred_boxes.shape[0]} box pairs")

    # Test ProbIoU
    with torch.no_grad():
        prob_ious = prob_iou(pred_boxes, target_boxes)
    print(f"ProbIoU IoU: {prob_ious.cpu().numpy()}")

    # Test ProbIoU Loss
    prob_loss = ProbIoULoss(mode="l1")
    with torch.no_grad():
        prob_loss_values = 1 - prob_loss(pred_boxes, target_boxes)
    print(f"ProbIoU (from loss): {prob_loss_values.cpu().numpy()}")

    # Test Approximation IoU
    with torch.no_grad():
        approx_ious = approx_iou(pred_boxes, target_boxes)
    print(f"Approximation IoU: {approx_ious.cpu().numpy()}")

    # Test Precise IoU
    with torch.no_grad():
        precise_ious = precise_iou(pred_boxes, target_boxes)
    print(f"Precise IoU: {precise_ious.cpu().numpy()}")

    # Test Shapely IoU if available
    if shapely_iou is not None:
        with torch.no_grad():
            shapely_ious = shapely_iou(pred_boxes, target_boxes)
        print(f"Shapely IoU: {shapely_ious.cpu().numpy()}")

    # Test identical boxes (should give IoU ≈ 1.0)
    print("\n--- Testing Identical Boxes ---")
    identical_prob = prob_iou(pred_boxes[:3], pred_boxes[:3])
    identical_approx = approx_iou(pred_boxes[:3], pred_boxes[:3])
    identical_precise = precise_iou(pred_boxes[:3], pred_boxes[:3])

    print(f"ProbIoU (identical): {identical_prob.cpu().numpy()}")
    print(f"Approximation (identical): {identical_approx.cpu().numpy()}")
    print(f"Precise (identical): {identical_precise.cpu().numpy()}")

    if shapely_iou is not None:
        identical_shapely = shapely_iou(pred_boxes[:3], pred_boxes[:3])
        print(f"Shapely (identical): {identical_shapely.cpu().numpy()}")

    # Accuracy comparison vs Shapely
    if shapely_iou is not None:
        print("\n--- Accuracy Comparison vs Shapely ---")
        shapely_results = shapely_iou(pred_boxes, target_boxes)
        precise_results = precise_iou(pred_boxes, target_boxes)
        approx_results = approx_iou(pred_boxes, target_boxes)

        precise_mae = torch.mean(torch.abs(precise_results - shapely_results)).item()
        approx_mae = torch.mean(torch.abs(approx_results - shapely_results)).item()

        print(f"Precise IoU MAE vs Shapely: {precise_mae:.6f}")
        print(f"Approximation IoU MAE vs Shapely: {approx_mae:.6f}")

    # Performance comparison
    print("\n--- Performance Test ---")
    large_batch = 1000
    large_pred = torch.rand(large_batch, 5, device=device) * 100
    large_target = torch.rand(large_batch, 5, device=device) * 100

    import time

    # ProbIoU timing
    start = time.time()
    for _ in range(10):
        _ = prob_iou(large_pred, large_target)
    prob_time = (time.time() - start) / 10

    print(f"ProbIoU time (1000 boxes): {prob_time * 1000:.2f}ms")

    # Approximation IoU timing
    start = time.time()
    for _ in range(3):
        _ = approx_iou(large_pred, large_target)
    approx_time = (time.time() - start) / 3

    print(f"Approximation IoU time (1000 boxes): {approx_time * 1000:.2f}ms")

    # Precise IoU timing
    start = time.time()
    for _ in range(3):
        _ = precise_iou(large_pred, large_target)
    precise_time = (time.time() - start) / 3

    print(f"Precise IoU time (1000 boxes): {precise_time * 1000:.2f}ms")

    if shapely_iou is not None:
        # Shapely timing (CPU only)
        cpu_pred = large_pred[:100].cpu()
        cpu_target = large_target[:100].cpu()

        start = time.time()
        _ = shapely_iou(cpu_pred, cpu_target)
        shapely_time = time.time() - start

        print(f"Shapely IoU time (100 boxes): {shapely_time * 1000:.2f}ms")

    # Test gradients (only for differentiable methods)
    print("\n--- Gradient Test ---")
    test_pred = pred_boxes[:3].clone().requires_grad_(True)
    test_target = target_boxes[:3].clone()

    # ProbIoU gradients
    loss = prob_iou(test_pred, test_target).sum()
    loss.backward()
    grad_norm = test_pred.grad.norm().item()

    print(f"ProbIoU gradient norm: {grad_norm:.6f}")
    print("Note: Precise and Approximation IoU methods are not differentiable")

    print("\nAll tests completed!")
