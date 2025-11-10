"""IoU Comparison Tool for Rotated Bounding Boxes.

Compares different IoU implementations including probabilistic, approximation,
precise geometric, and optional Shapely-based methods.
"""

from dataclasses import dataclass
import math
import time

import torch

try:
    from shapely.affinity import rotate, translate
    from shapely.geometry import Polygon

    SHAPELY_AVAILABLE = True
except ImportError:
    Polygon = rotate = translate = None
    SHAPELY_AVAILABLE = False

from rotated.iou.approx_iou import ApproxRotatedIoU
from rotated.iou.approx_sdf import ApproxSDFL1
from rotated.iou.precise_iou import PreciseRotatedIoU
from rotated.iou.prob_iou import ProbIoU
from rotated.utils.seed import seed_everything


@dataclass
class IoUMethod:
    """Configuration for an IoU calculation method."""

    name: str
    calculator: object
    supports_gradients: bool = False
    force_cpu: bool = False


IOU_METHODS = [
    IoUMethod("ProbIoU", ProbIoU(), supports_gradients=True),
    IoUMethod("Approximation", ApproxRotatedIoU()),
    IoUMethod("Precise", PreciseRotatedIoU()),
    IoUMethod("SDF-L1", ApproxSDFL1()),
    # Add your custom methods here, e.g.:
    # IoUMethod("MyCustomIoU", MyCustomIoU(), supports_gradients=True),
]


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


class IoUComparator:
    """Compares different IoU implementations based on configuration."""

    def __init__(self, device: torch.device, methods: list[IoUMethod] = None, reference_method: str = "Shapely"):
        self.device = device
        self.methods = methods if methods is not None else IOU_METHODS.copy()
        self.reference_method = reference_method

        # Add Shapely if available and not already in methods
        if SHAPELY_AVAILABLE and not any(m.name == "Shapely" for m in self.methods):
            self.methods.append(IoUMethod("Shapely", ShapelyIoU(), force_cpu=True))

    def _prepare_tensors(
        self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor, force_cpu: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare tensors for computation, moving to CPU if required."""
        if force_cpu:
            return pred_boxes.cpu(), target_boxes.cpu()
        return pred_boxes, target_boxes

    def compute_all(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute IoU using all configured methods."""
        results = {}

        for method in self.methods:
            test_pred, test_target = self._prepare_tensors(pred_boxes, target_boxes, method.force_cpu)
            with torch.no_grad():
                results[method.name] = method.calculator(test_pred, test_target)

        return results

    def benchmark(self, batch_size: int = 1000, num_iterations: int = 10) -> dict[str, float]:
        """Benchmark performance of all configured methods."""
        pred_boxes = torch.rand(batch_size, 5, device=self.device) * 100
        target_boxes = torch.rand(batch_size, 5, device=self.device) * 100

        timings = {}

        for method in self.methods:
            test_pred, test_target = self._prepare_tensors(pred_boxes, target_boxes, method.force_cpu)

            start = time.perf_counter()
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = method.calculator(test_pred, test_target)
            timings[method.name] = (time.perf_counter() - start) / num_iterations * 1000

        return timings

    def compare_accuracy(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> dict[str, float]:
        """Compare accuracy against the configured reference method."""
        # Find reference method
        reference_method = None
        for method in self.methods:
            if method.name == self.reference_method:
                reference_method = method
                break

        if reference_method is None:
            return {}

        # Get reference results
        ref_pred, ref_target = self._prepare_tensors(pred_boxes, target_boxes, reference_method.force_cpu)
        with torch.no_grad():
            reference_results = reference_method.calculator(ref_pred, ref_target)

        mae_scores = {}
        for method in self.methods:
            if method.name == self.reference_method:
                continue

            test_pred, test_target = self._prepare_tensors(pred_boxes, target_boxes, method.force_cpu)
            with torch.no_grad():
                results = method.calculator(test_pred, test_target)

            # Ensure both results are on same device for comparison
            if results.device != reference_results.device:
                results = results.to(reference_results.device)

            mae_scores[method.name] = torch.mean(torch.abs(results - reference_results)).item()

        return mae_scores

    def test_gradients(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> dict[str, float]:
        """Test gradient computation for methods that support it."""
        gradient_norms = {}

        for method in self.methods:
            if not method.supports_gradients:
                continue

            test_pred = pred_boxes.clone().requires_grad_(True)
            loss = method.calculator(test_pred, target_boxes).sum()
            loss.backward()
            gradient_norms[method.name] = test_pred.grad.norm().item()

        return gradient_norms


def create_test_boxes(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create test box pairs with various configurations."""
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

    return pred_boxes, target_boxes


def main():
    """Main execution function."""
    # Seed for reproducibility
    seed_everything(seed=42, deterministic=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing Rotated IoU Implementations")
    print("=" * 50)
    print(f"Device: {device}")
    print("Random seed: 42")

    # Use Shapely as reference if available, otherwise use first method
    reference = "Shapely" if SHAPELY_AVAILABLE else IOU_METHODS[0].name
    comparator = IoUComparator(device, reference_method=reference)

    print(f"Configured methods: {', '.join(m.name for m in comparator.methods)}")
    print(f"Reference method: {comparator.reference_method}")

    pred_boxes, target_boxes = create_test_boxes(device)

    # Test 1: Compare all methods
    print(f"\nTesting {pred_boxes.shape[0]} box pairs")
    results = comparator.compute_all(pred_boxes, target_boxes)
    for method_name, ious in results.items():
        print(f"{method_name:20s}: {ious.cpu().numpy()}")

    # Test 2: Identical boxes (should give IoU ≈ 1.0)
    print("\n--- Testing Identical Boxes ---")
    identical_results = comparator.compute_all(pred_boxes[:3], pred_boxes[:3])
    for method_name, ious in identical_results.items():
        print(f"{method_name:20s}: {ious.cpu().numpy()}")

    # Test 3: Accuracy comparison
    mae_scores = comparator.compare_accuracy(pred_boxes, target_boxes)
    if mae_scores:
        print(f"\n--- Accuracy Comparison vs {comparator.reference_method} ---")
        for method_name, mae in mae_scores.items():
            print(f"{method_name:20s} MAE: {mae:.6f}")

    # Test 4: Performance benchmark
    print("\n--- Performance Test ---")
    timings = comparator.benchmark(batch_size=1000, num_iterations=10)
    for method_name, time_ms in timings.items():
        print(f"{method_name:20s}: {time_ms:9.2f}ms")

    # Test 5: Gradient test
    print("\n--- Gradient Test ---")
    grad_norms = comparator.test_gradients(pred_boxes[:3], target_boxes[:3])
    if grad_norms:
        for method_name, grad_norm in grad_norms.items():
            print(f"{method_name:20s} gradient norm: {grad_norm:.6f}")
        non_diff = [m.name for m in comparator.methods if not m.supports_gradients]
        if non_diff:
            print(f"Note: {', '.join(non_diff)} are not differentiable")
    else:
        print("No methods support gradients")

    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
