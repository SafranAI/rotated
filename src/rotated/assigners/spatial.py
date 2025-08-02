"""Spatial containment checkers for different box formats."""

import torch


class RotatedSpatialChecker:
    """Spatial containment checker for rotated bounding boxes."""

    def __call__(self, points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Check if points are contained within rotated boxes.

        Args:
            points: Point coordinates [L, 2] - (x, y)
            boxes: Rotated boxes [N, 5] - (cx, cy, w, h, angle)

        Returns:
            Containment mask [N, L] - True where points are inside boxes
        """
        num_points = points.shape[0]
        num_boxes = boxes.shape[0]

        if num_points == 0 or num_boxes == 0:
            return torch.zeros(num_boxes, num_points, dtype=torch.bool, device=points.device)

        # Extract box parameters
        cx, cy, w, h, angle = boxes.unbind(-1)  # [N]
        px, py = points.unbind(-1)  # [L]

        # Transform points to box-local coordinates
        dx = px.unsqueeze(0) - cx.unsqueeze(1)  # [N, L]
        dy = py.unsqueeze(0) - cy.unsqueeze(1)  # [N, L]

        # Rotate points to align with box axes (inverse rotation)
        cos_a = torch.cos(-angle).unsqueeze(1)  # [N, 1]
        sin_a = torch.sin(-angle).unsqueeze(1)  # [N, 1]

        local_x = dx * cos_a - dy * sin_a  # [N, L]
        local_y = dx * sin_a + dy * cos_a  # [N, L]

        # Check if points are within box bounds
        half_w = (w * 0.5).unsqueeze(1)  # [N, 1]
        half_h = (h * 0.5).unsqueeze(1)  # [N, 1]

        inside_x = torch.abs(local_x) <= half_w
        inside_y = torch.abs(local_y) <= half_h

        return inside_x & inside_y


class HorizontalSpatialChecker:
    """Spatial containment checker for horizontal (axis-aligned) bounding boxes."""

    def __call__(self, points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Check if points are contained within horizontal boxes.

        Args:
            points: Point coordinates [L, 2] - (x, y)
            boxes: Horizontal boxes [N, 4] - (cx, cy, w, h)

        Returns:
            Containment mask [N, L] - True where points are inside boxes
        """
        num_points = points.shape[0]
        num_boxes = boxes.shape[0]

        if num_points == 0 or num_boxes == 0:
            return torch.zeros(num_boxes, num_points, dtype=torch.bool, device=points.device)

        # Extract box parameters
        cx, cy, w, h = boxes.unbind(-1)  # [N]
        px, py = points.unbind(-1)  # [L]

        # Compute box bounds
        x1 = (cx - w * 0.5).unsqueeze(1)  # [N, 1]
        y1 = (cy - h * 0.5).unsqueeze(1)  # [N, 1]
        x2 = (cx + w * 0.5).unsqueeze(1)  # [N, 1]
        y2 = (cy + h * 0.5).unsqueeze(1)  # [N, 1]

        # Check containment
        px_exp = px.unsqueeze(0)  # [1, L]
        py_exp = py.unsqueeze(0)  # [1, L]

        inside_x = (px_exp >= x1) & (px_exp <= x2)
        inside_y = (py_exp >= y1) & (py_exp <= y2)

        return inside_x & inside_y


if __name__ == "__main__":
    print("Testing spatial checkers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test points
    points = torch.tensor([[50, 50], [60, 60], [100, 100], [10, 10]], device=device, dtype=torch.float32)

    # Test rotated spatial checker
    rotated_checker = RotatedSpatialChecker()

    rotated_boxes = torch.tensor(
        [
            [50, 50, 20, 10, 0.0],  # Horizontal box at (50,50)
            [60, 60, 20, 10, 0.785],  # 45-degree rotated box at (60,60)
        ],
        device=device,
        dtype=torch.float32,
    )

    rotated_mask = rotated_checker(points, rotated_boxes)
    print(f"Rotated spatial mask:\n{rotated_mask}")

    # Test horizontal spatial checker
    horizontal_checker = HorizontalSpatialChecker()

    horizontal_boxes = torch.tensor(
        [
            [50, 50, 20, 10],  # Box at (50,50)
            [60, 60, 20, 10],  # Box at (60,60)
        ],
        device=device,
        dtype=torch.float32,
    )

    horizontal_mask = horizontal_checker(points, horizontal_boxes)
    print(f"Horizontal spatial mask:\n{horizontal_mask}")

    # Verify specific cases
    print("\nDetailed checks:")
    print(f"Point (50,50) in rotated box 0: {rotated_mask[0, 0].item()}")
    print(f"Point (60,60) in rotated box 1: {rotated_mask[1, 1].item()}")
    print(f"Point (50,50) in horizontal box 0: {horizontal_mask[0, 0].item()}")
    print(f"Point (60,60) in horizontal box 1: {horizontal_mask[1, 1].item()}")

    # Test edge cases
    empty_points = torch.zeros(0, 2, device=device)
    empty_boxes = torch.zeros(0, 4, device=device)

    empty_mask = horizontal_checker(empty_points, horizontal_boxes)
    print(f"Empty points mask shape: {empty_mask.shape}")

    empty_mask2 = horizontal_checker(points, empty_boxes)
    print(f"Empty boxes mask shape: {empty_mask2.shape}")
