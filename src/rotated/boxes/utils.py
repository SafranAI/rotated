import torch


def check_points_in_rotated_boxes(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Check if points are contained within rotated bounding boxes.

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


def check_points_in_horizontal_boxes(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Check if points are contained within horizontal bounding boxes.

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
