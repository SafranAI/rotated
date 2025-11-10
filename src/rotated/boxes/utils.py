import torch


def check_aabb_overlap(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Check for axis-aligned bounding box overlap.

    Args:
        boxes1: [N, 5] first set of boxes
        boxes2: [N, 5] second set of boxes

    Returns:
        [N] boolean mask indicating which box pairs might overlap
    """
    bounds1 = compute_aabb_bounds(boxes1)
    bounds2 = compute_aabb_bounds(boxes2)

    no_overlap_x = (bounds1[:, 1] < bounds2[:, 0]) | (bounds2[:, 1] < bounds1[:, 0])
    no_overlap_y = (bounds1[:, 3] < bounds2[:, 2]) | (bounds2[:, 3] < bounds1[:, 2])

    return ~(no_overlap_x | no_overlap_y)


def compute_aabb_bounds(boxes: torch.Tensor) -> torch.Tensor:
    """Compute axis-aligned bounding box bounds for rotated boxes.

    Args:
        boxes: [N, 5] (x, y, w, h, angle)

    Returns:
        [N, 4] bounds (min_x, max_x, min_y, max_y)
    """
    x, y, w, h, angle = boxes.unbind(-1)
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)

    # Half extents after rotation
    ext_x = 0.5 * (w * torch.abs(cos_a) + h * torch.abs(sin_a))
    ext_y = 0.5 * (w * torch.abs(sin_a) + h * torch.abs(cos_a))

    min_x = x - ext_x
    max_x = x + ext_x
    min_y = y - ext_y
    max_y = y + ext_y

    return torch.stack([min_x, max_x, min_y, max_y], dim=-1)


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
