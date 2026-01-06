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
    """Check if points are contained within rotated bounding boxes (cross-product mode).

    Checks EVERY point against EVERY box. Use this when points and boxes are independent.

    Args:
        points: Point coordinates [..., 2] - (x, y). Can have any leading batch dimensions.
        boxes: Rotated boxes [N, 5] - (cx, cy, w, h, angle_radians)

    Returns:
        Boolean tensor [N, ...] where result[i, ...] indicates if each point is inside box i.

    Example:
        >>> points = torch.randn(100, 2)  # 100 points
        >>> boxes = torch.randn(5, 5)     # 5 boxes
        >>> result = check_points_in_rotated_boxes(points, boxes)  # [5, 100]
        >>> # result[i, j] = True if point j is inside box i
    """
    point_shape = points.shape[:-1]
    num_boxes = boxes.shape[0]

    # Flatten all batch dimensions of points
    points_flat = points.reshape(-1, 2)  # [M, 2]
    num_points = points_flat.shape[0]

    if num_points == 0 or num_boxes == 0:
        result_shape = (num_boxes,) + point_shape
        return torch.zeros(result_shape, dtype=torch.bool, device=points.device)

    # Expand dimensions for broadcasting
    boxes_exp = boxes.unsqueeze(1)  # [N, 1, 5]
    points_exp = points_flat.unsqueeze(0)  # [1, M, 2]

    # Transform points to box-local coordinates
    local_points = points_exp - boxes_exp[:, :, :2]  # [N, M, 2]

    # Rotate points to align with box axes (inverse rotation)
    angle = boxes_exp[:, :, 4]  # [N, 1]
    cos_a = torch.cos(-angle)  # [N, 1]
    sin_a = torch.sin(-angle)  # [N, 1]

    rotated_x = local_points[:, :, 0] * cos_a - local_points[:, :, 1] * sin_a  # [N, M]
    rotated_y = local_points[:, :, 0] * sin_a + local_points[:, :, 1] * cos_a  # [N, M]

    half_w = boxes_exp[:, :, 2] * 0.5  # [N, 1]
    half_h = boxes_exp[:, :, 3] * 0.5  # [N, 1]

    inside_x = torch.abs(rotated_x) < half_w
    inside_y = torch.abs(rotated_y) < half_h

    result_flat = inside_x & inside_y  # [N, M]

    # Reshape back to match original point dimensions
    result_shape = (num_boxes,) + point_shape
    return result_flat.reshape(result_shape)


def check_points_in_rotated_boxes_paired(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Check if points are contained within rotated bounding boxes (paired/batched mode).

    Checks points[i] against box[i] only. Use this for batched operations with 1-to-1 correspondence.

    Args:
        points: Point coordinates [N, ..., 2] - (x, y). First dimension is batch size N.
        boxes: Rotated boxes [N, 5] - (cx, cy, w, h, angle_radians). Must match batch size.

    Returns:
        Boolean tensor [N, ...] where result[i, ...] indicates if points[i] are inside box[i].

    Raises:
        ValueError: If batch size (first dimension) of points and boxes don't match.

    Example:
        >>> points = torch.randn(32, 100, 2)  # 32 batches of 100 points each
        >>> boxes = torch.randn(32, 5)        # 32 boxes
        >>> result = check_points_in_rotated_boxes_paired(points, boxes)  # [32, 100]
        >>> # result[i, j] = True if points[i, j] is inside boxes[i]
    """
    N = boxes.shape[0]
    if points.shape[0] != N:
        raise ValueError(
            f"Batch size must match between points and boxes. Got points.shape[0]={points.shape[0]}, boxes.shape[0]={N}"
        )

    point_shape = points.shape[1:-1]
    points_flat = points.reshape(N, -1, 2)  # [N, M, 2]
    num_points = points_flat.shape[1]

    if num_points == 0 or N == 0:
        result_shape = (N,) + point_shape
        return torch.zeros(result_shape, dtype=torch.bool, device=points.device)

    # Transform points to box-local coordinates (paired operation)
    local_points = points_flat - boxes[:, None, :2]  # [N, M, 2]

    # Rotate points to align with box axes (inverse rotation)
    angle = boxes[:, 4]  # [N]
    cos_a = torch.cos(-angle).unsqueeze(1)  # [N, 1]
    sin_a = torch.sin(-angle).unsqueeze(1)  # [N, 1]

    rotated_x = local_points[:, :, 0] * cos_a - local_points[:, :, 1] * sin_a  # [N, M]
    rotated_y = local_points[:, :, 0] * sin_a + local_points[:, :, 1] * cos_a  # [N, M]

    half_w = (boxes[:, 2] * 0.5).unsqueeze(1)  # [N, 1]
    half_h = (boxes[:, 3] * 0.5).unsqueeze(1)  # [N, 1]

    inside_x = torch.abs(rotated_x) < half_w
    inside_y = torch.abs(rotated_y) < half_h

    result_flat = inside_x & inside_y  # [N, M]

    # Reshape back to match original point dimensions
    result_shape = (N,) + point_shape
    return result_flat.reshape(result_shape)


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

    inside_x = (px_exp > x1) & (px_exp < x2)
    inside_y = (py_exp > y1) & (py_exp < y2)

    return inside_x & inside_y


def check_is_valid_box(boxes: torch.Tensor, min_box_size: float = 1e-6) -> torch.Tensor:
    """Check if boxes have valid dimensions.

    Args:
        boxes: Boxes tensor [N, D] where D >= 4 - (..., w, h, ...)
               For rotated: (cx, cy, w, h, angle)
               For horizontal: (cx, cy, w, h)
        min_box_size: Minimum valid box dimension

    Returns:
        Boolean mask [N] indicating valid boxes
    """
    # Width is always at index 2, height at index 3
    w = boxes[:, 2]
    h = boxes[:, 3]

    # Box is valid if:
    # 1. Width and height are above minimum threshold
    # 2. No NaN values in any dimension
    # 3. No inf values in any dimension
    valid_size = (w >= min_box_size) & (h >= min_box_size)
    valid_values = ~torch.isnan(boxes).any(dim=-1) & ~torch.isinf(boxes).any(dim=-1)

    return valid_size & valid_values
