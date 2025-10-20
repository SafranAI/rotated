import numpy as np
import torch


def standard_to_opencv_format(
    center_x: float, center_y: float, width: float, height: float, angle_rad: float
) -> tuple[float, float, float, float, float]:
    """Convert standard OBB format to OpenCV format using OpenCV's minAreaRect.

    Uses OpenCV's actual minAreaRect function to ensure exact compatibility
    with OpenCV's angle computation behavior. This is the most reliable approach
    as it matches OpenCV's internal algorithm exactly.

    Args:
        center_x: X coordinate of rectangle center.
        center_y: Y coordinate of rectangle center.
        width: Width of the rectangle.
        height: Height of the rectangle.
        angle_rad: Rotation angle in radians, range [-π/2, π/2].

    Returns:
        Tuple containing (center_x, center_y, opencv_width, opencv_height, opencv_angle_deg)
        where opencv_angle_deg is in range [0, 90) degrees.
    """
    import math

    import cv2

    # Convert OBB to corner points
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Half dimensions
    w2, h2 = width / 2, height / 2

    # Corner points relative to center (before rotation)
    corners_local = np.array(
        [
            [-w2, -h2],  # bottom-left
            [w2, -h2],  # bottom-right
            [w2, h2],  # top-right
            [-w2, h2],  # top-left
        ]
    )

    # Rotation matrix
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Rotate and translate corners
    corners_rotated = np.dot(corners_local, rotation_matrix.T)
    corners_global = corners_rotated + np.array([center_x, center_y])

    # Use OpenCV's minAreaRect to get the exact result
    opencv_result = cv2.minAreaRect(corners_global.astype(np.float32))
    opencv_center, opencv_size, opencv_angle = opencv_result

    return opencv_center[0], opencv_center[1], opencv_size[0], opencv_size[1], opencv_angle


def opencv_format_to_corners(obb_tensor: torch.Tensor) -> torch.Tensor:
    """Convert 5-coordinate format to 8 corner coordinates.

    Transforms oriented bounding box representation to 4 corner points as
    8 flattened coordinates [x1, y1, x2, y2, x3, y3, x4, y4].
    Works with any batch dimensions.

    Args:
        obb_tensor: Tensor of shape (..., 5) containing
                   [center_x, center_y, width, height, angle_deg].

    Returns:
        Tensor of shape (..., 8) containing corner coordinates.
        Ordered as: [x1, y1, x2, y2, x3, y3, x4, y4] for
        bottom-left, bottom-right, top-right, top-left corners.
    """
    # Extract components
    center_xy = obb_tensor[..., :2].unsqueeze(-2)
    width = obb_tensor[..., 2]
    height = obb_tensor[..., 3]
    angle_deg = obb_tensor[..., 4]

    # Convert angle to radians
    angle_rad = torch.deg2rad(angle_deg)
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    # Half dimensions
    w2 = width / 2
    h2 = height / 2

    # Corner points relative to center (before rotation)
    # Shape: (..., 4, 2)
    corners_local = torch.stack(
        [
            torch.stack([-w2, -h2], dim=-1),  # bottom-left
            torch.stack([w2, -h2], dim=-1),  # bottom-right
            torch.stack([w2, h2], dim=-1),  # top-right
            torch.stack([-w2, h2], dim=-1),  # top-left
        ],
        dim=-2,
    )

    # Rotation matrix: (..., 2, 2)
    rotation_matrix = torch.stack([torch.stack([cos_a, -sin_a], dim=-1), torch.stack([sin_a, cos_a], dim=-1)], dim=-2)

    # Rotate corners: (..., 4, 2) @ (..., 2, 2) -> (..., 4, 2)
    corners_rotated = torch.matmul(corners_local, rotation_matrix.transpose(-2, -1))

    # Translate to final position
    corners_global = corners_rotated + center_xy

    # Flatten to 8 coordinates: (..., 4, 2) -> (..., 8)
    corners_flattened = corners_global.reshape(*corners_global.shape[:-2], 8)

    return corners_flattened


def opencv_to_standard_format(obb_tensor: torch.Tensor) -> torch.Tensor:
    """Convert OpenCV format back to standard OBB format.

    Transforms from OpenCV minAreaRect format with angle in [0°, 90°) degrees
    back to standard oriented bounding box format with angle in [-π/2, π/2] radians.
    Works with any batch dimensions.

    Args:
        obb_tensor: Tensor of shape (..., 5) containing
                   [center_x, center_y, width, height, opencv_angle_deg].

    Returns:
        Tensor of shape (..., 5) containing
        [center_x, center_y, width, height, angle_rad]
        where angle_rad is in range [-π/2, π/2] radians.
    """
    width = obb_tensor[..., 2]
    height = obb_tensor[..., 3]
    opencv_angle_deg = obb_tensor[..., 4]

    # Determine if dimension swap occurred: if width < height
    width_lt_height = width < height

    # Compute standard angle
    standard_angle_deg = torch.where(
        width_lt_height,
        opencv_angle_deg - 90,  # Subtract 90° if dimensions were swapped
        opencv_angle_deg,  # Use as-is if no swap
    )

    # Convert to radians
    obb_tensor[..., 4] = torch.deg2rad(standard_angle_deg)

    return obb_tensor


def corners_to_standard_format(corners_tensor: torch.Tensor) -> torch.Tensor:
    """Convert 8 corner coordinates back to standard OBB format.

    Transforms 8 flattened corner coordinates back to 5-coordinate oriented bounding box
    representation with angle in [-π/2, π/2] radians. Works with any batch dimensions.

    Args:
        corners_tensor: Tensor of shape (..., 8) containing corner coordinates as
                       [x1, y1, x2, y2, x3, y3, x4, y4].

    Returns:
        Tensor of shape (..., 5) containing
        [center_x, center_y, width, height, angle_rad]
        where angle_rad is in range [-π/2, π/2] radians.
    """
    # Reshape from 8 coordinates to (4, 2): (..., 8) -> (..., 4, 2)
    corners_reshaped = corners_tensor.reshape(*corners_tensor.shape[:-1], 4, 2)

    # Compute center
    center = torch.mean(corners_reshaped, dim=-2)  # (..., 2)
    center_x = center[..., 0]
    center_y = center[..., 1]

    # Compute edge vectors
    edge1 = corners_reshaped[..., 1, :] - corners_reshaped[..., 0, :]  # bottom edge
    edge2 = corners_reshaped[..., 3, :] - corners_reshaped[..., 0, :]  # left edge

    # Compute dimensions
    width = torch.norm(edge1, dim=-1)
    height = torch.norm(edge2, dim=-1)

    # Compute angle from the first edge (bottom edge)
    angle_rad = torch.atan2(edge1[..., 1], edge1[..., 0])

    # Normalize angle to [-π/2, π/2] range
    # Use modular arithmetic to handle the wrapping
    angle_rad = torch.remainder(angle_rad + torch.pi / 2, torch.pi) - torch.pi / 2

    # Stack results
    result = torch.stack([center_x, center_y, width, height, angle_rad], dim=-1)

    return result
