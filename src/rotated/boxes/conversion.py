import numpy as np
import torch


def obb_to_corners_format(obb_tensor: torch.Tensor, degrees: bool = True) -> torch.Tensor:
    """Convert 5-coordinate format to 8 corner coordinates.

    Transforms oriented bounding box representation to 4 corner points as [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    Works with any batch dimensions.

    Args:
        obb_tensor: Tensor of shape (..., 5) containing
                   [center_x, center_y, width, height, angle_deg].
        degrees: if True, convert angle from degrees to radians

    Returns:
        Tensor of shape (..., 4, 2) containing corner coordinates.
        Ordered as: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] for
        bottom-left, bottom-right, top-right, top-left corners.
    """
    # Extract components
    center_x = obb_tensor[..., 0]
    center_y = obb_tensor[..., 1]
    width = obb_tensor[..., 2]
    height = obb_tensor[..., 3]
    angle = obb_tensor[..., 4]

    # Convert angle to radians
    if degrees:
        angle = torch.deg2rad(angle)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

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
    center = torch.stack([center_x, center_y], dim=-1).unsqueeze(-2)  # (..., 1, 2)
    return corners_rotated + center


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
    pi = torch.tensor(np.pi, dtype=angle_rad.dtype, device=angle_rad.device)
    angle_rad = torch.remainder(angle_rad + pi / 2, pi) - pi / 2

    # Stack results
    result = torch.stack([center_x, center_y, width, height, angle_rad], dim=-1)

    return result
