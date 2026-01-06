"""Box decoding utilities for PP-YOLOE-R rotated object detection.

Provides decoding functions for converting raw predictions to rotated bounding boxes, with support for both standard
regression and Distribution Focal Loss (DFL) modes.

All decoding functions return boxes in PIXEL SPACE for simplicity.
Stride normalization is handled at the loss level when needed.
"""

import math

import torch
import torch.nn.functional as F


def decode_ppyoloer_boxes(
    anchor_points: torch.Tensor,
    reg_dist: torch.Tensor,
    raw_angles: torch.Tensor,
    stride_tensor: torch.Tensor,
    angle_proj: torch.Tensor,
    use_dfl: bool = False,
    use_angle_bins: bool = True,
) -> torch.Tensor:
    """Decode PP-YOLOE-R predictions to rotated boxes.

    Supports two parameterizations:
    - Standard (use_dfl=False): xy_offset + wh_scale with ELU activation
    - DFL (use_dfl=True): LTRB distances from anchor points

    Args:
        anchor_points: Anchor coordinates [1, N, 2] in pixels
        reg_dist: Distance predictions [B, N, 4] (stride-normalized in DFL mode)
        raw_angles: Angle logits [B, N, angle_bins+1] or [B, N, 1]
        stride_tensor: Stride values [1, N, 1]
        angle_proj: Angle projection weights [angle_bins+1] or dummy [1]
        use_dfl: Whether DFL mode is enabled
        use_angle_bins: Whether angle is binned (True) or continuous (False)

    Returns:
        Rotated boxes [B, N, 5] as [cx, cy, w, h, angle] in PIXEL SPACE
    """
    if use_dfl:
        return _decode_ltrb(anchor_points, reg_dist, raw_angles, stride_tensor, angle_proj, use_angle_bins)
    return _decode_offset_scale(anchor_points, reg_dist, raw_angles, stride_tensor, angle_proj, use_angle_bins)


def _decode_offset_scale(
    anchor_points: torch.Tensor,
    reg_dist: torch.Tensor,
    raw_angles: torch.Tensor,
    stride_tensor: torch.Tensor,
    angle_proj: torch.Tensor,
    use_angle_bins: bool,
) -> torch.Tensor:
    """Decode using xy_offset + wh_scale parameterization (standard mode).

    Args:
        anchor_points: Anchor coordinates [1, N, 2] in pixels
        reg_dist: [B, N, 4] as [xy_offset, wh_scale]
        raw_angles: Angle logits [B, N, angle_bins+1] or [B, N, 1]
        stride_tensor: Stride values [1, N, 1]
        angle_proj: Angle projection weights [angle_bins+1] or dummy [1]
        use_angle_bins: Whether angle is binned or continuous

    Returns:
        Rotated boxes [B, N, 5] in PIXEL SPACE
    """
    xy_offset, wh_scale = reg_dist.split(2, dim=-1)

    # Decode center coordinates and dimensions
    centers = anchor_points + xy_offset * stride_tensor  # [B, N, 2]
    sizes = (F.elu(wh_scale) + 1.0) * stride_tensor  # [B, N, 2]

    # Decode angles
    if use_angle_bins:
        # Binned angle: weighted sum of angle bins
        angle_probs = F.softmax(raw_angles, dim=-1)  # [B, N, angle_bins+1]
        angles = torch.sum(angle_probs * angle_proj.view(1, 1, -1), dim=-1, keepdim=True)  # [B, N, 1]
    else:
        # Single continuous angle: sigmoid to [0, π/2]
        angles = raw_angles.sigmoid() * (math.pi / 2)  # [B, N, 1]

    return torch.cat([centers, sizes, angles], dim=-1)  # [B, N, 5]


def _decode_ltrb(
    anchor_points: torch.Tensor,
    reg_dist: torch.Tensor,
    raw_angles: torch.Tensor,
    stride_tensor: torch.Tensor,
    angle_proj: torch.Tensor,
    use_angle_bins: bool,
) -> torch.Tensor:
    """Decode using LTRB (left-top-right-bottom) parameterization (DFL mode).

    Args:
        anchor_points: Anchor coordinates [1, N, 2] in pixels
        reg_dist: [B, N, 4] as [left, top, right, bottom] distances (stride-normalized)
        raw_angles: Angle logits [B, N, angle_bins+1] or [B, N, 1]
        stride_tensor: Stride values [1, N, 1]
        angle_proj: Angle projection weights [angle_bins+1] or dummy [1]
        use_angle_bins: Whether angle is binned or continuous

    Returns:
        Rotated boxes [B, N, 5] in PIXEL SPACE
    """
    lt, rb = reg_dist.split(2, dim=-1)  # [B, N, 2] each - stride-normalized

    # Convert stride-normalized LTRB to pixel-space corners
    x1y1 = anchor_points - lt * stride_tensor  # [B, N, 2]
    x2y2 = anchor_points + rb * stride_tensor  # [B, N, 2]

    # Convert corners to center and size (pixel space)
    centers = (x1y1 + x2y2) / 2  # [B, N, 2]
    sizes = x2y2 - x1y1  # [B, N, 2]

    # Decode angles
    if use_angle_bins:
        # Binned angle: weighted sum of angle bins
        angle_probs = F.softmax(raw_angles, dim=-1)  # [B, N, angle_bins+1]
        angles = torch.sum(angle_probs * angle_proj.view(1, 1, -1), dim=-1, keepdim=True)  # [B, N, 1]
    else:
        # Single continuous angle: sigmoid to [0, π/2]
        angles = raw_angles.sigmoid() * (math.pi / 2)  # [B, N, 1]

    return torch.cat([centers, sizes, angles], dim=-1)  # [B, N, 5]


def apply_dfl(reg_dist: torch.Tensor, dfl_proj: torch.Tensor) -> torch.Tensor:
    """Apply Distribution Focal Loss integral operation.

    Converts distribution logits to continuous values via softmax and weighted sum, computing the expected value of the
    predicted distribution.

    Args:
        reg_dist: Distribution logits [B, N, 4*reg_max]
        dfl_proj: Projection weights [reg_max], typically [0, 1, 2, ..., reg_max-1]

    Returns:
        Continuous regression values [B, N, 4] (stride-normalized distances)
    """
    batch_size, num_anchors = reg_dist.shape[:2]
    reg_max = dfl_proj.shape[0]

    # Reshape to [B, N, 4, reg_max]
    reg_dist = reg_dist.view(batch_size, num_anchors, 4, reg_max)

    # Apply softmax and weighted sum
    reg_probs = F.softmax(reg_dist, dim=-1)  # [B, N, 4, reg_max]
    reg_dist = torch.sum(reg_probs * dfl_proj.view(1, 1, 1, -1), dim=-1)  # [B, N, 4]

    return reg_dist
