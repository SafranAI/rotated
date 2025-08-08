import torch
import torch.nn.functional as F


def decode_ppyoloe_r_boxes(
    anchor_points: torch.Tensor,
    reg_dist: torch.Tensor,
    raw_angles: torch.Tensor,
    stride_tensor: torch.Tensor,
    angle_proj: torch.Tensor,
) -> torch.Tensor:
    """Decode PP-YOLOE-R predictions to rotated boxes.

    Args:
        anchor_points: Anchor point coordinates [1, N, 2]
        reg_dist: Distance predictions [B, N, 4] - [xy_offset, wh_scale]
        raw_angles: Raw angle logits [B, N, angle_bins+1]
        stride_tensor: Stride values [1, N, 1]
        angle_proj: Angle projection weights [angle_bins+1]

    Returns:
        Decoded rotated boxes [B, N, 5] - [cx, cy, w, h, angle] in absolute pixels
    """
    xy_offset, wh_scale = reg_dist.split(2, dim=-1)

    # Decode center coordinates and dimensions
    centers = anchor_points + xy_offset * stride_tensor  # [B, N, 2]
    sizes = (F.elu(wh_scale) + 1.0) * stride_tensor  # [B, N, 2]

    # Decode angles using weighted sum of angle bins
    angle_probs = F.softmax(raw_angles, dim=-1)  # [B, N, angle_bins+1]
    angles = torch.sum(angle_probs * angle_proj.view(1, 1, -1), dim=-1, keepdim=True)  # [B, N, 1]

    return torch.cat([centers, sizes, angles], dim=-1)  # [B, N, 5]
