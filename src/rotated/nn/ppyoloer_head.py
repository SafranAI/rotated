# Modified from PaddleDetection (https://github.com/PaddlePaddle/PaddleDetection)
# Copyright (c) 2024 PaddlePaddle Authors. Apache 2.0 License.


from collections.abc import Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rotated.boxes.decode import apply_dfl, decode_ppyoloer_boxes
from rotated.losses.ppyoloer_criterion import LossComponents
from rotated.nn.common import ActivationType, ConvBNLayer
from rotated.utils.export import compile_compatible_lru_cache


class ESEAttn(nn.Module):
    """Efficient Squeeze-and-Excitation Attention."""

    def __init__(self, feat_channels: int, act: ActivationType = "swish"):
        super().__init__()

        if feat_channels <= 0:
            raise ValueError(f"Feature channels must be positive, got {feat_channels}")

        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc.weight, std=0.01)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, feat: torch.Tensor, avg_feat: torch.Tensor) -> torch.Tensor:
        weight = torch.sigmoid(self.fc(avg_feat))
        return self.conv(feat * weight)


class PPYOLOERHead(nn.Module):
    """PP-YOLOE-R detection head for rotated object detection.

    Supports two prediction modes:
    - Standard (use_dfl=False): xy_offset + wh_scale regression
    - DFL (use_dfl=True): LTRB distances with distribution modeling

    Supports two angle modes:
    - Binned (use_angle_bins=True): angle_max+1 outputs with softmax
    - Continuous (use_angle_bins=False): single output with sigmoid to [0, π/2]

    Args:
        in_channels: Input channels for each FPN level
        num_classes: Number of object classes
        act: Activation function name
        fpn_strides: Strides for each FPN level
        grid_cell_offset: Grid cell offset for anchor generation
        angle_max: Number of angle bins
        use_dfl: Enable Distribution Focal Loss
        reg_max: Number of DFL bins
        use_angle_bins: Enable binned angle prediction
        criterion: Loss criterion module

    Raises:
        ValueError: If any input validation fails
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (192, 384, 768),
        num_classes: int = 15,
        act: ActivationType = "swish",
        fpn_strides: Sequence[int] = (8, 16, 32),
        grid_cell_offset: float = 0.5,
        angle_max: int = 90,
        use_dfl: bool = False,
        reg_max: int = 16,
        use_angle_bins: bool = True,
        criterion: nn.Module = None,
    ):
        super().__init__()

        # Validation
        if len(in_channels) == 0:
            raise ValueError("in_channels should not be empty")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if angle_max <= 0:
            raise ValueError(f"angle_max must be positive, got {angle_max}")
        if reg_max <= 0:
            raise ValueError(f"reg_max must be positive, got {reg_max}")
        if len(fpn_strides) != len(in_channels):
            raise ValueError(f"fpn_strides length {len(fpn_strides)} must match in_channels length {len(in_channels)}")
        if any(ch <= 0 for ch in in_channels):
            raise ValueError(f"All channel counts must be positive, got {in_channels}")

        self.in_channels = list(in_channels)
        self.fpn_strides = list(fpn_strides)
        self.num_classes = num_classes
        self.grid_cell_offset = grid_cell_offset
        self.angle_max = angle_max
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.use_angle_bins = use_angle_bins
        self.criterion = criterion

        # Angle-related constants
        self.register_buffer("half_pi", torch.tensor(math.pi / 2, dtype=torch.float32))
        self.register_buffer("half_pi_bin", torch.tensor(math.pi / 2 / angle_max, dtype=torch.float32))

        # Angle projection weights
        if use_angle_bins:
            angle_proj = torch.linspace(0, self.angle_max, self.angle_max + 1) * (math.pi / 2 / angle_max)
            self.register_buffer("angle_proj", angle_proj)
            angle_out_channels = self.angle_max + 1
        else:
            self.register_buffer("angle_proj", torch.zeros(1))
            angle_out_channels = 1

        # DFL projection weights
        if use_dfl:
            dfl_proj = torch.arange(0, reg_max, dtype=torch.float32)
            self.register_buffer("dfl_proj", dfl_proj)
        else:
            self.register_buffer("dfl_proj", torch.zeros(1))

        # ESE attention modules
        self.stem_cls = nn.ModuleList()
        self.stem_reg = nn.ModuleList()
        self.stem_angle = nn.ModuleList()

        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, act=act))
            self.stem_reg.append(ESEAttn(in_c, act=act))
            self.stem_angle.append(ESEAttn(in_c, act=act))

        # Prediction heads
        self.pred_cls = nn.ModuleList()
        self.pred_reg = nn.ModuleList()
        self.pred_angle = nn.ModuleList()

        reg_out_channels = 4 * self.reg_max if self.use_dfl else 4

        for in_c in self.in_channels:
            self.pred_cls.append(nn.Conv2d(in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(nn.Conv2d(in_c, reg_out_channels, 3, padding=1))
            self.pred_angle.append(nn.Conv2d(in_c, angle_out_channels, 3, padding=1))

        self._init_weights()

        if criterion is not None:
            self._sync_criterion_config()

    def _sync_criterion_config(self) -> None:
        """Sync configuration to criterion."""
        if self.criterion is not None and hasattr(self.criterion, "_sync_config"):
            self.criterion._sync_config(self.use_dfl, self.reg_max, self.use_angle_bins)

    def _init_weights(self):
        """Initialize weights."""
        bias_cls = -math.log((1 - 0.01) / 0.01)

        for cls_head, reg_head, angle_head in zip(self.pred_cls, self.pred_reg, self.pred_angle, strict=True):
            # Classification head
            nn.init.normal_(cls_head.weight, std=0.01)
            nn.init.constant_(cls_head.bias, bias_cls)

            # Regression head
            nn.init.normal_(reg_head.weight, std=0.01)
            nn.init.constant_(reg_head.bias, 1.0 if self.use_dfl else 0.0)

            # Angle head
            nn.init.normal_(angle_head.weight, std=0.01)

            if self.use_angle_bins:
                # Initialize biases to zero to start with no angle prior
                # Network will learn the angle distribution from data
                nn.init.constant_(angle_head.bias, 0.0)
            else:
                # For continuous angle (sigmoid output), bias toward middle of range
                # sigmoid(0) = 0.5 → angle = 0.5 * π/2 = π/4 (45 degrees)
                nn.init.constant_(angle_head.bias, 0.0)

    @compile_compatible_lru_cache(maxsize=8)
    def _generate_anchors_cached(
        self, shapes: tuple[tuple[int, int], ...], device_str: str, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Generate anchors with caching for efficiency."""
        device = torch.device(device_str)

        anchor_points = []
        stride_tensor = []
        num_anchors_list = []

        for (height, width), stride in zip(shapes, self.fpn_strides, strict=True):
            shift_x = torch.arange(width, device=device, dtype=dtype)
            shift_y = torch.arange(height, device=device, dtype=dtype)

            shift_x = (shift_x + self.grid_cell_offset) * stride
            shift_y = (shift_y + self.grid_cell_offset) * stride

            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            anchor_point = torch.stack([shift_x, shift_y], dim=-1).view(1, -1, 2)
            anchor_points.append(anchor_point)

            stride_tensor.append(torch.full((1, height * width, 1), stride, device=device, dtype=dtype))
            num_anchors_list.append(height * width)

        anchor_points = torch.cat(anchor_points, dim=1)
        stride_tensor = torch.cat(stride_tensor, dim=1)

        return anchor_points, stride_tensor, num_anchors_list

    def _generate_anchors(self, feats: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Generate anchors from feature maps."""
        shapes_list = []
        for feat in feats:
            shapes_list.append((int(feat.shape[2]), int(feat.shape[3])))
        shapes = tuple(shapes_list)

        device_str = str(feats[0].device)
        dtype = feats[0].dtype

        return self._generate_anchors_cached(shapes, device_str, dtype)

    def _forward_common(self, feats: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate raw predictions from features."""
        cls_logit_list = []
        reg_dist_list = []
        reg_angle_list = []

        reg_channels = 4 * self.reg_max if self.use_dfl else 4
        angle_channels = self.angle_max + 1 if self.use_angle_bins else 1

        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))

            cls_enhanced = self.stem_cls[i](feat, avg_feat)
            reg_enhanced = self.stem_reg[i](feat, avg_feat)
            angle_enhanced = self.stem_angle[i](feat, avg_feat)

            cls_logit = self.pred_cls[i](cls_enhanced + feat)
            reg_dist = self.pred_reg[i](reg_enhanced)
            reg_angle = self.pred_angle[i](angle_enhanced)

            cls_logit_list.append(cls_logit.view(feat.size(0), self.num_classes, -1).transpose(1, 2))
            reg_dist_list.append(reg_dist.view(feat.size(0), reg_channels, -1).transpose(1, 2))
            reg_angle_list.append(reg_angle.view(feat.size(0), angle_channels, -1).transpose(1, 2))

        cls_logits = torch.cat(cls_logit_list, dim=1)
        reg_dist = torch.cat(reg_dist_list, dim=1)
        raw_angles = torch.cat(reg_angle_list, dim=1)

        return cls_logits, reg_dist, raw_angles

    def forward(
        self, feats: Sequence[torch.Tensor], targets: dict[str, torch.Tensor] = None
    ) -> tuple[LossComponents, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            feats: Feature maps from neck
            targets: Optional training targets with 'labels', 'boxes', 'valid_mask'

        Returns:
            Tuple of (losses, cls_scores, decoded_boxes):
                - losses: Loss components (zeros during inference)
                - cls_scores: Classification scores [B, N, C]
                - decoded_boxes: Decoded rotated boxes [B, N, 5] in pixels

        Raises:
            ValueError: If feature map count doesn't match stride count
        """
        if len(feats) != len(self.fpn_strides):
            raise ValueError(f"feats length {len(feats)} must equal fpn_strides length {len(self.fpn_strides)}")

        cls_logits, reg_dist, raw_angles = self._forward_common(feats)
        anchor_points, stride_tensor, _ = self._generate_anchors(feats)

        # Apply DFL to convert distributions to continuous distances
        reg_dist_continuous = apply_dfl(reg_dist, self.dfl_proj) if self.use_dfl else reg_dist

        # Decode boxes in pixel space
        decoded_boxes = decode_ppyoloer_boxes(
            anchor_points,
            reg_dist=reg_dist_continuous,
            raw_angles=raw_angles,
            stride_tensor=stride_tensor,
            angle_proj=self.angle_proj,
            use_dfl=self.use_dfl,
            use_angle_bins=self.use_angle_bins,
        )

        cls_scores = torch.sigmoid(cls_logits)

        # Compute losses if targets provided
        if targets is not None:
            if self.criterion is None:
                raise ValueError("Criterion must be set to compute losses")

            losses = self.criterion(
                cls_logits=cls_logits,
                reg_dist=reg_dist,
                raw_angles=raw_angles,
                decoded_boxes=decoded_boxes,
                targets=targets,
                anchor_points=anchor_points,
                stride_tensor=stride_tensor,
            )
        else:
            losses = LossComponents.empty(cls_logits.device)

        return losses, cls_scores, decoded_boxes
