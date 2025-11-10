# Modified from PaddleDetection (https://github.com/PaddlePaddle/PaddleDetection)
# Copyright (c) 2024 PaddlePaddle Authors. Apache 2.0 License.

from collections.abc import Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rotated.boxes.decode import decode_ppyoloer_boxes
from rotated.nn.common import ConvBNLayer


class ESEAttn(nn.Module):
    """Efficient Squeeze-and-Excitation Attention."""

    def __init__(self, feat_channels: int, act: str = "swish"):
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
    """PP-YOLOE-R Detection Head with unified forward logic.

    Head returns raw predictions for training and processed predictions for inference.
    Training mode: Returns (losses, cls_logits, reg_dist)
    Inference mode: Returns (losses, cls_scores, decoded_boxes)
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (192, 384, 768),  # P3', P4', P5' (shallow -> deep)
        num_classes: int = 15,
        act: str = "swish",
        fpn_strides: Sequence[int] = (8, 16, 32),  # P3, P4, P5 (shallow -> deep)
        grid_cell_offset: float = 0.5,
        angle_max: int = 90,
        cache_anchors: bool = True,
        criterion: nn.Module | None = None,
    ):
        super().__init__()

        # Validation
        if len(in_channels) == 0:
            raise ValueError("in_channels should not be empty")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if angle_max <= 0:
            raise ValueError(f"angle_max must be positive, got {angle_max}")
        if len(fpn_strides) != len(in_channels):
            raise ValueError(f"fpn_strides length {len(fpn_strides)} must match in_channels length {len(in_channels)}")
        if any(ch <= 0 for ch in in_channels):
            raise ValueError(f"All channel counts must be positive, got {in_channels}")

        # Convert to lists for TorchScript compatibility
        self.in_channels = list(in_channels)
        self.fpn_strides = list(fpn_strides)
        self.num_classes = num_classes
        self.grid_cell_offset = grid_cell_offset
        self.angle_max = angle_max
        self.cache_anchors = cache_anchors
        self.criterion = criterion

        # Angle-related constants
        self.register_buffer("half_pi", torch.tensor(math.pi / 2, dtype=torch.float32))
        self.register_buffer("half_pi_bin", torch.tensor(math.pi / 2 / angle_max, dtype=torch.float32))

        # Pre-compute angle projection weights
        angle_proj = torch.linspace(0, self.angle_max, self.angle_max + 1) * (math.pi / 2 / angle_max)
        self.register_buffer("angle_proj", angle_proj)

        # Build ESE attention modules for each task
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

        for in_c in self.in_channels:
            self.pred_cls.append(nn.Conv2d(in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(nn.Conv2d(in_c, 4, 3, padding=1))
            self.pred_angle.append(nn.Conv2d(in_c, self.angle_max + 1, 3, padding=1))

        # Cache for anchor points and metadata
        self._anchor_cache = {}

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following PP-YOLOE-R convention."""
        bias_cls = -math.log((1 - 0.01) / 0.01)

        # Angle bias: [10.0, 1.0, 1.0, ..., 1.0]
        bias_angle = [10.0] + [1.0] * self.angle_max

        for cls_head, reg_head, angle_head in zip(self.pred_cls, self.pred_reg, self.pred_angle, strict=True):
            # Classification head
            nn.init.normal_(cls_head.weight, std=0.01)
            nn.init.constant_(cls_head.bias, bias_cls)

            # Regression head
            nn.init.normal_(reg_head.weight, std=0.01)
            nn.init.constant_(reg_head.bias, 0)

            # Angle head
            nn.init.constant_(angle_head.weight, 0)
            angle_head.bias.data = torch.tensor(bias_angle, dtype=torch.float32)

    def _generate_anchors(self, feats: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Generate anchor points with optional caching for efficiency.

        Args:
            feats: Feature maps [P3', P4', P5'] (shallow â†’ deep order)

        Returns:
            anchor_points: Anchor point coordinates [1, N, 2]
            stride_tensor: Stride values for each anchor [1, N, 1]
            num_anchors_list: Number of anchors per FPN level
        """
        # Create cache key based on feature sizes and device
        cache_key = tuple((f.shape[2], f.shape[3], str(f.device)) for f in feats)

        if self.cache_anchors and cache_key in self._anchor_cache:
            return self._anchor_cache[cache_key]

        anchor_points = []
        stride_tensor = []
        num_anchors_list = []

        # Process in input order: P3', P4', P5' with strides 8, 16, 32
        for feat, stride in zip(feats, self.fpn_strides, strict=True):
            batch_size, channels, height, width = feat.shape

            # Generate grid coordinates
            device = feat.device
            shift_x = torch.arange(width, device=device, dtype=torch.float32)
            shift_y = torch.arange(height, device=device, dtype=torch.float32)

            shift_x = (shift_x + self.grid_cell_offset) * stride
            shift_y = (shift_y + self.grid_cell_offset) * stride

            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            anchor_point = torch.stack([shift_x, shift_y], dim=-1).view(1, -1, 2)
            anchor_points.append(anchor_point)

            stride_tensor.append(torch.full((1, height * width, 1), stride, device=device, dtype=torch.float32))
            num_anchors_list.append(height * width)

        anchor_points = torch.cat(anchor_points, dim=1)
        stride_tensor = torch.cat(stride_tensor, dim=1)

        result = (anchor_points, stride_tensor, num_anchors_list)

        # Cache result if enabled
        if self.cache_anchors:
            self._anchor_cache[cache_key] = result

        return result

    def _forward_common(self, feats: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unified forward logic generating only raw predictions.

        Processes features through ESE attention and prediction heads,
        returning raw predictions without any post-processing.

        Args:
            feats: Feature maps [P3', P4', P5'] from neck

        Returns:
            cls_logits: [B, N, C] - Raw classification logits (pre-sigmoid)
            reg_dist: [B, N, 4] - Raw distance predictions [xy_offset, wh_scale]
            raw_angles: [B, N, angle_max+1] - Raw angle logits (pre-softmax)
        """
        cls_logit_list = []
        reg_dist_list = []
        reg_angle_list = []

        # Process each FPN level
        for i, feat in enumerate(feats):
            # Global average pooling for ESE attention
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))

            # Get enhanced features for each task
            cls_enhanced = self.stem_cls[i](feat, avg_feat)
            reg_enhanced = self.stem_reg[i](feat, avg_feat)
            angle_enhanced = self.stem_angle[i](feat, avg_feat)

            # Generate predictions
            cls_logit = self.pred_cls[i](cls_enhanced + feat)  # Residual connection
            reg_dist = self.pred_reg[i](reg_enhanced)
            reg_angle = self.pred_angle[i](angle_enhanced)

            # Flatten and collect predictions (keep logits raw)
            cls_logit_list.append(cls_logit.view(feat.size(0), self.num_classes, -1).transpose(1, 2))
            reg_dist_list.append(reg_dist.view(feat.size(0), 4, -1).transpose(1, 2))
            reg_angle_list.append(reg_angle.view(feat.size(0), self.angle_max + 1, -1).transpose(1, 2))

        # Concatenate all FPN levels
        cls_logits = torch.cat(cls_logit_list, dim=1)  # [B, N, C]
        reg_dist = torch.cat(reg_dist_list, dim=1)  # [B, N, 4]
        raw_angles = torch.cat(reg_angle_list, dim=1)  # [B, N, angle_max+1]

        return cls_logits, reg_dist, raw_angles

    def forward(
        self, feats: Sequence[torch.Tensor], targets: dict[str, torch.Tensor] = None
    ) -> tuple[dict[str, torch.Tensor] | None, torch.Tensor, torch.Tensor]:
        """Unified forward pass with consistent API.

        Always returns decoded boxes and classification scores regardless of mode.
        Uses raw predictions internally for criterion interface.

        Args:
            feats: Feature maps [P3', P4', P5'] from neck (shallow -> deep)
            targets: Training targets (optional):
                - labels: [B, M, 1] - Class labels
                - boxes: [B, M, 5] - Rotated boxes [cx, cy, w, h, angle]
                    * cx, cy, w, h: in absolute pixels
                    * angle: in radians, should be in range [0, π/2)
                - valid_mask: [B, M, 1] - Valid target mask

        Returns:
            Tuple of (losses, cls_scores, decoded_boxes):
                - losses: Loss dictionary if targets provided, None otherwise
                - cls_scores: [B, N, C] - Classification scores (post-sigmoid)
                - decoded_boxes: [B, N, 5] - Decoded rotated boxes in absolute pixels, angle in [0, π/2)

        Raises:
            ValueError: If feats length doesn't match fpn_strides length
            ValueError: If criterion is None when targets are provided
        """
        if len(feats) != len(self.fpn_strides):
            raise ValueError(f"feats length {len(feats)} must equal fpn_strides length {len(self.fpn_strides)}")

        # Generate raw predictions
        cls_logits, reg_dist, raw_angles = self._forward_common(feats)

        # Generate anchor metadata
        anchor_points, stride_tensor, _ = self._generate_anchors(feats)

        cls_scores = torch.sigmoid(cls_logits)
        decoded_boxes = decode_ppyoloer_boxes(anchor_points, reg_dist, raw_angles, stride_tensor, self.angle_proj)

        # Compute losses if targets provided
        losses = None
        if targets is not None:
            if self.criterion is None:
                raise ValueError("Criterion must be set to compute losses. Use set_criterion() or pass it to __init__")

            losses = self.criterion(
                cls_logits, reg_dist, raw_angles, targets, anchor_points, stride_tensor, self.angle_proj
            )

        return losses, cls_scores, decoded_boxes

    def set_criterion(self, criterion: nn.Module):
        """Set criterion for training mode."""
        self.criterion = criterion

    def get_anchors(self, feats: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Get anchor points and strides for external use."""
        anchor_points, stride_tensor, _ = self._generate_anchors(feats)
        return anchor_points, stride_tensor

    def clear_cache(self):
        """Clear anchor cache to free memory."""
        self._anchor_cache.clear()
