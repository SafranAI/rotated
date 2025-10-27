# Modified from PaddleDetection (https://github.com/PaddlePaddle/PaddleDetection)
# Copyright (c) 2024 PaddlePaddle Authors. Apache 2.0 License.

import torch
import torch.nn as nn

from rotated.backbones import CSPResNet
from rotated.losses.ppyoloer_criterion import RotatedDetectionLoss
from rotated.nn.custom_pan import CustomCSPPAN
from rotated.nn.ppyoloer_head import PPYOLOERHead


class PPYOLOER(nn.Module):
    """PP-YOLOE-R Architecture for Rotated Object Detection.

    Simple composition of backbone, neck, and head components.
    The head uses a consistent interface that always returns the same format:
    (losses, cls_scores, decoded_boxes) where losses is None during inference.

    Reference:
        Title: "PP-YOLOE-R: An Efficient Anchor-Free Rotated Object Detector"
        Authors: Xinxin Wang, Guanzhong Wang, Qingqing Dang, Yi Liu, Xiaoguang Hu, Dianhai Yu
        Paper link: https://arxiv.org/pdf/2211.02386
    """

    def __init__(self, backbone: nn.Module, neck: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(
        self, images: torch.Tensor, targets: dict[str, torch.Tensor] = None
    ) -> tuple[dict[str, torch.Tensor] | None, torch.Tensor, torch.Tensor]:
        """Forward pass through the complete architecture.

        Args:
            images: Input images [B, 3, H, W]
            targets: Training targets (optional):
                - labels: [B, M, 1] - Class labels [0, num_classes-1]
                - boxes: [B, M, 5] - Rotated boxes [cx, cy, w, h, angle]
                * cx, cy: Center coordinates in absolute pixels
                * w, h: Width and height in absolute pixels
                * angle: Rotation angle in radians [0, π/2)
                - valid_mask: [B, M, 1] - Valid target mask (1.0=valid, 0.0=pad)

        Returns:
            Tuple of (losses, cls_scores, decoded_boxes):
                - losses: Loss dictionary if targets provided, None otherwise
                - cls_scores: [B, N, C] - Classification scores (post-sigmoid)
                - decoded_boxes: [B, N, 5] - Decoded rotated boxes in absolute pixels
        """
        features = self.backbone(images)
        features = self.neck(features)
        return self.head(features, targets)

    def export(self) -> None:
        """Convert the model to deployment mode by reparameterizing layers if available."""
        if self.training:
            raise RuntimeError("Model must be in eval mode before export. Call model.eval() first.")

        if hasattr(self.backbone, "export"):
            self.backbone.export()


def create_ppyoloer_model(num_classes: int = 15) -> PPYOLOER:
    """Factory function to create a PP-YOLOE-R model with default configuration.

    Args:
        num_classes: Number of object classes

    Returns:
        Complete PP-YOLOE-R model ready for training/inference
    """
    # Create backbone
    backbone = CSPResNet(
        layers=[3, 6, 6, 3],
        channels=[64, 128, 256, 512, 1024],
        return_levels=[1, 2, 3],  # Return P3, P4, P5 features
        use_large_stem=True,
        act="swish",
    )

    # Create neck
    neck = CustomCSPPAN(
        in_channels=backbone.out_channels,
        out_channels=[192, 384, 768],
        stage_num=1,
        block_num=3,
        act="swish",
        spp=True,
        use_alpha=True,
    )

    # Create criterion
    criterion = RotatedDetectionLoss(
        num_classes=num_classes,
        angle_bins=90,
        use_varifocal=True,
        loss_weights={"cls": 1.0, "box": 2.5, "angle": 0.05},
    )

    # Create head with integrated criterion
    head = PPYOLOERHead(
        in_channels=neck.out_channels,
        num_classes=num_classes,
        fpn_strides=[8, 16, 32],  # P3, P4, P5 strides
        grid_cell_offset=0.5,
        angle_max=90,
        act="swish",
        criterion=criterion,
    )

    # Compose full model
    model = PPYOLOER(backbone, neck, head)
    return model
