# Modified from PaddleDetection (https://github.com/PaddlePaddle/PaddleDetection)
# Copyright (c) 2024 PaddlePaddle Authors. Apache 2.0 License.

from typing import Literal

import torch
import torch.nn as nn

from rotated.backbones import create_csp_resnet
from rotated.losses.ppyoloer_criterion import LossComponents, RotatedDetectionLoss
from rotated.nn.custom_pan import CustomCSPPAN
from rotated.nn.postprocessor import DetectionPostProcessor
from rotated.nn.ppyoloer_head import PPYOLOERHead


class PPYOLOER(nn.Module):
    """PP-YOLOE-R Architecture for Rotated Object Detection.

    Simple composition of backbone, neck, head, and optional postprocessor components.
    The head always returns (losses, decoded_boxes, scores, labels).
    The postprocessor optionally applies NMS for final detection output.

    Reference:
        Title: "PP-YOLOE-R: An Efficient Anchor-Free Rotated Object Detector"
        Authors: Xinxin Wang, Guanzhong Wang, Qingqing Dang, Yi Liu, Xiaoguang Hu, Dianhai Yu
        Paper link: https://arxiv.org/pdf/2211.02386
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: PPYOLOERHead,
        postprocessor: DetectionPostProcessor | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.postprocessor = postprocessor

    def forward(
        self,
        images: torch.Tensor,
        targets: dict[str, torch.Tensor] = None,
        postprocess: bool = True,
    ) -> tuple[LossComponents, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            postprocess: Whether to apply post-processing (NMS)

        Returns:
            Tuple of (losses, boxes, scores, labels):
                - losses: Named tuple of loss components (empty with zeros during inference)
                - boxes: [B, K, 5] or [B, N, 5] - Detection boxes (post-NMS if postprocessor set)
                - scores: [B, K] - Detection scores (post-NMS if postprocessor set, else max class scores [B, N])
                - labels: [B, K] or [B, N] - Detection labels (predicted class indices)
        """
        features = self.backbone(images)
        features = self.neck(features)
        losses, cls_scores, decoded_boxes = self.head(features, targets)

        # Always extract predicted labels and max scores
        pred_labels = torch.argmax(cls_scores, dim=-1)  # [B, N]
        max_scores, _ = torch.max(cls_scores, dim=-1)  # [B, N]

        # Apply post-processing if postprocessor exists (only during inference)
        if self.postprocessor is not None and postprocess:
            decoded_boxes, max_scores, pred_labels = self.postprocessor(decoded_boxes, max_scores, pred_labels)

        return losses, decoded_boxes, max_scores, pred_labels

    def export(self) -> None:
        """Convert the model to deployment mode by reparameterizing layers if available."""
        if self.training:
            raise RuntimeError("Model must be in eval mode before export. Call model.eval() first.")

        if hasattr(self.backbone, "export"):
            self.backbone.export()


PPYOLOERSize = Literal["s", "m", "l", "x"]

PPYOLOER_CONFIGS = {
    "s": {"depth_mult": 0.33, "width_mult": 0.50},
    "m": {"depth_mult": 0.67, "width_mult": 0.75},
    "l": {"depth_mult": 1.00, "width_mult": 1.00},
    "x": {"depth_mult": 1.33, "width_mult": 1.25},
}


def create_ppyoloer_model(
    num_classes: int = 15,
    size: PPYOLOERSize = "s",
    in_chans: int = 3,
    pretrained_backbone: bool = True,
    postprocessor: DetectionPostProcessor | None = None,
) -> PPYOLOER:
    """Factory function to create a PP-YOLOE-R model with specified size.

    Args:
        num_classes: Number of object classes to detect
        size: Model size - "s" (small), "m" (medium), "l" (large), or "x" (extra large)
            Larger models have better accuracy but require more compute
        in_chans: Number of input channels
        pretrained_backbone: If True, loads pretrained backbone weights from GitHub releases
        postprocessor: Optional postprocessor for inference-time NMS

    Returns:
        Complete PP-YOLOE-R model

    Raises:
        ValueError: If an invalid size is provided

    Examples:
        >>> model = create_ppyoloer_model(num_classes=15, size="s")
        >>> # Without pretrained backbone
        >>> model = create_ppyoloer_model(num_classes=15, size="m", pretrained_backbone=False)
        >>> # With postprocessing for inference
        >>> postprocessor = DetectionPostProcessor(score_threshold=0.01, iou_threshold=0.6)
        >>> model = create_ppyoloer_model(num_classes=20, size="l", postprocessor=postprocessor)
    """
    size = size.lower()

    if size not in PPYOLOER_CONFIGS:
        valid_sizes = list(PPYOLOER_CONFIGS.keys())
        raise ValueError(
            f"Invalid size '{size}'. Must be one of {valid_sizes}. "
            f"Choose 's' for small, 'm' for medium, 'l' for large, or 'x' for extra large."
        )

    # Create backbone with optional pretrained weights
    backbone = create_csp_resnet(variant=size, in_chans=in_chans, pretrained=pretrained_backbone)

    # Create neck
    config = PPYOLOER_CONFIGS[size]
    neck = CustomCSPPAN(
        in_channels=backbone.out_channels,
        out_channels=[192, 384, 768],
        stage_num=1,
        block_num=3,
        act="swish",
        spp=True,
        use_alpha=True,
        depth_mult=config["depth_mult"],
        width_mult=config["width_mult"],
    )

    # Create criterion
    criterion = RotatedDetectionLoss(num_classes=num_classes)

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
    model = PPYOLOER(backbone, neck, head, postprocessor)
    return model
