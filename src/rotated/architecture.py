import math

import torch
import torch.nn as nn

from rotated.backbone import CSPResNet
from rotated.criterion import RotatedDetectionLoss
from rotated.head import PPYOLOERHead
from rotated.neck import CustomCSPPAN


class PPYOLOERv3(nn.Module):
    """PP-YOLOE-R Architecture for Rotated Object Detection.

    Simple composition of backbone, neck, and head components.
    The head uses a consistent interface that always returns the same format:
    (losses, cls_scores, decoded_boxes) where losses is None during inference.
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


def create_ppyoloe_r_model(num_classes: int = 15) -> PPYOLOERv3:
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
        return_idx=[1, 2, 3],  # Return P3, P4, P5 features
        use_large_stem=True,
        act="swish",
    )

    # Create neck
    neck = CustomCSPPAN(
        in_channels=[256, 512, 1024],  # From backbone return_idx
        out_channels=[192, 384, 768],  # Match head expectations
        stage_num=1,
        block_num=3,
        act="swish",
        spp=True,
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
        in_channels=[192, 384, 768],  # Match neck outputs
        num_classes=num_classes,
        fpn_strides=[8, 16, 32],  # P3, P4, P5 strides
        grid_cell_offset=0.5,
        angle_max=90,
        act="swish",
        criterion=criterion,
    )

    # Compose full model
    model = PPYOLOERv3(backbone, neck, head)
    return model


if __name__ == "__main__":
    print("Testing PP-YOLOE-R Architecture")

    # Test parameters
    batch_size = 2
    num_classes = 15
    img_size = 640
    num_targets = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = create_ppyoloe_r_model(num_classes=num_classes)
    model = model.to(device)

    # Create test inputs
    test_images = torch.randn(batch_size, 3, img_size, img_size, device=device)
    test_targets = {
        "labels": torch.randint(0, num_classes, (batch_size, num_targets, 1), device=device),
        "boxes": torch.cat(
            [
                torch.rand(batch_size, num_targets, 2, device=device) * 400 + 100,
                torch.rand(batch_size, num_targets, 2, device=device) * 50 + 20,
                torch.rand(batch_size, num_targets, 1, device=device) * (math.pi / 2),
            ],
            dim=-1,
        ),
        "valid_mask": torch.ones(batch_size, num_targets, 1, device=device),
    }

    # Test forward and backward
    model.train()
    losses, cls_scores, decoded_boxes = model(test_images, test_targets)
    losses["total"].backward()

    print("Forward and backward pass successful")
