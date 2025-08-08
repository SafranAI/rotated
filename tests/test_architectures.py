import math

import torch

from rotated.architecture import PPYOLOER, create_ppyoloe_r_model
from rotated.backbones import CSPResNet
from rotated.head import PPYOLOERHead
from rotated.neck import CustomCSPPAN


def test_ppyoloer_init():
    """Test PPYOLOER initialization."""
    # Create simple components
    backbone = CSPResNet(layers=[1, 1, 1, 1], channels=[32, 64, 128, 256, 512])
    neck = CustomCSPPAN(in_channels=[128, 256, 512], out_channels=[96, 192, 384])
    head = PPYOLOERHead(in_channels=[96, 192, 384], num_classes=10)

    model = PPYOLOER(backbone, neck, head)

    assert model.backbone is backbone
    assert model.neck is neck
    assert model.head is head


def test_ppyoloer_forward_inference():
    """Test PPYOLOER forward pass in inference mode."""
    model = create_ppyoloe_r_model(num_classes=15)
    model.eval()

    batch_size = 2
    img_size = 640
    test_images = torch.randn(batch_size, 3, img_size, img_size)

    with torch.no_grad():
        losses, cls_scores, decoded_boxes = model(test_images)

    # Verify inference outputs
    assert losses is None
    assert cls_scores.shape[0] == batch_size
    assert decoded_boxes.shape[0] == batch_size
    assert cls_scores.shape[2] == 15  # num_classes
    assert decoded_boxes.shape[2] == 5  # [cx, cy, w, h, angle]

    assert not torch.isnan(cls_scores).any()
    assert not torch.isnan(decoded_boxes).any()
    assert torch.all(cls_scores >= 0) and torch.all(cls_scores <= 1)


def test_ppyoloer_forward_training():
    """Test PPYOLOER forward pass in training mode."""
    model = create_ppyoloe_r_model(num_classes=15)
    model.train()

    batch_size = 2
    img_size = 640
    num_targets = 3

    test_images = torch.randn(batch_size, 3, img_size, img_size)
    test_targets = {
        "labels": torch.randint(0, 15, (batch_size, num_targets, 1)),
        "boxes": torch.cat([
            torch.rand(batch_size, num_targets, 2) * 400 + 100,  # cx, cy
            torch.rand(batch_size, num_targets, 2) * 50 + 20,    # w, h
            torch.rand(batch_size, num_targets, 1) * (math.pi / 2),  # angle
        ], dim=-1),
        "valid_mask": torch.ones(batch_size, num_targets, 1),
    }

    losses, cls_scores, decoded_boxes = model(test_images, test_targets)

    # Verify training outputs
    assert losses is not None
    assert isinstance(losses, dict)
    assert "total" in losses
    assert "cls" in losses
    assert "box" in losses
    assert "angle" in losses

    # Verify loss values
    for loss_name, loss_value in losses.items():
        assert torch.isfinite(loss_value), f"{loss_name} loss is not finite"
        assert loss_value >= 0, f"{loss_name} loss is negative"

    # Test backward pass
    losses["total"].backward()

    # Verify some gradients exist
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            break
    assert has_gradients, "No gradients found after backward pass"


def test_create_ppyoloe_r_model():
    """Test create_ppyoloe_r_model factory function."""
    # Test default configuration
    model = create_ppyoloe_r_model()
    assert isinstance(model, PPYOLOER)
    assert hasattr(model, 'backbone')
    assert hasattr(model, 'neck')
    assert hasattr(model, 'head')

    # Test custom num_classes
    model_custom = create_ppyoloe_r_model(num_classes=20)
    assert model_custom.head.num_classes == 20

    # Test model components are properly configured
    assert isinstance(model.backbone, CSPResNet)
    assert isinstance(model.neck, CustomCSPPAN)
    assert isinstance(model.head, PPYOLOERHead)


def test_ppyoloer_output_shapes():
    """Test PPYOLOER output shapes are correct."""
    model = create_ppyoloe_r_model(num_classes=10)

    batch_size = 1
    img_size = 416  # Different size to test flexibility
    test_images = torch.randn(batch_size, 3, img_size, img_size)

    model.eval()
    with torch.no_grad():
        losses, cls_scores, decoded_boxes = model(test_images)

    # Calculate expected number of anchors
    fpn_strides = [8, 16, 32]
    expected_anchors = sum((img_size // stride) ** 2 for stride in fpn_strides)

    assert cls_scores.shape == (batch_size, expected_anchors, 10)
    assert decoded_boxes.shape == (batch_size, expected_anchors, 5)
