import math

import torch
import torch.nn as nn

from rotated.backbones import CSPResNet
from rotated.models.ppyoloer import PPYOLOER, create_ppyoloer_model
from rotated.nn.custom_pan import CustomCSPPAN
from rotated.nn.ppyoloer_head import PPYOLOERHead


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
    model = create_ppyoloer_model(num_classes=15)
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
    model = create_ppyoloer_model(num_classes=15)
    model.train()

    batch_size = 2
    img_size = 640
    num_targets = 3

    test_images = torch.randn(batch_size, 3, img_size, img_size)
    test_targets = {
        "labels": torch.randint(0, 15, (batch_size, num_targets, 1)),
        "boxes": torch.cat(
            [
                torch.rand(batch_size, num_targets, 2) * 400 + 100,  # cx, cy
                torch.rand(batch_size, num_targets, 2) * 50 + 20,  # w, h
                torch.rand(batch_size, num_targets, 1) * (math.pi / 2),  # angle
            ],
            dim=-1,
        ),
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


def test_create_ppyoloer_model():
    """Test create_ppyoloer_model factory function."""
    # Test default configuration
    model = create_ppyoloer_model()
    assert isinstance(model, PPYOLOER)
    assert hasattr(model, "backbone")
    assert hasattr(model, "neck")
    assert hasattr(model, "head")

    # Test custom num_classes
    model_custom = create_ppyoloer_model(num_classes=20)
    assert model_custom.head.num_classes == 20

    # Test model components are properly configured
    assert isinstance(model.backbone, CSPResNet)
    assert isinstance(model.neck, CustomCSPPAN)
    assert isinstance(model.head, PPYOLOERHead)


def test_ppyoloer_output_shapes():
    """Test PPYOLOER output shapes are correct."""
    model = create_ppyoloer_model(num_classes=10)

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


class DummyBackbone(nn.Module):
    """Simple backbone without export method."""

    def __init__(self):
        super().__init__()
        self._out_channels = [128, 256, 512]
        self._out_strides = [8, 16, 32]

    def forward(self, x):
        b = x.shape[0]
        return [
            torch.randn(b, 128, 80, 80),  # stride 8
            torch.randn(b, 256, 40, 40),  # stride 16
            torch.randn(b, 512, 20, 20),  # stride 32
        ]

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def out_strides(self):
        return self._out_strides


class DummyBackboneWithExport(DummyBackbone):
    """Backbone with export method."""

    def __init__(self):
        super().__init__()
        self._exported = False

    def export(self):
        self._exported = True


def test_export_with_backbone_no_export():
    """Test export when backbone doesn't have export method."""
    backbone = DummyBackbone()
    neck = CustomCSPPAN(in_channels=[128, 256, 512], out_channels=[96, 192, 384])
    head = PPYOLOERHead(in_channels=[96, 192, 384], num_classes=10)

    model = PPYOLOER(backbone, neck, head)

    # Set model in eval mode for export
    model.eval()

    # Should not raise error even without export method
    model.export()

    # Test forward still works
    x = torch.randn(1, 3, 640, 640)
    losses, cls_scores, _ = model(x)
    assert losses is None
    assert cls_scores.shape[0] == 1


def test_export_with_backbone_export():
    """Test export when backbone has export method."""
    backbone = DummyBackboneWithExport()
    neck = CustomCSPPAN(in_channels=[128, 256, 512], out_channels=[96, 192, 384])
    head = PPYOLOERHead(in_channels=[96, 192, 384], num_classes=10)

    model = PPYOLOER(backbone, neck, head)

    # Set model in eval mode for export
    model.eval()

    assert not backbone._exported
    model.export()
    assert backbone._exported


def test_export_with_csp_resnet():
    """Test export with CSPResNet backbone."""
    model = create_ppyoloer_model(num_classes=10)

    # Set model in eval mode for export
    model.eval()

    first_block = model.backbone.stages[0].blocks[0]
    assert hasattr(first_block.conv2, "conv1")
    assert not model.backbone._exported

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out_before = model(x)

    model.export()

    assert model.backbone._exported
    assert hasattr(first_block.conv2, "conv")
    assert not hasattr(first_block.conv2, "conv1")

    # Multiple exports should be safe (idempotent)
    model.export()
    model.export()

    with torch.no_grad():
        out_after = model(x)

    assert torch.allclose(out_before[1], out_after[1], atol=1e-4)
    assert torch.allclose(out_before[2], out_after[2], atol=1e-4)


def test_ppyoloer_export_requires_eval():
    """Test that PPYOLOER export requires eval mode."""
    model = create_ppyoloer_model(num_classes=10)

    # Explicitly set to training mode
    model.train()

    try:
        model.export()
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as e:
        assert "eval mode" in str(e)
