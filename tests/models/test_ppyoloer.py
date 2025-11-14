import math
import warnings

import pytest
import torch
import torch.nn as nn

from rotated.backbones import CSPResNet
from rotated.losses.ppyoloer_criterion import LossComponents
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
        losses, decoded_boxes, scores, labels = model(test_images)

    # Verify inference outputs
    torch.testing.assert_close(losses.total, torch.tensor(0.0))
    assert decoded_boxes.shape[0] == batch_size
    assert decoded_boxes.shape[2] == 5  # [cx, cy, w, h, angle]
    assert scores.shape[0] == batch_size
    assert labels.shape[0] == batch_size

    assert not torch.isnan(decoded_boxes).any()
    assert not torch.isnan(scores).any()
    assert torch.all(scores >= 0) and torch.all(scores <= 1)


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

    losses, *_ = model(test_images, test_targets)

    # Verify training outputs
    assert losses is not None
    assert isinstance(losses, LossComponents)

    # Verify loss components are valid
    assert torch.isfinite(losses.total), "Total loss is not finite"
    assert torch.isfinite(losses.cls), "Classification loss is not finite"
    assert torch.isfinite(losses.box), "Box loss is not finite"
    assert torch.isfinite(losses.angle), "Angle loss is not finite"

    assert losses.total >= 0, "Total loss is negative"
    assert losses.cls >= 0, "Classification loss is negative"
    assert losses.box >= 0, "Box loss is negative"
    assert losses.angle >= 0, "Angle loss is negative"

    # Test backward pass
    losses.total.backward()

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
        _, decoded_boxes, scores, labels = model(test_images)

    # Calculate expected number of anchors
    fpn_strides = [8, 16, 32]
    expected_anchors = sum((img_size // stride) ** 2 for stride in fpn_strides)

    assert decoded_boxes.shape == (batch_size, expected_anchors, 5)
    assert scores.shape == (batch_size, expected_anchors)
    assert labels.shape == (batch_size, expected_anchors)


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
    losses, _, scores, _ = model(x)
    torch.testing.assert_close(losses.total, torch.tensor(0.0))
    assert scores.shape[0] == 1


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

    with pytest.raises(RuntimeError, match="Model must be in eval mode before export."):
        model.export()


def test_torchscript_tracing():
    """Test TorchScript tracing with varying batch sizes."""
    # Suppress specific TracerWarnings about:
    # 1. Data flow from anchors generation with fixed image size
    # 2. Constant losses during inference
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    model = create_ppyoloer_model(num_classes=15)
    model.eval()
    model.export()

    # Trace with batch_size=1
    dummy_input = torch.randn(1, 3, 640, 640)
    traced_model = torch.jit.trace(model, dummy_input)

    # Test with different batch sizes
    for batch_size in [1, 2, 4]:
        test_input = torch.randn(batch_size, 3, 640, 640)

        with torch.no_grad():
            # Original model
            _, boxes_orig, scores_orig, labels_orig = model(test_input)
            # Traced model
            _, boxes_trace, scores_trace, labels_trace = traced_model(test_input)

        # Verify outputs match
        assert torch.allclose(boxes_orig, boxes_trace, atol=1e-4), f"Boxes mismatch for batch_size={batch_size}"
        assert torch.allclose(scores_orig, scores_trace, atol=1e-4), f"Scores mismatch for batch_size={batch_size}"
        assert torch.equal(labels_orig, labels_trace), f"Labels mismatch for batch_size={batch_size}"

        # Verify shapes
        assert boxes_orig.shape[0] == batch_size
        assert scores_orig.shape[0] == batch_size
        assert labels_orig.shape[0] == batch_size

    warnings.resetwarnings()
