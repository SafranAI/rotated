import torch

from rotated.backbones.csp_resnet import CSPResNet


def test_csp_resnet():
    """Test CSPResNet functionality."""
    model = CSPResNet(
        layers=(3, 6, 6, 3),
        channels=(64, 128, 256, 512, 1024),
        return_levels=(1, 2, 3),  # P3, P4, P5
    )

    input_tensor = torch.randn(1, 3, 640, 640)
    outputs = model(input_tensor)

    # Should return 3 features for P3, P4, P5
    assert len(outputs) == 3

    # Check output shapes (stride 8, 16, 32)
    assert outputs[0].shape == (1, 256, 80, 80)  # P3: 640/8 = 80
    assert outputs[1].shape == (1, 512, 40, 40)  # P4: 640/16 = 40
    assert outputs[2].shape == (1, 1024, 20, 20)  # P5: 640/32 = 20

    for output in outputs:
        assert not torch.isnan(output).any()


def test_csp_resnet_export():
    """Test CSPResNet export functionality."""
    model = CSPResNet(
        layers=(1, 1, 1, 1),
        channels=(32, 64, 128, 256, 512),
        return_levels=(1, 2, 3),
        use_alpha=True,
    )

    # Set model in eval mode for export
    model.eval()

    # Check initial state
    assert not model._exported
    for stage in model.stages:
        for block in stage.blocks:
            assert hasattr(block.conv2, "conv1")
            assert hasattr(block.conv2, "conv2")

    input_tensor = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        outputs_before = model(input_tensor)

    model.export()

    # Check exported state
    assert model._exported
    for stage in model.stages:
        for block in stage.blocks:
            assert hasattr(block.conv2, "conv")
            assert not hasattr(block.conv2, "conv1")
            assert not hasattr(block.conv2, "conv2")

    # Multiple exports should be safe (idempotent)
    model.export()
    model.export()

    with torch.no_grad():
        outputs_after = model(input_tensor)

    assert len(outputs_after) == 3
    for output in outputs_after:
        assert not torch.isnan(output).any()

    for out_before, out_after in zip(outputs_before, outputs_after, strict=False):
        assert torch.allclose(out_before, out_after, atol=1e-4)


def test_csp_resnet_export_requires_eval():
    """Test that export requires eval mode."""
    model = CSPResNet(
        layers=(1, 1, 1, 1),
        channels=(32, 64, 128, 256, 512),
        return_levels=(1, 2, 3),
    )

    # Explicitly set to training mode
    model.train()

    try:
        model.export()
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as e:
        assert "eval mode" in str(e)
