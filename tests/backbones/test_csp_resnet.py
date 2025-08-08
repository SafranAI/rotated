import torch

from rotated.backbones.csp_resnet import CSPResNet


def test_csp_resnet():
    """Test CSPResNet functionality."""
    model = CSPResNet(
        layers=(3, 6, 6, 3),
        channels=(64, 128, 256, 512, 1024),
        return_levels=(1, 2, 3)  # P3, P4, P5
    )

    input_tensor = torch.randn(1, 3, 640, 640)
    outputs = model(input_tensor)

    # Should return 3 features for P3, P4, P5
    assert len(outputs) == 3

    # Check output shapes (stride 8, 16, 32)
    assert outputs[0].shape == (1, 256, 80, 80)   # P3: 640/8 = 80
    assert outputs[1].shape == (1, 512, 40, 40)   # P4: 640/16 = 40
    assert outputs[2].shape == (1, 1024, 20, 20)  # P5: 640/32 = 20

    for output in outputs:
        assert not torch.isnan(output).any()
