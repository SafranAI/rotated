import pytest
import torch

from rotated.layers import ConvBNLayer


def test_conv_bn_layer_basic():
    """Test ConvBNLayer basic functionality."""
    layer = ConvBNLayer(in_channels=3, out_channels=64, kernel_size=3, padding=1, act="swish")
    input_tensor = torch.randn(1, 3, 224, 224)

    output_tensor = layer(input_tensor)

    assert output_tensor.shape == (1, 64, 224, 224)
    assert not torch.isnan(output_tensor).any()


def test_conv_bn_layer_activations():
    """Test different activation functions work."""
    activations = ["swish", "relu", "gelu", None]

    for act in activations:
        layer = ConvBNLayer(in_channels=3, out_channels=16, kernel_size=1, act=act)
        input_tensor = torch.randn(1, 3, 32, 32)

        output_tensor = layer(input_tensor)
        assert output_tensor.shape == (1, 16, 32, 32)


def test_conv_bn_layer_invalid_inputs():
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError):
        ConvBNLayer(in_channels=0, out_channels=64)

    with pytest.raises(NotImplementedError):
        ConvBNLayer(in_channels=3, out_channels=64, act="invalid_activation")
