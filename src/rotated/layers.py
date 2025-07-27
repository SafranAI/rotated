import torch
import torch.nn as nn


class ConvBNLayer(nn.Module):
    """Basic convolutional layer with batch normalization and activation.

    Combines Conv2d, BatchNorm2d, and activation function into a single module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        padding: int = 0,
        act: str | None = None,
    ):
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"Input channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"Output channels must be positive, got {out_channels}")
        if filter_size <= 0:
            raise ValueError(f"Filter size must be positive, got {filter_size}")

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = self._get_activation(act)

    def _get_activation(self, act: str | None) -> nn.Module:
        """Get activation function based on string identifier."""
        activation_map = {
            "swish": nn.SiLU(),
            "leaky": nn.LeakyReLU(0.1),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "hardsigmoid": nn.Hardsigmoid(),
            None: nn.Identity(),
        }
        if act not in activation_map:
            raise NotImplementedError(f"Activation {act} not implemented")
        return activation_map[act]

    def forward(self, x):
        """Forward pass through conv-bn-activation sequence."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


if __name__ == "__main__":
    import torch

    print("Testing ConvBNLayer")

    # Test basic functionality
    layer = ConvBNLayer(in_channels=3, out_channels=64, filter_size=3, padding=1, act="swish")
    test_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = layer(test_input)

    assert output.shape == (1, 64, 224, 224), f"Expected (1, 64, 224, 224), got {output.shape}"

    # Test all activations
    activations = ["swish", "leaky", "relu", "gelu", "hardsigmoid", None]
    for act in activations:
        layer = ConvBNLayer(in_channels=3, out_channels=16, act=act)
        output = layer(test_input)
        print(f"✓ Activation '{act}' works")

    print("All tests passed")
