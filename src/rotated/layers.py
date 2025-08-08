import torch.nn as nn


class ConvBNLayer(nn.Module):
    """Basic convolutional layer with batch normalization and activation.

    Combines Conv2d, BatchNorm2d, and activation function into a single module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
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
        if kernel_size <= 0:
            raise ValueError(f"Kernel size must be positive, got {kernel_size}")

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
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

    def forward(self, input_tensor):
        """Forward pass through conv-bn-activation sequence."""
        output_tensor = self.conv(input_tensor)
        output_tensor = self.bn(output_tensor)
        output_tensor = self.act(output_tensor)
        return output_tensor
