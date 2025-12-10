from typing import Literal, TypeAlias

import torch.nn as nn

ActivationType: TypeAlias = Literal["swish", "leaky", "relu", "gelu", "hardsigmoid", "sigmoid", None]
"""Type alias for supported activation functions."""


def get_activation(act: ActivationType = None, inplace: bool = False) -> nn.Module:
    """Get activation function based on string identifier.

    Args:
        act: String identifier for activation function. Options are:
            - 'swish': SiLU activation
            - 'leaky': LeakyReLU with negative slope 0.1
            - 'relu': ReLU activation
            - 'gelu': GELU activation
            - 'hardsigmoid': Hardsigmoid activation
            - 'sigmoid': Sigmoid activation
            - None: Identity (no activation)
        inplace: Whether to use in-place operations where supported

    Returns:
        nn.Module: Activation function module

    Raises:
        NotImplementedError: If activation is not supported
    """
    activation_map = {
        "swish": nn.SiLU(inplace=inplace),
        "leaky": nn.LeakyReLU(0.1, inplace=inplace),
        "relu": nn.ReLU(inplace=inplace),
        "gelu": nn.GELU(),  # GELU doesn't support inplace
        "hardsigmoid": nn.Hardsigmoid(inplace=inplace),
        "sigmoid": nn.Sigmoid(),  # Sigmoid doesn't support inplace
        None: nn.Identity(),
    }
    if act not in activation_map:
        raise NotImplementedError(f"Activation {act} not implemented")
    return activation_map[act]


class ConvBNLayer(nn.Module):
    """Combines Conv2d, BatchNorm2d, and activation function into a single module.

    Note:
    BatchNorm2d layer uses eps=1e-3 and momentum=0.03, which are common values used in YOLO family models
    for better numerical stability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        padding: int = 0,
        act: ActivationType = None,
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

        # BatchNorm2d with eps=1e-3 and momentum=0.03, common values used in YOLO family models
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.act = get_activation(act, inplace=True)  # Store activation module with in-place operations

    def forward(self, input_tensor):
        """Forward pass through conv-bn-activation sequence."""
        output_tensor = self.conv(input_tensor)
        output_tensor = self.bn(output_tensor)
        return self.act(output_tensor)
