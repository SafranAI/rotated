from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rotated.layers import ConvBNLayer


class RepVggBlock(nn.Module):
    """RepVGG block with structural re-parameterization capability.

    During training, uses separate 3x3 and 1x1 convolutions that can be
    fused into a single 3x3 conv during inference for improved efficiency.
    """

    def __init__(self, in_channels: int, out_channels: int, act: str = "relu", alpha: bool = False):
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"Input channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"Output channels must be positive, got {out_channels}")

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = ConvBNLayer(in_channels, out_channels, 3, stride=1, padding=1, act=None)
        self.conv2 = ConvBNLayer(in_channels, out_channels, 1, stride=1, padding=0, act=None)
        self.act = self._get_activation(act)

        if alpha:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.alpha = None

    def _get_activation(self, act: str) -> nn.Module:
        """Get activation function based on string identifier."""
        activation_map = {
            "swish": nn.SiLU(),
            "leaky": nn.LeakyReLU(0.1),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
        }
        return activation_map.get(act, nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional alpha weighting."""
        if hasattr(self, "conv"):
            # Deployed mode: single fused conv
            y = self.conv(x)
        else:
            # Training mode: separate convs
            alpha = self.alpha if self.alpha is not None else 1.0
            y = self.conv1(x) + alpha * self.conv2(x)
        y = self.act(y)
        return y

    def convert_to_deploy(self) -> None:
        """Convert to deployment mode by fusing convolutions."""
        if not hasattr(self, "conv"):
            self.conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
            )

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

        # Remove training components
        delattr(self, "conv1")
        delattr(self, "conv2")

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get equivalent kernel and bias for fused convolution."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        if self.alpha is not None:
            return (kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + self.alpha * bias1x1)
        else:
            return (kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1)

    def _pad_1x1_to_3x3_tensor(self, kernel1x1: torch.Tensor) -> torch.Tensor:
        """Pad 1x1 kernel to 3x3 size."""
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvBNLayer) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse conv and batch norm into equivalent conv parameters."""
        if branch is None:
            return 0, 0

        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std


class BasicBlock(nn.Module):
    """Basic residual block for CSPResNet.

    Implements a residual connection with two convolutions: a standard ConvBNLayer
    followed by a RepVggBlock, with optional shortcut connection.
    """

    def __init__(
        self, in_channels: int, out_channels: int, act: str = "relu", shortcut: bool = True, use_alpha: bool = False
    ):
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"Input channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"Output channels must be positive, got {out_channels}")
        if shortcut and in_channels != out_channels:
            raise ValueError(
                f"Input and output channels must match for residual connection, got {in_channels} != {out_channels}"
            )

        self.conv1 = ConvBNLayer(in_channels, out_channels, 3, stride=1, padding=1, act=act)
        self.conv2 = RepVggBlock(out_channels, out_channels, act=act, alpha=use_alpha)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection."""
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


class EffectiveSELayer(nn.Module):
    """Effective Squeeze-Excitation layer.

    From "CenterMask: Real-Time Anchor-Free Instance Segmentation"
    Applies global average pooling followed by channel attention.
    """

    def __init__(self, channels: int, act: str = "hardsigmoid"):
        super().__init__()

        if channels <= 0:
            raise ValueError(f"Channels must be positive, got {channels}")

        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = self._get_activation(act)

    def _get_activation(self, act: str) -> nn.Module:
        """Get activation function based on string identifier."""
        activation_map = {
            "hardsigmoid": nn.Hardsigmoid(),
            "sigmoid": nn.Sigmoid(),
            "swish": nn.SiLU(),
            "relu": nn.ReLU(),
        }
        return activation_map.get(act, nn.Hardsigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply squeeze-excitation attention mechanism."""
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)


class CSPResStage(nn.Module):
    """Cross Stage Partial ResNet stage.

    Implements CSP structure with optional downsampling, attention mechanism,
    and residual blocks. Splits features into two paths and merges them.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        act: str = "relu",
        attn: str | None = "eca",
        use_alpha: bool = False,
    ):
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"Input channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"Output channels must be positive, got {out_channels}")
        if num_blocks <= 0:
            raise ValueError(f"Number of blocks must be positive, got {num_blocks}")
        if stride not in [1, 2]:
            raise ValueError(f"Stride must be 1 or 2, got {stride}")

        mid_channels = (in_channels + out_channels) // 2

        # Optional downsampling conv
        if stride == 2:
            self.conv_down = ConvBNLayer(in_channels, mid_channels, 3, stride=2, padding=1, act=act)
        else:
            self.conv_down = None

        # CSP split convolutions
        self.conv1 = ConvBNLayer(mid_channels, mid_channels // 2, 1, act=act)
        self.conv2 = ConvBNLayer(mid_channels, mid_channels // 2, 1, act=act)

        # Residual blocks
        self.blocks = nn.Sequential(
            *[
                BasicBlock(mid_channels // 2, mid_channels // 2, act=act, shortcut=True, use_alpha=use_alpha)
                for i in range(num_blocks)
            ]
        )

        # Optional attention mechanism
        if attn:
            self.attn = EffectiveSELayer(mid_channels, act="hardsigmoid")
        else:
            self.attn = None

        # Final output conv
        self.conv3 = ConvBNLayer(mid_channels, out_channels, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CSP stage."""
        # Optional downsampling
        if self.conv_down is not None:
            x = self.conv_down(x)

        # CSP split and process
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], dim=1)

        # Optional attention
        if self.attn is not None:
            y = self.attn(y)

        y = self.conv3(y)
        return y


class CSPResNet(nn.Module):
    """Cross Stage Partial ResNet backbone for object detection.

    CSPResNet improves information flow and reduces computational cost by
    splitting feature maps into two parts and applying different processing
    paths. Designed for efficient object detection tasks.
    """

    def __init__(
        self,
        layers: Sequence[int] = (3, 6, 6, 3),
        channels: Sequence[int] = (64, 128, 256, 512, 1024),
        act: str = "swish",
        return_idx: Sequence[int] = (1, 2, 3),
        use_large_stem: bool = False,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        use_alpha: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        # Validation
        if not layers or any(layer <= 0 for layer in layers):
            raise ValueError(f"All layer counts must be positive, got {layers}")
        if not channels or any(ch <= 0 for ch in channels):
            raise ValueError(f"All channel counts must be positive, got {channels}")
        if len(channels) != len(layers) + 1:
            raise ValueError(f"Channels list must have {len(layers) + 1} elements, got {len(channels)}")
        if width_mult <= 0:
            raise ValueError(f"Width multiplier must be positive, got {width_mult}")
        if depth_mult <= 0:
            raise ValueError(f"Depth multiplier must be positive, got {depth_mult}")
        if any(idx < 0 or idx >= len(layers) for idx in return_idx):
            raise ValueError(f"Return indices must be in range [0, {len(layers) - 1}], got {return_idx}")

        self.use_checkpoint = use_checkpoint
        self.return_idx = list(return_idx)

        # Apply multipliers and convert to lists for TorchScript compatibility
        channels_list = [max(round(c * width_mult), 1) for c in channels]
        layers_list = [max(round(layer * depth_mult), 1) for layer in layers]

        # Build stem
        self.stem = self._build_stem(channels_list[0], act, use_large_stem)

        # Build stages
        num_stages = len(channels_list) - 1
        self.stages = nn.ModuleList(
            [
                CSPResStage(
                    channels_list[i], channels_list[i + 1], layers_list[i], stride=2, act=act, use_alpha=use_alpha
                )
                for i in range(num_stages)
            ]
        )

        # Store output specifications
        self._out_channels = channels_list[1:]
        self._out_strides = [4 * 2**i for i in range(num_stages)]

    def _build_stem(self, base_channels: int, act: str, use_large_stem: bool) -> nn.Sequential:
        """Build the stem (initial feature extraction) layers."""
        if use_large_stem:
            return nn.Sequential(
                ConvBNLayer(3, base_channels // 2, 3, stride=2, padding=1, act=act),
                ConvBNLayer(base_channels // 2, base_channels // 2, 3, stride=1, padding=1, act=act),
                ConvBNLayer(base_channels // 2, base_channels, 3, stride=1, padding=1, act=act),
            )
        else:
            return nn.Sequential(
                ConvBNLayer(3, base_channels // 2, 3, stride=2, padding=1, act=act),
                ConvBNLayer(base_channels // 2, base_channels, 3, stride=1, padding=1, act=act),
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through CSPResNet backbone.

        Args:
            x: Input tensor with shape [B, 3, H, W]

        Returns:
            List of feature maps from specified return indices
        """
        x = self.stem(x)
        outs = []

        for idx, stage in enumerate(self.stages):
            if self.use_checkpoint and self.training:
                # Gradient checkpointing for memory efficiency
                x = torch.utils.checkpoint.checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)

            if idx in self.return_idx:
                outs.append(x)

        return outs

    @property
    def out_channels(self) -> list[int]:
        """Get output channel dimensions for returned feature levels."""
        return [self._out_channels[i] for i in self.return_idx]


if __name__ == "__main__":
    print("Testing CSPResNet")

    # Create model
    model = CSPResNet()
    model.eval()

    # Test input
    test_input = torch.randn(1, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        outputs = model(test_input)

    # Verify outputs
    assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"

    print("Forward pass successful")
