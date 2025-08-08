# Modified from PaddleDetection (https://github.com/PaddlePaddle/PaddleDetection)
# Copyright (c) 2024 PaddlePaddle Authors. Apache 2.0 License.

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNLayer(nn.Module):
    """Basic convolutional layer with batch normalization and activation."""

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
        return activation_map.get(act, nn.Identity())

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv-bn-activation sequence."""
        output_tensor = self.conv(input_tensor)
        output_tensor = self.bn(output_tensor)
        output_tensor = self.act(output_tensor)
        return output_tensor


class RepVggBlock(nn.Module):
    """RepVGG block with structural re-parameterization capability."""

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

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional alpha weighting."""
        if hasattr(self, "conv"):
            # Deployed mode: single fused conv
            output_tensor = self.conv(input_tensor)
        else:
            # Training mode: separate convs
            alpha = self.alpha if self.alpha is not None else 1.0
            output_tensor = self.conv1(input_tensor) + alpha * self.conv2(input_tensor)
        output_tensor = self.act(output_tensor)
        return output_tensor

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
    """Basic residual block for CSPResNet."""

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

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection."""
        output_tensor = self.conv1(input_tensor)
        output_tensor = self.conv2(output_tensor)
        if self.shortcut:
            return input_tensor + output_tensor
        else:
            return output_tensor


class EffectiveSELayer(nn.Module):
    """Effective Squeeze-Excitation layer."""

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

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply squeeze-excitation attention mechanism."""
        se_features = input_tensor.mean((2, 3), keepdim=True)
        se_features = self.fc(se_features)
        return input_tensor * self.act(se_features)


class CSPResStage(nn.Module):
    """Cross Stage Partial ResNet stage."""

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
                for block_idx in range(num_blocks)
            ]
        )

        # Optional attention mechanism
        if attn:
            self.attn = EffectiveSELayer(mid_channels, act="hardsigmoid")
        else:
            self.attn = None

        # Final output conv
        self.conv3 = ConvBNLayer(mid_channels, out_channels, 1, act=act)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through CSP stage."""
        # Optional downsampling
        if self.conv_down is not None:
            input_tensor = self.conv_down(input_tensor)

        # CSP split and process
        y1 = self.conv1(input_tensor)
        y2 = self.blocks(self.conv2(input_tensor))
        combined_features = torch.cat([y1, y2], dim=1)

        # Optional attention
        if self.attn is not None:
            combined_features = self.attn(combined_features)

        output_tensor = self.conv3(combined_features)
        return output_tensor


class CSPResNet(nn.Module):
    """Cross Stage Partial ResNet backbone for object detection.

    Feature levels correspond to stages after the stem:
    - Level 0: First stage (stride 4)
    - Level 1: Second stage (stride 8)
    - Level 2: Third stage (stride 16)
    - Level 3: Fourth stage (stride 32)

    Default return_levels=(1, 2, 3) gives strides [8, 16, 32].
    For detection including P2: return_levels=(0, 1, 2) gives strides [4, 8, 16].
    """

    def __init__(
        self,
        layers: Sequence[int] = (3, 6, 6, 3),
        channels: Sequence[int] = (64, 128, 256, 512, 1024),
        act: str = "swish",
        return_levels: Sequence[int] = (1, 2, 3),  # Return stride [8, 16, 32] features
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
        if any(idx < 0 or idx >= len(layers) for idx in return_levels):
            raise ValueError(f"Return levels must be in range [0, {len(layers) - 1}], got {return_levels}")

        self.use_checkpoint = use_checkpoint
        self.return_levels = list(return_levels)

        # Apply multipliers and convert to lists for TorchScript compatibility
        channels_list = [max(round(channels_val * width_mult), 1) for channels_val in channels]
        layers_list = [max(round(layer * depth_mult), 1) for layer in layers]

        # Build stem
        self.stem = self._build_stem(channels_list[0], act, use_large_stem)

        # Build stages
        num_stages = len(channels_list) - 1
        self.stages = nn.ModuleList(
            [
                CSPResStage(
                    channels_list[stage_idx], channels_list[stage_idx + 1], layers_list[stage_idx], stride=2, act=act, use_alpha=use_alpha
                )
                for stage_idx in range(num_stages)
            ]
        )

        # Store output specifications
        self._out_channels = channels_list[1:]
        self._out_strides = [4 * 2**stage_idx for stage_idx in range(num_stages)]

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

    def forward(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
        """Extract features at specified levels."""
        features = self.stem(input_tensor)
        outs = []

        for stage_idx, stage in enumerate(self.stages):
            if self.use_checkpoint and self.training:
                # Gradient checkpointing for memory efficiency
                features = torch.utils.checkpoint.checkpoint(stage, features, use_reentrant=False)
            else:
                features = stage(features)

            if stage_idx in self.return_levels:
                outs.append(features)

        return outs

    @property
    def out_channels(self) -> list[int]:
        return [self._out_channels[level_idx] for level_idx in self.return_levels]

    @property
    def out_strides(self) -> list[int]:
        return [self._out_strides[level_idx] for level_idx in self.return_levels]
