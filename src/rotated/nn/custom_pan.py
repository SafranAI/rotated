# Modified from PaddleDetection (https://github.com/PaddlePaddle/PaddleDetection)
# Copyright (c) 2024 PaddlePaddle Authors. Apache 2.0 License.

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rotated.nn.common import ActivationType, ConvBNLayer


class BasicBlock(nn.Module):
    """Basic residual block with optional shortcut connection and alpha parameter."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act: ActivationType = "swish",
        shortcut: bool = True,
        use_alpha: bool = False,
    ):
        super().__init__()
        self.conv1 = ConvBNLayer(in_channels, out_channels, 3, padding=1, act=act)
        self.conv2 = ConvBNLayer(out_channels, out_channels, 3, padding=1, act=act)
        self.shortcut = shortcut and in_channels == out_channels
        self.use_alpha = use_alpha

        if use_alpha:
            self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection and alpha weighting."""
        output_tensor = self.conv1(input_tensor)
        output_tensor = self.conv2(output_tensor)
        if self.shortcut:
            if self.use_alpha:
                return self.alpha * input_tensor + output_tensor
            return input_tensor + output_tensor
        return output_tensor


class SPP(nn.Module):
    """Spatial Pyramid Pooling module for multi-scale feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: Sequence[int],
        act: ActivationType = "swish",
    ):
        super().__init__()
        self.pool = nn.ModuleList([nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2) for size in pool_size])
        self.conv = ConvBNLayer(in_channels, out_channels, kernel_size, padding=kernel_size // 2, act=act)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale pooling and feature fusion."""
        outs = [input_tensor]
        for pool in self.pool:
            outs.append(pool(input_tensor))
        combined_features = torch.cat(outs, dim=1)
        return self.conv(combined_features)


class CSPStage(nn.Module):
    """Cross Stage Partial (CSP) stage for efficient feature learning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        act: ActivationType = "swish",
        spp: bool = False,
        use_alpha: bool = False,
    ):
        super().__init__()

        mid_channels = int(out_channels // 2)
        self.conv1 = ConvBNLayer(in_channels, mid_channels, 1, act=act)
        self.conv2 = ConvBNLayer(in_channels, mid_channels, 1, act=act)
        self.convs = self._build_blocks(mid_channels, num_blocks, act, spp, use_alpha)
        self.conv3 = ConvBNLayer(mid_channels * 2, out_channels, 1, act=act)

    def _build_blocks(self, mid_channels: int, num_blocks: int, act: str, spp: bool, use_alpha: bool) -> nn.Sequential:
        """Build the sequence of blocks for the CSP stage."""
        convs = nn.Sequential()
        next_ch_in = mid_channels

        for block_idx in range(num_blocks):
            block = BasicBlock(next_ch_in, mid_channels, act=act, shortcut=False, use_alpha=use_alpha)
            convs.add_module(str(block_idx), block)

            # Add SPP module at the middle of the stage if requested
            if block_idx == (num_blocks - 1) // 2 and spp:
                convs.add_module("spp", SPP(mid_channels * 4, mid_channels, 1, (5, 9, 13), act=act))
            next_ch_in = mid_channels

        return convs

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through CSP stage with feature splitting and merging."""
        y1 = self.conv1(input_tensor)
        y2 = self.conv2(input_tensor)
        y2 = self.convs(y2)
        combined_features = torch.cat([y1, y2], dim=1)
        return self.conv3(combined_features)


class CustomCSPPAN(nn.Module):
    """CSP-PAN neck with consistent shallow→deep ordering.

    Input: [P3, P4, P5] features (shallow → deep, increasing stride)
    Output: [P3', P4', P5'] features (same order, enhanced)
    Configuration: All channel specs follow shallow → deep order
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (256, 512, 1024),  # P3, P4, P5 (shallow → deep)
        out_channels: Sequence[int] = (192, 384, 768),  # P3', P4', P5' (shallow → deep)
        act: ActivationType = "swish",
        stage_num: int = 1,
        block_num: int = 3,
        spp: bool = True,
        use_alpha: bool = False,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
    ):
        super().__init__()

        # Validate input consistency
        if len(in_channels) != len(out_channels):
            raise ValueError(
                f"in_channels and out_channels must have same length: {len(in_channels)} vs {len(out_channels)}"
            )

        # Apply multipliers
        self.out_channels = [max(round(channels * width_mult), 1) for channels in out_channels]
        self.block_num = max(round(block_num * depth_mult), 1)
        self.num_levels = len(in_channels)
        self.act = act
        self.stage_num = stage_num
        self.spp = spp
        self.use_alpha = use_alpha

        # Store for shape specification
        self.in_channels = list(in_channels)

        # Initialize FPN and PAN components
        self._init_fpn_stages()
        self._init_pan_stages()

    def _init_fpn_stages(self) -> None:
        """Initialize FPN stages in deep→shallow processing order.

        FPN processes from deep to shallow (P5 → P4 → P3) with upsampling.
        """
        fpn_stages = []
        fpn_routes = []

        # Process in reverse order: P5, P4, P3 (deep → shallow)
        reversed_in = self.in_channels[::-1]  # [1024, 512, 256]
        reversed_out = self.out_channels[::-1]  # [768, 384, 192]

        ch_pre = 0
        for level_idx, (ch_in, ch_out) in enumerate(zip(reversed_in, reversed_out, strict=True)):
            # Add routing channels from previous level
            if level_idx > 0:
                ch_in += ch_pre // 2

            # Create FPN stage (SPP only for deepest level, i.e., level_idx==0 which is P5)
            stage = self._create_stage(ch_in, ch_out, spp=(level_idx == 0 and self.spp))
            fpn_stages.append(stage)

            # Create routing layer for upsampling (except for last level)
            if level_idx < self.num_levels - 1:
                fpn_routes.append(
                    ConvBNLayer(
                        in_channels=ch_out, out_channels=ch_out // 2, kernel_size=1, stride=1, padding=0, act=self.act
                    )
                )

            ch_pre = ch_out

        self.fpn_stages = nn.ModuleList(fpn_stages)
        self.fpn_routes = nn.ModuleList(fpn_routes)

    def _init_pan_stages(self) -> None:
        """Initialize PAN stages in shallow→deep processing order.

        PAN processes from shallow to deep (P3 → P4 → P5) with downsampling.
        """
        pan_stages = []
        pan_routes = []

        # Process P3→P4, P4→P5 (shallow → deep)
        for level_idx in range(self.num_levels - 1):
            # Create downsampling routing layer
            pan_routes.append(
                ConvBNLayer(
                    in_channels=self.out_channels[level_idx],  # From shallower level
                    out_channels=self.out_channels[level_idx],  # Same channels
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act=self.act,
                )
            )

            # Create PAN stage (combine current + downsampled)
            ch_in = self.out_channels[level_idx + 1] + self.out_channels[level_idx]  # Deep + shallow
            ch_out = self.out_channels[level_idx + 1]  # Deep level output
            stage = self._create_stage(ch_in, ch_out, spp=False)  # No SPP in PAN
            pan_stages.append(stage)

        self.pan_stages = nn.ModuleList(pan_stages)
        self.pan_routes = nn.ModuleList(pan_routes)

    def _create_stage(self, in_channels: int, out_channels: int, spp: bool = False) -> nn.Sequential:
        """Create a stage with multiple CSP blocks."""
        stage = nn.Sequential()

        for stage_idx in range(self.stage_num):
            input_channels = in_channels if stage_idx == 0 else out_channels
            stage_layer = CSPStage(
                input_channels, out_channels, self.block_num, act=self.act, spp=spp, use_alpha=self.use_alpha
            )
            stage.add_module(str(stage_idx), stage_layer)

        return stage

    def _forward_fpn(self, blocks: list[torch.Tensor]) -> list[torch.Tensor]:
        """FPN forward pass: deep→shallow with upsampling.

        Args:
            blocks: [P5, P4, P3] (deep → shallow)

        Returns:
            fpn_feats: [FPN_P5, FPN_P4, FPN_P3] (deep → shallow)
        """
        fpn_feats = []
        route = None

        for level_idx, block in enumerate(blocks):
            # Concatenate with upsampled features from deeper level
            if level_idx > 0:
                route = F.interpolate(route, scale_factor=2.0, mode="nearest")
                block = torch.cat([route, block], dim=1)

            # Process through FPN stage
            route = self.fpn_stages[level_idx](block)
            fpn_feats.append(route)

            # Prepare route for next level (except for last level)
            if level_idx < self.num_levels - 1:
                route = self.fpn_routes[level_idx](route)

        return fpn_feats

    def _forward_pan(self, fpn_feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """PAN forward pass: shallow→deep with downsampling.

        Args:
            fpn_feats: [FPN_P5, FPN_P4, FPN_P3] (deep → shallow)

        Returns:
            pan_feats: [PAN_P3, PAN_P4, PAN_P5] (shallow → deep)
        """
        # Start with reversed FPN features: [FPN_P3, FPN_P4, FPN_P5]
        fpn_reversed = fpn_feats[::-1]
        pan_feats = [fpn_reversed[0]]  # Start with P3
        route = fpn_reversed[0]

        # Process shallow → deep: P3→P4, P4→P5
        for level_idx in range(self.num_levels - 1):
            deep_feat = fpn_reversed[level_idx + 1]  # Deeper FPN feature

            # Downsample current level and concatenate
            route = self.pan_routes[level_idx](route)
            combined = torch.cat([route, deep_feat], dim=1)

            # Process through PAN stage
            route = self.pan_stages[level_idx](combined)
            pan_feats.append(route)

        return pan_feats

    def forward(self, blocks: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass with consistent shallow→deep ordering.

        Args:
            blocks: [P3, P4, P5] backbone features (shallow → deep, standard order)

        Returns:
            [P3', P4', P5'] enhanced features (same order as input)
        """
        # Reverse for FPN processing: [P5, P4, P3] (deep → shallow)
        blocks_reversed = blocks[::-1]

        # FPN: deep → shallow processing
        fpn_feats = self._forward_fpn(blocks_reversed)

        # PAN: shallow → deep processing, returns in standard order
        pan_feats = self._forward_pan(fpn_feats)

        return pan_feats

    @property
    def out_channels_list(self) -> list[int]:
        """Get output channel configuration in shallow→deep order."""
        return self.out_channels
