from collections.abc import Sequence
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from rotated.layers import ConvBNLayer


def _get_clones(module: nn.Module, num_copies: int) -> nn.ModuleList:
    """Create N identical copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_copies)])


class BasicBlock(nn.Module):
    """Basic residual block with optional shortcut connection."""

    def __init__(self, in_channels: int, out_channels: int, act: str = "swish", shortcut: bool = True):
        super().__init__()
        self.conv1 = ConvBNLayer(in_channels, out_channels, 3, padding=1, act=act)
        self.conv2 = ConvBNLayer(out_channels, out_channels, 3, padding=1, act=act)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection."""
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        return y


class SPP(nn.Module):
    """Spatial Pyramid Pooling module for multi-scale feature extraction."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, pool_size: Sequence[int], act: str = "swish"
    ):
        super().__init__()
        self.pool = nn.ModuleList([nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2) for size in pool_size])
        self.conv = ConvBNLayer(in_channels, out_channels, kernel_size, padding=kernel_size // 2, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale pooling and feature fusion."""
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, dim=1)
        return self.conv(y)


class CSPStage(nn.Module):
    """Cross Stage Partial (CSP) stage for efficient feature learning."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, act: str = "swish", spp: bool = False):
        super().__init__()

        mid_channels = int(out_channels // 2)
        self.conv1 = ConvBNLayer(in_channels, mid_channels, 1, act=act)
        self.conv2 = ConvBNLayer(in_channels, mid_channels, 1, act=act)
        self.convs = self._build_blocks(mid_channels, num_blocks, act, spp)
        self.conv3 = ConvBNLayer(mid_channels * 2, out_channels, 1, act=act)

    def _build_blocks(self, mid_channels: int, num_blocks: int, act: str, spp: bool) -> nn.Sequential:
        """Build the sequence of blocks for the CSP stage."""
        convs = nn.Sequential()
        next_ch_in = mid_channels

        for i in range(num_blocks):
            block = BasicBlock(next_ch_in, mid_channels, act=act, shortcut=False)
            convs.add_module(str(i), block)

            # Add SPP module at the middle of the stage if requested
            if i == (num_blocks - 1) // 2 and spp:
                convs.add_module("spp", SPP(mid_channels * 4, mid_channels, 1, (5, 9, 13), act=act))
            next_ch_in = mid_channels

        return convs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CSP stage with feature splitting and merging."""
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.convs(y2)
        y = torch.cat([y1, y2], dim=1)
        return self.conv3(y)


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
        act: str = "swish",
        stage_num: int = 1,
        block_num: int = 3,
        spp: bool = True,
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
        self.out_channels = [max(round(c * width_mult), 1) for c in out_channels]
        self.block_num = max(round(block_num * depth_mult), 1)
        self.num_levels = len(in_channels)
        self.act = act
        self.stage_num = stage_num
        self.spp = spp

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
        for i, (ch_in, ch_out) in enumerate(zip(reversed_in, reversed_out, strict=True)):
            # Add routing channels from previous level
            if i > 0:
                ch_in += ch_pre // 2

            # Create FPN stage (SPP only for deepest level, i.e., i==0 which is P5)
            stage = self._create_stage(ch_in, ch_out, spp=(i == 0 and self.spp))
            fpn_stages.append(stage)

            # Create routing layer for upsampling (except for last level)
            if i < self.num_levels - 1:
                fpn_routes.append(
                    ConvBNLayer(
                        in_channels=ch_out, out_channels=ch_out // 2, filter_size=1, stride=1, padding=0, act=self.act
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
        for i in range(self.num_levels - 1):
            # Create downsampling routing layer
            pan_routes.append(
                ConvBNLayer(
                    in_channels=self.out_channels[i],  # From shallower level
                    out_channels=self.out_channels[i],  # Same channels
                    filter_size=3,
                    stride=2,
                    padding=1,
                    act=self.act,
                )
            )

            # Create PAN stage (combine current + downsampled)
            ch_in = self.out_channels[i + 1] + self.out_channels[i]  # Deep + shallow
            ch_out = self.out_channels[i + 1]  # Deep level output
            stage = self._create_stage(ch_in, ch_out, spp=False)  # No SPP in PAN
            pan_stages.append(stage)

        self.pan_stages = nn.ModuleList(pan_stages)
        self.pan_routes = nn.ModuleList(pan_routes)

    def _create_stage(self, in_channels: int, out_channels: int, spp: bool = False) -> nn.Sequential:
        """Create a stage with multiple CSP blocks."""
        stage = nn.Sequential()

        for j in range(self.stage_num):
            input_channels = in_channels if j == 0 else out_channels
            stage_layer = CSPStage(input_channels, out_channels, self.block_num, act=self.act, spp=spp)
            stage.add_module(str(j), stage_layer)

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

        for i, block in enumerate(blocks):
            # Concatenate with upsampled features from deeper level
            if i > 0:
                route = F.interpolate(route, scale_factor=2.0, mode="nearest")
                block = torch.cat([route, block], dim=1)

            # Process through FPN stage
            route = self.fpn_stages[i](block)
            fpn_feats.append(route)

            # Prepare route for next level (except for last level)
            if i < self.num_levels - 1:
                route = self.fpn_routes[i](route)

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
        for i in range(self.num_levels - 1):
            deep_feat = fpn_reversed[i + 1]  # Deeper FPN feature

            # Downsample current level and concatenate
            route = self.pan_routes[i](route)
            combined = torch.cat([route, deep_feat], dim=1)

            # Process through PAN stage
            route = self.pan_stages[i](combined)
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


if __name__ == "__main__":
    print("Testing CustomCSPPAN")

    # Create model
    neck = CustomCSPPAN(
        in_channels=(256, 512, 1024),
        out_channels=(192, 384, 768),
        act="swish",
        spp=True,
    )

    # Test with realistic feature map sizes
    batch_size = 2
    img_size = 640

    # Create test inputs (shallow → deep: decreasing size, increasing channels)
    test_features = [
        torch.randn(batch_size, 256, img_size // 8, img_size // 8),  # P3: stride 8
        torch.randn(batch_size, 512, img_size // 16, img_size // 16),  # P4: stride 16
        torch.randn(batch_size, 1024, img_size // 32, img_size // 32),  # P5: stride 32
    ]

    # Forward pass
    neck.eval()
    with torch.no_grad():
        outputs = neck(test_features)

    # Verify outputs
    expected_channels = (192, 384, 768)
    for i, output in enumerate(outputs):
        assert output.shape[1] == expected_channels[i], f"Channel mismatch at P{i + 3}"

    print("Forward pass successful")
