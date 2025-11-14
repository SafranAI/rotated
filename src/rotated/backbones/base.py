from collections.abc import Sequence

import torch
import torch.nn as nn


class Backbone(nn.Module):
    """Base class for backbone networks with common properties."""

    def __init__(self, return_levels: Sequence[int]):
        super().__init__()
        self.return_levels = list(return_levels)

    @torch.jit.unused
    @property
    def out_channels(self) -> list[int]:
        """Get output channels for each return level."""
        raise NotImplementedError("Subclasses must implement this property")

    @torch.jit.unused
    @property
    def out_strides(self) -> list[int]:
        """Get output strides for each return level."""
        raise NotImplementedError("Subclasses must implement this property")
