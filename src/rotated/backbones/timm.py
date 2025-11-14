from collections.abc import Sequence

import torch

from rotated.backbones.base import Backbone

try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class TimmBackbone(Backbone):
    """TIMM backbone wrapper with consistent feature extraction interface.

    Args:
        model_name: TIMM model name (e.g., 'resnet50', 'efficientnet_b0')
        pretrained: Whether to load ImageNet pre-trained weights
        return_levels: Which feature levels to return (0=highest res, 4=lowest res)
        **kwargs: Additional arguments passed to timm.create_model
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        return_levels: Sequence[int] = (2, 3, 4),  # Typically stride 8, 16, 32
        **kwargs,
    ):
        super().__init__(return_levels=return_levels)

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for TimmBackbone. Install with: pip install timm")

        if not model_name:
            raise ValueError("model_name cannot be empty")

        if not return_levels:
            raise ValueError("return_levels cannot be empty")

        if any(level < 0 for level in return_levels):
            raise ValueError("All return levels must be non-negative")

        self.model_name = model_name
        self.return_levels = list(return_levels)

        # Create timm model with feature extraction enabled
        self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True, **kwargs)
        self.feature_info = self.model.feature_info

        max_level = len(self.feature_info) - 1
        if any(level > max_level for level in return_levels):
            raise ValueError(
                f"Return levels {return_levels} contain indices > {max_level} "
                f"(model has {len(self.feature_info)} feature levels)"
            )

        # Cache output channel information
        self._out_channels = [self.feature_info[level_idx]["num_chs"] for level_idx in return_levels]
        self._out_strides = [self.feature_info[level_idx]["reduction"] for level_idx in return_levels]

    def forward(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
        """Extract features at specified levels."""
        all_features = self.model(input_tensor)
        return [all_features[level_idx] for level_idx in self.return_levels]

    @torch.jit.unused
    @property
    def out_channels(self) -> list[int]:
        return self._out_channels.copy()

    @torch.jit.unused
    @property
    def out_strides(self) -> list[int]:
        return self._out_strides.copy()
