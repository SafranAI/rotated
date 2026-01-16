"""Utilities for backbone pretrained weight loading and model creation."""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import warnings

import torch

if TYPE_CHECKING:
    from rotated.backbones.csp_resnet import CSPResNet

# CSPResNet variant configurations
CSPRESNET_CONFIGS = {
    "s": {"depth_mult": 0.33, "width_mult": 0.50},
    "m": {"depth_mult": 0.67, "width_mult": 0.75},
    "l": {"depth_mult": 1.00, "width_mult": 1.00},
    "x": {"depth_mult": 1.33, "width_mult": 1.25},
}

# GitHub Release URLs for pretrained weights
CSPRESNET_PRETRAINED_URLS = {
    "s": "https://github.com/SafranAI/rotated/releases/download/v0.2.0/CSPResNetb_s_pretrained.pth",
    "m": "https://github.com/SafranAI/rotated/releases/download/v0.2.0/CSPResNetb_m_pretrained.pth",
    "l": "https://github.com/SafranAI/rotated/releases/download/v0.2.0/CSPResNetb_l_pretrained.pth",
    "x": "https://github.com/SafranAI/rotated/releases/download/v0.2.0/CSPResNetb_x_pretrained.pth",
}


def _load_pretrained_weights(
    url: str,
    cache_dir: str | None = None,
    progress: bool = True,
) -> dict:
    """Load pretrained weights from URL with caching.

    Args:
        url: URL to download weights from
        cache_dir: Directory to cache downloaded weights.
                   Defaults to ~/.cache/torch/hub/checkpoints/
        progress: Show download progress bar

    Returns:
        State dict with pretrained weights
    """
    if cache_dir is None:
        cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Download or load from cache
    state_dict = torch.hub.load_state_dict_from_url(
        url,
        model_dir=cache_dir,
        progress=progress,
        map_location="cpu",
    )

    return state_dict


def create_csp_resnet(
    variant: Literal["s", "m", "l", "x"],
    pretrained: bool = False,
    cache_dir: str | None = None,
    **kwargs,
) -> "CSPResNet":
    """Create a CSPResNet backbone with optional pretrained weights.

    This factory creates CSPResNet models with the default configuration used by PPYOLOE and PPYOLOE-R detectors.
    For custom configurations, instantiate CSPResNet directly.

    Args:
        variant: Model variant - 's', 'm', 'l', or 'x'
        pretrained: If True, loads pretrained weights from GitHub releases
        cache_dir: Directory to cache downloaded weights.
                   Defaults to ~/.cache/torch/hub/checkpoints/
        **kwargs: Additional arguments override defaults
                  (e.g., return_levels, in_chans, use_checkpoint)

    Returns:
        CSPResNet model initialized on CPU

    Raises:
        ValueError: If variant is not supported or pretrained weights are unavailable

    Example:
        >>> from rotated.backbones import create_csp_resnet
        >>>
        >>> # Create small model with pretrained weights
        >>> model = create_csp_resnet("s", pretrained=True)
        >>>
        >>> # Create large model without pretrained weights
        >>> model = create_csp_resnet("l", pretrained=False)
        >>>
        >>> # Create with custom return levels
        >>> model = create_csp_resnet("m", pretrained=True, return_levels=[0, 1, 2, 3])
    """
    from rotated.backbones.csp_resnet import CSPResNet

    if variant not in CSPRESNET_CONFIGS:
        raise ValueError(f"Variant '{variant}' not supported. Available variants: {list(CSPRESNET_CONFIGS.keys())}")

    config = CSPRESNET_CONFIGS[variant]

    # Build model with default parameters + variant-specific multipliers
    model = CSPResNet(
        layers=[3, 6, 6, 3],
        channels=[64, 128, 256, 512, 1024],
        return_levels=[1, 2, 3],
        use_large_stem=True,
        act="swish",
        depth_mult=config["depth_mult"],
        width_mult=config["width_mult"],
        use_alpha=True,
        **kwargs,  # Allow overrides
    )

    if pretrained:
        if variant not in CSPRESNET_PRETRAINED_URLS:
            raise ValueError(
                f"No pretrained weights available for variant '{variant}'. "
                f"Available variants: {list(CSPRESNET_PRETRAINED_URLS.keys())}"
            )

        url = CSPRESNET_PRETRAINED_URLS[variant]
        state_dict = _load_pretrained_weights(url, cache_dir=cache_dir)

        # Load with strict=False to allow parameter overrides
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys or unexpected_keys:
            warnings.warn(
                f"Pretrained weights loaded with mismatches. "
                f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}. "
                f"This may be expected if you passed custom parameters (e.g., return_levels, in_chans).",
                UserWarning,
                stacklevel=2,
            )

    return model
