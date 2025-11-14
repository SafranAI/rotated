# Adapted from HuggingFace Transformers
# https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py
# Apache 2.0 License

from collections.abc import Callable
from functools import lru_cache, wraps
from typing import Any, TypeVar


def is_torchdynamo_compiling() -> bool:
    """Check if currently compiling with torch.compile (dynamo).

    Importing torch._dynamo causes issues with PyTorch profiler, hence rather
    relying on torch.compiler.is_compiling() when possible (torch>=2.3).
    """
    try:
        import torch

        return torch.compiler.is_compiling()
    except Exception:
        try:
            import torch._dynamo as dynamo

            return dynamo.is_compiling()
        except Exception:
            return False


def is_torchdynamo_exporting() -> bool:
    """Check if currently exporting with torch.export (dynamo)."""
    try:
        import torch

        return torch.compiler.is_exporting()
    except Exception:
        try:
            import torch._dynamo as dynamo

            return dynamo.is_exporting()
        except Exception:
            return False


def is_torch_fx_proxy(x) -> bool:
    """Check if object is a torch.fx.Proxy."""
    try:
        import torch.fx

        return isinstance(x, torch.fx.Proxy)
    except Exception:
        return False


def is_jit_tracing() -> bool:
    """Check if currently tracing with torch.jit."""
    try:
        import torch

        return torch.jit.is_tracing()
    except Exception:
        return False


def is_jit_scripting() -> bool:
    """Check if currently scripting with torch.jit."""
    try:
        import torch

        return torch.jit.is_scripting()
    except Exception:
        return False


def is_tracing(tensor: Any = None) -> bool:
    """Check if tracing a graph with dynamo (compile or export), torch.jit, or torch.fx.

    Args:
        tensor: Optional tensor to check if it's a torch.fx.Proxy

    Returns:
        True if any tracing/compilation mode is active
    """
    _is_tracing = is_torchdynamo_compiling() or is_jit_tracing()
    if tensor is not None:
        _is_tracing |= is_torch_fx_proxy(tensor)
    return _is_tracing


F = TypeVar("F", bound=Callable[..., Any])


@wraps(lru_cache)
def compile_compatible_lru_cache(*lru_args: Any, **lru_kwargs: Any) -> Callable[[F], F]:
    """LRU cache decorator that disables caching during compilation/tracing.

    This decorator wraps functools.lru_cache but disables caching when:
    - torch.compile is compiling (torchdynamo)
    - torch.jit is scripting
    - torch.jit is tracing

    This ensures the cached method works correctly with TorchScript and torch.compile.

    Args:
        *lru_args: Arguments passed to functools.lru_cache
        **lru_kwargs: Keyword arguments passed to functools.lru_cache

    Returns:
        Decorated function with compile-compatible caching

    Example:
        @compile_compatible_lru_cache(maxsize=32)
        def generate_anchors(self, height, width, device):
            # expensive computation
            return anchors
    """

    def decorator(func):
        func_with_cache = lru_cache(*lru_args, **lru_kwargs)(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Disable caching during any tracing/compilation mode
            if is_torchdynamo_compiling() or is_jit_scripting() or is_jit_tracing():
                return func(*args, **kwargs)

            return func_with_cache(*args, **kwargs)

        return wrapper

    return decorator
