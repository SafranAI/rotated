import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility across all libraries.

    Note:
        Setting deterministic=True may reduce performance but ensures complete reproducibility.
        Some PyTorch operations don't have deterministic implementations and will raise errors.
    """
    # Python's built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        # Enable cudnn benchmark for better performance (non-deterministic)
        torch.backends.cudnn.benchmark = True
