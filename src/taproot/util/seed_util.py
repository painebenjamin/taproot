import os
import random

from typing import Optional, Union

__all__ = [
    "seed_all",
    "seed_everything",
    "get_seed",
    "SeedType"
]

SeedType = Optional[Union[int, str]]

def get_seed(seed: SeedType=None) -> int:
    """
    Returns a seed value.
    """
    if seed is None:
        try:
            seed = int(os.environ["GLOBAL_SEED"])
        except:
            seed = random.randint(0x10000000, 0xFFFFFFFF)
    elif isinstance(seed, str):
        seed = int(seed)
    elif not isinstance(seed, int):
        raise ValueError("Seed must be an integer or a string.")
    return seed

def seed_all(seed: SeedType=None) -> int:
    """
    Seeds all random number generators.
    `seed_everything` is an alias for `seed_all`.
    """
    seed = get_seed(seed)
    os.environ["GLOBAL_SEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    return seed

seed_everything = seed_all # Alias
