import torch

from typing import Optional, Union

__all__ = ["normalize"]

def normalize(
    latent: torch.Tensor,
    target_min: Optional[Union[float, int, torch.Tensor]]=None,
    target_max: Optional[Union[float, int, torch.Tensor]]=None
) -> torch.Tensor:
    """
    Normalize a tensor `latent` between `target_min` and `target_max`.
    """
    min_val = latent.min()
    max_val = latent.max()
    
    if target_min is None:
        target_min = min_val
    if target_max is None:
        target_max = max_val
        
    normalized = (latent - min_val) / (max_val - min_val)
    scaled = normalized * (target_max - target_min) + target_min
    return scaled
