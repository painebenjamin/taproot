# Adapted from https://github.com/Zyphra/Zonos
import torch
import torch.nn.functional as F

__all__ = ["apply_delay_pattern", "revert_delay_pattern"]

def apply_delay_pattern(codes: torch.Tensor, mask_token: int) -> torch.Tensor:
    """
    Apply delay pattern to the codes.
    :param codes: torch.Tensor, shape (batch_size, n_q, seq_len)
    :param mask_token: int, the mask token
    :return: torch.Tensor, shape (batch_size, n_q, seq_len)
    """
    codes = F.pad(codes, (0, codes.shape[1]), value=mask_token)
    return torch.stack([codes[:, k].roll(k + 1) for k in range(codes.shape[1])], dim=1)

def revert_delay_pattern(codes: torch.Tensor) -> torch.Tensor:
    """
    Revert the delay pattern from the codes.
    :param codes: torch.Tensor, shape (batch_size, n_q, seq_len)
    :return: torch.Tensor, shape (batch_size, n_q, seq_len)
    """
    _, n_q, seq_len = codes.shape
    return torch.stack([codes[:, k, k + 1 : seq_len - n_q + k + 1] for k in range(n_q)], dim=1)
