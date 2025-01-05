import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LinearNorm",
    "LayerNorm"
]

class LinearNorm(nn.Module):
    """
    LinearNorm is a linear layer with xavier uniform initialization.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool=True,
        w_init_gain: str="linear"
    ) -> None:
        """
        :param in_dim: input dimension
        :param out_dim: output dimension
        :param bias: whether to use bias
        :param w_init_gain: weight initialization gain
        """
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain) # type: ignore[no-untyped-call]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor [batch_size, *, in_dim]
        :return: output tensor [batch_size, *, out_dim]
        """
        return self.linear_layer(x) # type: ignore[no-any-return]

class LayerNorm(nn.Module):
    """
    A learnable layer normalization module.
    """
    def __init__(
        self,
        channels: int,
        eps: float=1e-5
    ) -> None:
        """
        :param channels: number of channels
        :param eps: epsilon value for numerical stability
        """
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor [batch_size, channels, *]
        :return: output tensor [batch_size, channels, *]
        """
        x = x.transpose(1, -1)
        x = F.layer_norm(
            x,
            (self.channels,),
            self.gamma,
            self.beta,
            self.eps
        )
        return x.transpose(1, -1)
