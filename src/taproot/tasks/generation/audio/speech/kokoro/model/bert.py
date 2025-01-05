import torch

from typing import Optional

from transformers import AlbertConfig, AlbertModel # type: ignore[import-untyped]

__all__ = ["KokoroAlbert"]

class KokoroAlbert(AlbertModel): # type: ignore[misc]
    """
    A thin wrapper around the AlbertModel class that provides a more
    convenient interface for instantiation in a vanilla PyTorch model.
    """
    def __init__(
        self,
        vocab_size: int = 178,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 512,
        num_hidden_layers: int = 12,
        dropout: float = 0.1,
    ) -> None:
        """
        :param vocab_size: Vocabulary size.
        :param hidden_size: Hidden size.
        :param num_attention_heads: Number of attention heads.
        :param intermediate_size: Intermediate size.
        :param max_position_embeddings: Maximum position embeddings.
        :param num_hidden_layers: Number of hidden layers.
        :param dropout: Dropout rate.
        """
        super().__init__(
            AlbertConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                max_position_embeddings=max_position_embeddings,
                num_hidden_layers=num_hidden_layers,
                dropout=dropout,
            )
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        :param input_ids: Input token IDs. Shape (batch_size, seq_len).
        :param attention_mask: Attention mask. Shape (batch_size, seq_len).
        :return: Output tensor. Shape (batch_size, seq_len, hidden_size).
        """
        return super().forward( # type: ignore[no-any-return]
            input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
