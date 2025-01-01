from __future__ import annotations

from typing import Union, Optional, Tuple, TYPE_CHECKING

from ...constants import *

if TYPE_CHECKING:
    from torch import Tensor
    from .compel_util import TokenizerType, TextModelType

__all__ = ["encode_prompt_for_model"]

def encode_prompt_for_model(
    prompt: str,
    tokenizer: TokenizerType,
    text_encoder: TextModelType,
    model_type: Optional[DIFFUSERS_MODEL_TYPE_LITERAL]=None,
    device: Optional[str]=None,
    clip_skip: Optional[int]=None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Encodes a text prompt using compel.
    """
    from transformers import T5Tokenizer, T5TokenizerFast # type: ignore[import-untyped]
    from compel.embeddings_provider import ReturnedEmbeddingsType # type: ignore[import-untyped]
    from .compel_util import PromptEncoder

    requires_pooled = False
    if model_type in ["sdxl", "sd3", "flux"]:
        return_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
        requires_pooled = not isinstance(tokenizer, (T5Tokenizer, T5TokenizerFast))
    elif clip_skip:
        return_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED 
    else:
        return_type = ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED

    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.pad_token_id

    compel = PromptEncoder(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        returned_embeddings_type=return_type,
        requires_pooled=requires_pooled
    )

    compel.clip_skip = 0 if not clip_skip else clip_skip

    if requires_pooled:
        embeds, pooled_embeds = compel([prompt])
        if device:
            embeds = embeds.to(device)
            pooled_embeds = pooled_embeds.to(device)
        return embeds, pooled_embeds
    else:
        embeds = compel([prompt])
        if device:
            embeds = embeds.to(device)
        return embeds # type: ignore[no-any-return]
