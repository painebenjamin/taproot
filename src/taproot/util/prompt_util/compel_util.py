from __future__ import annotations

import torch

from typing import Optional, Union, List, Sequence, Callable

from compel import Compel, DownweightMode, BaseTextualInversionManager # type: ignore[import-untyped]
from compel.embeddings_provider import EmbeddingsProvider, EmbeddingsProviderMulti, ReturnedEmbeddingsType # type: ignore[import-untyped]

from transformers import ( # type: ignore[import-untyped]
    CLIPTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    T5EncoderModel,
    T5Tokenizer
)

__all__ = ["PromptEncoder"]

TokenizerType = Union[CLIPTokenizer, T5Tokenizer]
TextModelType = Union[CLIPTextModel, CLIPTextModelWithProjection, T5EncoderModel]

def default_get_dtype_for_device(device: torch.device) -> torch.dtype:
    """
    Format expected by compel
    """
    return torch.float32

class PromptEncoder(Compel): # type: ignore[misc]
    """
    Extend compel slightly to permit multiple CLIP skip levels and make encoding more generic
    """
    def __init__(
        self,
         tokenizer: Union[TokenizerType, Sequence[TokenizerType]],
         text_encoder: Union[TextModelType, Sequence[TextModelType]],
         textual_inversion_manager: Optional[BaseTextualInversionManager]=None,
         dtype_for_device_getter: Callable[[torch.device], torch.dtype]=default_get_dtype_for_device,
         truncate_long_prompts: bool=True,
         padding_attention_mask_value: int=1,
         downweight_mode: DownweightMode=DownweightMode.MASK,
         returned_embeddings_type: ReturnedEmbeddingsType = ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
         requires_pooled: Union[bool, Sequence[bool]]=False,
         device: Optional[str]=None
     ) -> None:
        """
        Copied from https://github.com/damian0815/compel/blob/main/src/compel/compel.py
        Modified slightly to change EmbeddingsProvider to FlexibleEmbeddingsProvider
        """
        if isinstance(tokenizer, (tuple, list)) and not isinstance(text_encoder, (tuple, list)):
            raise ValueError("Cannot provide list of tokenizers, but not of text encoders.")
        elif not isinstance(tokenizer, (tuple, list)) and isinstance(text_encoder, (tuple, list)):
            raise ValueError("Cannot provide list of text encoders, but not of tokenizers.")
        elif isinstance(tokenizer, (tuple, list)) and isinstance(text_encoder, (tuple, list)):
            self.conditioning_provider = EmbeddingsProviderMulti(
                tokenizers=tokenizer,
                text_encoders=text_encoder,
                textual_inversion_manager=textual_inversion_manager,
                dtype_for_device_getter=dtype_for_device_getter,
                truncate=truncate_long_prompts,
                padding_attention_mask_value = padding_attention_mask_value,
                downweight_mode=downweight_mode,
                returned_embeddings_type=returned_embeddings_type,
                requires_pooled_mask = requires_pooled
            )
        else:
            self.conditioning_provider = FlexibleEmbeddingsProvider(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                textual_inversion_manager=textual_inversion_manager,
                dtype_for_device_getter=dtype_for_device_getter,
                truncate=truncate_long_prompts,
                padding_attention_mask_value = padding_attention_mask_value,
                downweight_mode=None if isinstance(text_encoder, T5EncoderModel) else downweight_mode,
                returned_embeddings_type=returned_embeddings_type,
                device=device
            )
        self._device = device
        self.requires_pooled = requires_pooled

    @property
    def clip_skip(self) -> int:
        """
        Passes clip-skip through to conditioning provider
        """
        return getattr(self.conditioning_provider, "clip_skip", 0)

    @clip_skip.setter
    def clip_skip(self, skip: int) -> None:
        """
        Passes clip-skip through to conditioning provider
        """
        setattr(self.conditioning_provider, "clip_skip", skip)

class FlexibleEmbeddingsProvider(EmbeddingsProvider): # type: ignore[misc]
    """
    Extend compel slightly to permit multiple CLIP skip levels and make encoding more generic
    """
    clip_skip: int = 0

    def _encode_token_ids_to_embeddings(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        """
        Extends compels functionality to permit any level of clip skip and T5
        """
        from transformers import T5EncoderModel
        if isinstance(self.text_encoder, T5EncoderModel):
            return self.text_encoder( # type: ignore[no-any-return]
                input_ids=token_ids,
                attention_mask=attention_mask,
            )[0]

        needs_hidden_states = (
            self.returned_embeddings_type == ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED or
            self.returned_embeddings_type == ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
        )
        text_encoder_output = self.text_encoder(
            token_ids,
            attention_mask,
            output_hidden_states=needs_hidden_states,
            return_dict=True
        )
        if self.returned_embeddings_type is ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED:
            penultimate_hidden_state = text_encoder_output.hidden_states[-(self.clip_skip + 2)]
            return penultimate_hidden_state # type: ignore[no-any-return]
        elif self.returned_embeddings_type is ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED:
            penultimate_hidden_state = text_encoder_output.hidden_states[-(self.clip_skip + 1)]
            return self.text_encoder.text_model.final_layer_norm(penultimate_hidden_state) # type: ignore[no-any-return]
        elif self.returned_embeddings_type is ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED:
            # already normalized
            return text_encoder_output.last_hidden_state # type: ignore[no-any-return]

        assert False, f"unrecognized ReturnEmbeddingsType: {self.returned_embeddings_type}"

    def get_pooled_embeddings(
        self,
        texts: List[str],
        attention_mask: Optional[torch.Tensor]=None,
        device: Optional[str]=None
    ) -> Optional[torch.Tensor]:
        """
        Uses the generic way to get pooled embeddings
        """
        import torch
        device = device or self.device

        token_ids = self.get_token_ids(texts, padding="max_length", truncation_override=True)
        token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)

        return self.text_encoder(token_ids, attention_mask)[0] # type: ignore[no-any-return]
