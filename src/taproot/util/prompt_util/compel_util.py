from __future__ import annotations

import math
import torch

from typing import Optional, Union, List, Sequence, Callable, Tuple

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
        returned_embeddings_type: ReturnedEmbeddingsType=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
        requires_pooled: Union[bool, Sequence[bool]]=False,
        max_sequence_length: Optional[int]=None,
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
            self.conditioning_provider = FlexibleEmbeddingsProviderMulti(
                tokenizers=tokenizer,
                text_encoders=text_encoder,
                textual_inversion_manager=textual_inversion_manager,
                dtype_for_device_getter=dtype_for_device_getter,
                truncate=truncate_long_prompts,
                padding_attention_mask_value = padding_attention_mask_value,
                downweight_mode=downweight_mode,
                returned_embeddings_type=returned_embeddings_type,
                requires_pooled_mask=[requires_pooled] if isinstance(requires_pooled, bool) else requires_pooled, # type: ignore[arg-type]
                max_sequence_length=max_sequence_length,
                device=device
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
                max_sequence_length=max_sequence_length,
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

    @property
    def max_sequence_length(self) -> Optional[int]:
        """
        Passes max_sequence_length through to conditioning provider
        """
        return getattr(self.conditioning_provider, "max_sequence_length", None)

    @max_sequence_length.setter
    def max_sequence_length(self, max_length: Optional[int]) -> None:
        """
        Passes max_sequence_length through to conditioning provider
        """
        setattr(self.conditioning_provider, "max_sequence_length", max_length)

class FlexibleEmbeddingsProvider(EmbeddingsProvider): # type: ignore[misc]
    """
    Extend compel slightly to permit multiple CLIP skip levels and make encoding more generic
    """
    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection], # convert a list of int token ids to a tensor of embeddings
        textual_inversion_manager: BaseTextualInversionManager = None,
        dtype_for_device_getter: Callable[[torch.device], torch.dtype] = default_get_dtype_for_device,
        truncate: bool = True,
        padding_attention_mask_value: int = 1,
        downweight_mode: DownweightMode = DownweightMode.MASK,
        returned_embeddings_type: ReturnedEmbeddingsType = ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
        max_sequence_length: Optional[int] = None,
        clip_skip: int = 0,
        device: Optional[str] = None,
     ) -> None:
        """
        Copied from src/compel/embeddings_provider.py
        """
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            textual_inversion_manager=textual_inversion_manager,
            dtype_for_device_getter=dtype_for_device_getter,
            truncate=truncate,
            padding_attention_mask_value=padding_attention_mask_value,
            downweight_mode=downweight_mode,
            returned_embeddings_type=returned_embeddings_type,
            device=device
        )
        self.clip_skip = clip_skip
        self.max_sequence_length = max_sequence_length

    @property
    def max_token_count(self) -> int:
        """
        Gets max token count from tokenizer
        """
        if self.max_sequence_length is None:
            return self.tokenizer.model_max_length # type: ignore[no-any-return]
        return min( # type: ignore[no-any-return]
            self.max_sequence_length,
            self.tokenizer.model_max_length
        )

    def get_embeddings_for_weighted_prompt_fragments(
        self,
        text_batch: List[List[str]],
        fragment_weights_batch: List[List[float]],
        should_return_tokens: bool = False,
        device: Union[torch.device, str]="cpu",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        :param text_batch: A list of fragments of text to which different weights are to be applied.
        :param fragment_weights_batch: A list of weights, one for each entry in `fragments`.
        :param should_return_tokens: If True, return a tuple of (embeddings, tokens), otherwise just return embeddings.
        :param device: Where to put the constructed tensor(s)
        :return: A tensor of shape `[1, 77, token_dim]` containing weighted embeddings where token_dim is 768 for SD1
                    and 1280 for SD2
        """
        if len(text_batch) != len(fragment_weights_batch):
            raise ValueError(
                f"lengths of text and fragment_weights lists are not the same "+
                f"({len(text_batch)} != {len(fragment_weights_batch)})"
            )

        batch_z = None
        batch_tokens = None
        for fragments, weights in zip(text_batch, fragment_weights_batch):
            # First, weight tokens in individual fragments by scaling the feature vectors as requested (effectively
            # applying a multiplier to the CFG scale on a per-token basis).
            # For tokens weighted<1, intuitively we want SD to become not merely *less* interested in the concept
            # captured by the fragment but actually *dis*interested in it (a 0.01 interest in "red" is still an active
            # interest, however small, in redness; what the user probably intends when they attach the number 0.01 to
            # "red" is to tell SD that it should almost completely *ignore* redness).
            # To do this, the embedding is lerped away from base_embedding in the direction of an embedding for a prompt
            # string from which the low-weighted fragment has been simply removed. The closer the weight is to zero, the
            # closer the resulting embedding is to an embedding for a prompt that simply lacks this fragment.

            # handle weights >=1
            tokens, per_token_weights, mask = self.get_token_ids_and_expand_weights(fragments, weights, device=device)
            base_embedding = self.build_weighted_embedding_tensor(tokens, per_token_weights, mask, device=device)

            # this is our starting point
            embeddings = base_embedding.unsqueeze(0)
            per_embedding_weights = [1.0]

            # now handle weights <1
            # Do this by building extra embeddings tensors that lack the words being <1 weighted. These will be lerped
            # with the embeddings tensors that have the words, such that if the weight of a word is 0.5, the resulting
            # embedding will be exactly half-way between the unweighted prompt and the prompt with the <1 weighted words
            # removed.
            # e.g. for "mountain:1 man:0.5", intuitively the "man" should be "half-gone". therefore, append an embedding
            # for "mountain" (i.e. without "man") to the already-produced embedding for "mountain man", and weight it
            # such that the resulting lerped embedding is exactly half-way between "mountain man" and "mountain".
            fragment_token_index_ranges = self._get_token_ranges_for_fragments(tokens.tolist(), fragments)

            for index in range(len(fragment_token_index_ranges)):
                fragment_weight = weights[index]
                if fragment_weight < 1:
                    if self.downweight_mode == DownweightMode.MASK:
                        fragment_start_token_id, fragment_end_token_id = fragment_token_index_ranges[index]
                        # mask out this fragment
                        mask_without_fragment = mask.clone()
                        mask_without_fragment[fragment_start_token_id:fragment_end_token_id+1] = 0
                        if not self.truncate_to_model_max_length:
                            # but don't mask chunk-delimiting eos/bos markers
                            mask_without_fragment[0::self.max_token_count] = 1
                            mask_without_fragment[self.max_token_count-1::self.max_token_count] = 1

                        embedding_without_this = self.build_weighted_embedding_tensor(
                            tokens,
                            per_token_weights,
                            mask_without_fragment,
                            device=device
                        )
                    else:
                        fragments_without_this = fragments[0:index] + fragments[index+1:]
                        weights_without_this = weights[0:index] + weights[index+1:]
                        (
                            tokens_without_fragment,
                            per_token_weights_without_fragment,
                            mask_without_fragment
                        ) = self.get_token_ids_and_expand_weights(
                            fragments_without_this,
                            weights_without_this,
                            device=device
                        )

                        embedding_without_this = self.build_weighted_embedding_tensor(
                            tokens_without_fragment,
                            per_token_weights_without_fragment,
                            device=device
                        )

                    embeddings = torch.cat((embeddings, embedding_without_this.unsqueeze(0)), dim=1)
                    # weight of the embedding *without* this fragment gets *stronger* as its weight approaches 0
                    # if fragment_weight = 0, basically we want embedding_without_this to completely overwhelm base_embedding
                    # therefore:
                    # fragment_weight = 1: we are at base_z => lerp weight 0
                    # fragment_weight = 0.5: we are halfway between base_z and here => lerp weight 1
                    # fragment_weight = 0: we're now entirely overriding base_z ==> lerp weight inf
                    # so let's use tan(), because:
                    # tan is 0.0 at 0,
                    #        1.0 at PI/4, and
                    #        inf at PI/2
                    # -> tan((1-weight)*PI/2) should give us ideal lerp weights
                    epsilon = 1e-5
                    fragment_weight = max(epsilon, fragment_weight) # inf is bad
                    embedding_lerp_weight = math.tan((1.0 - fragment_weight) * math.pi / 2)

                    per_embedding_weights.append(embedding_lerp_weight)

            lerped_embeddings = self.apply_embedding_weights(embeddings, per_embedding_weights, normalize=True).squeeze(0)

            # append to batch
            batch_z = lerped_embeddings.unsqueeze(0) if batch_z is None else torch.cat([batch_z, lerped_embeddings.unsqueeze(0)], dim=1)
            batch_tokens = tokens.unsqueeze(0) if batch_tokens is None else torch.cat([batch_tokens, tokens.unsqueeze(0)], dim=1)

        # should have shape (B, 77, 768)

        if should_return_tokens:
            return batch_z, batch_tokens # type: ignore[return-value]
        else:
            return batch_z # type: ignore[return-value]

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

class FlexibleEmbeddingsProviderMulti(EmbeddingsProviderMulti): # type: ignore[misc]
    """
    Extend compel slightly to permit multiple CLIP skip levels and make encoding more generic
    """
    def __init__(
        self,
        tokenizers: CLIPTokenizer,
        text_encoders: Union[CLIPTextModel, CLIPTextModelWithProjection], # convert a list of int token ids to a tensor of embeddings
        textual_inversion_manager: BaseTextualInversionManager=None,
        dtype_for_device_getter: Callable[[torch.device], torch.dtype]=default_get_dtype_for_device,
        truncate: bool=True,
        padding_attention_mask_value: int=1,
        downweight_mode: DownweightMode=DownweightMode.MASK,
        returned_embeddings_type: Union[List[ReturnedEmbeddingsType], ReturnedEmbeddingsType]=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
        requires_pooled_mask: List[bool]=[],
        max_sequence_length: Optional[int]=None,
        clip_skip: int=0,
        device: Optional[str] = None,
    ) -> None:
        """
        Copied from src/compel/embeddings_provider.py
        """
        returned_embeddings_type = (
            len(text_encoders) * [returned_embeddings_type]
            if not isinstance(returned_embeddings_type, (list,tuple))
            else returned_embeddings_type
        )
        self.embedding_providers = [
            FlexibleEmbeddingsProvider(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                textual_inversion_manager=textual_inversion_manager,
                dtype_for_device_getter=dtype_for_device_getter,
                truncate=truncate,
                padding_attention_mask_value=padding_attention_mask_value,
                downweight_mode=downweight_mode,
                returned_embeddings_type=returned_embeddings_type,
                max_sequence_length=max_sequence_length,
                clip_skip=clip_skip,
                device=device
            )
            for tokenizer, text_encoder, returned_embeddings_type
            in zip(tokenizers, text_encoders, returned_embeddings_type)
        ]
        self.requires_pooled_mask = requires_pooled_mask

    @property
    def max_token_count(self) -> int:
        """
        Gets max token count from tokenizer
        """
        return min([provider.max_token_count for provider in self.embedding_providers])

    @property
    def clip_skip(self) -> int:
        """
        Gets clip_skip from first provider
        """
        return self.embedding_providers[0].clip_skip

    @clip_skip.setter
    def clip_skip(self, skip: int) -> None:
        """
        Sets clip_skip for all providers
        """
        for provider in self.embedding_providers:
            provider.clip_skip = skip

    @property
    def max_sequence_length(self) -> Optional[int]:
        """
        Gets max_sequence_length from first provider
        """
        return self.embedding_providers[0].max_sequence_length

    @max_sequence_length.setter
    def max_sequence_length(self, max_length: Optional[int]) -> None:
        """
        Sets max_sequence_length for all providers
        """
        for provider in self.embedding_providers:
            provider.max_sequence_length = max_length
