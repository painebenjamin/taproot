# Adapted from https://github.com/Zyphra/Zonos
from __future__ import annotations

import torch
import torch.nn as nn

from typing import Dict, Optional, Any
from mamba_ssm.utils.generation import InferenceParams # type: ignore[import-untyped]

from .backbone import ZonosBackbone
from .codebook import apply_delay_pattern, revert_delay_pattern
from .conditioning import PrefixConditioner
from .config import ZonosConfig
from .sampling import sample_from_logits

__all__ = ["Zonos"]

class Zonos(nn.Module):
    _cg_graph: Optional[torch.cuda.CUDAGraph]
    _cg_batch_size: Optional[int]
    _cg_input_ids: Optional[torch.Tensor]
    _cg_logits: Optional[torch.Tensor]
    _cg_inference_params: Optional[InferenceParams]
    _cg_scale: Optional[float]

    def __init__(
        self,
        config: ZonosConfig
    ) -> None:
        super().__init__()
        self.config = config
        self.eos_token_id = config.eos_token_id
        self.masked_token_id = config.masked_token_id
        self.backbone = ZonosBackbone(config.backbone)
        self.prefix_conditioner = PrefixConditioner(
            config.prefix_conditioner,
            config.backbone.d_model,
        )
        
        # TODO: pad to multiple of at least 8
        self.embeddings = nn.ModuleList([
            nn.Embedding(1026, self.config.backbone.d_model)
            for _ in range(self.config.n_codebooks)
        ])
        self.heads = nn.ModuleList([
            nn.Linear(self.config.backbone.d_model, 1025, bias=False)
            for _ in range(self.config.n_codebooks)
        ])

        self._cg_graph = None
        self._cg_batch_size = None
        self._cg_input_ids = None
        self._cg_logits = None
        self._cg_inference_params = None
        self._cg_scale = None

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return sum(emb(codes[:, i]) for i, emb in enumerate(self.embeddings)) # type: ignore[no-any-return]

    def apply_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(hidden_states) for head in self.heads], dim=1)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float
    ) -> torch.Tensor:
        """
        Pass `hidden_states` into `backbone` and `multi_head`, applying
        classifier-free guidance if `cfg_scale != 1.0`.
        """
        last_hidden_states = self.backbone(hidden_states, inference_params)[:, -1, :].unsqueeze(1)
        logits = self.apply_heads(last_hidden_states).squeeze(2).float()
        if cfg_scale != 1.0:
            cond_logits, uncond_logits = logits.chunk(2)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        return logits

    def decode_one_token(
        self,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        Single-step decode. Prepares the hidden states, possibly replicates them
        for CFG, and then delegates to `compute_logits`.

        Below we wrap this function with a simple CUDA Graph capturing mechanism,
        doing 3 warmup steps if needed and then capturing or replaying the graph.
        We only recapture if the batch size changes.
        """
        # TODO: support cfg_scale==1
        if cfg_scale == 1.0:
            hidden_states = self.embed_codes(input_ids)
            return self.compute_logits(hidden_states, inference_params, cfg_scale)

        bsz = input_ids.size(0)

        need_capture = (self._cg_graph is None) or (self._cg_batch_size != bsz)

        if need_capture:
            self._cg_graph = None

            self._cg_batch_size = bsz
            self._cg_inference_params = inference_params
            self._cg_scale = cfg_scale

            for _ in range(3):
                hidden_states = self.embed_codes(input_ids)
                hidden_states = hidden_states.repeat(2, 1, 1)  # because cfg != 1.0
                logits = self.compute_logits(hidden_states, inference_params, cfg_scale)

            self._cg_input_ids = input_ids.clone()
            self._cg_logits = torch.empty_like(logits)

            g = torch.cuda.CUDAGraph() # type: ignore[no-untyped-call]

            def capture_region() -> None:
                hidden_states_local = self.embed_codes(self._cg_input_ids) # type: ignore[arg-type]
                hidden_states_local = hidden_states_local.repeat(2, 1, 1)
                self._cg_logits = self.compute_logits(hidden_states_local, self._cg_inference_params, self._cg_scale) # type: ignore[arg-type]

            with torch.cuda.graph(g):
                capture_region()

            self._cg_graph = g

        else:
            self._cg_input_ids.copy_(input_ids) # type: ignore[union-attr]

        assert self._cg_graph is not None, "Could not capture CUDA graph"
        self._cg_graph.replay() # type: ignore[no-untyped-call]
        assert self._cg_logits is not None, "Could not capture logits in CUDA graph"
        return self._cg_logits

    def prefill(
        self,
        prefix_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        "Prefill" mode: we already have `prefix_hidden_states`, and we want
        to append new embeddings, then compute the logits.
        """
        # Replicate input_ids if CFG is enabled
        if cfg_scale != 1.0:
            input_ids = input_ids.expand(prefix_hidden_states.shape[0], -1, -1)
        hidden_states = torch.cat([prefix_hidden_states, self.embed_codes(input_ids)], dim=1)
        return self.compute_logits(hidden_states, inference_params, cfg_scale)

    def setup_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype = torch.bfloat16
    ) -> InferenceParams:
        key_value_memory_dict = {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
            for i, layer in enumerate(self.backbone.layers)
        }
        lengths_per_sample = torch.full((batch_size,), 0, dtype=torch.int32, device="cuda")
        return InferenceParams(max_seqlen, batch_size, 0, 0, key_value_memory_dict, lengths_per_sample)

    def prepare_conditioning(
        self,
        cond_dict: Dict[str, Any],
        uncond_dict: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        if uncond_dict is None:
            uncond_dict = {k: cond_dict[k] for k in self.prefix_conditioner.required_keys}

        c = self.prefix_conditioner(cond_dict)
        return torch.cat(
            [
                self.prefix_conditioner(cond_dict),
                self.prefix_conditioner(uncond_dict),
            ]
        )

    @torch.inference_mode()
    def generate(
        self,
        prefix_conditioning: torch.Tensor,  # [bsz, cond_seq_len, d_model]
        audio_prefix_codes: Optional[torch.Tensor] = None,  # [bsz, 9, prefix_audio_seq_len]
        max_new_tokens: int = 86 * 30,
        cfg_scale: float = 2.0,
        batch_size: int = 1,
        sampling_params: Dict[str, Any] = dict(min_p=0.1),
    ) -> torch.Tensor:
        assert cfg_scale != 1, "TODO: add support for cfg_scale=1"
        prefix_audio_len = 0 if audio_prefix_codes is None else audio_prefix_codes.shape[2]

        unknown_token = -1
        audio_seq_len = prefix_audio_len + max_new_tokens
        seq_len = prefix_conditioning.shape[1] + audio_seq_len

        inference_params = self.setup_cache(batch_size=batch_size * 2, max_seqlen=seq_len)

        codes = torch.full((batch_size, 9, audio_seq_len), unknown_token, device=prefix_conditioning.device)
        if audio_prefix_codes is not None:
            codes[..., :prefix_audio_len] = audio_prefix_codes

        delayed_codes = apply_delay_pattern(codes, self.masked_token_id)

        delayed_prefix_audio_codes = delayed_codes[..., : prefix_audio_len + 1]

        logits = self.prefill(prefix_conditioning, delayed_prefix_audio_codes, inference_params, cfg_scale)
        next_token = sample_from_logits(logits, **sampling_params)

        offset = delayed_prefix_audio_codes.shape[2]
        frame = delayed_codes[..., offset : offset + 1]
        frame.masked_scatter_(frame == unknown_token, next_token)

        prefix_length = prefix_conditioning.shape[1] + prefix_audio_len + 1
        inference_params.seqlen_offset += prefix_length
        inference_params.lengths_per_sample[:] += prefix_length

        logit_bias = torch.zeros_like(logits)
        logit_bias[:, 1:, self.eos_token_id] = -torch.inf  # only allow codebook 0 to predict EOS

        stopping = torch.zeros(batch_size, dtype=torch.bool, device=prefix_conditioning.device)
        max_steps = delayed_codes.shape[2] - offset
        remaining_steps = torch.full((batch_size,), max_steps, device=prefix_conditioning.device)

        step = 0
        while torch.max(remaining_steps) > 0:
            offset += 1
            input_ids = delayed_codes[..., offset - 1 : offset]
            logits = self.decode_one_token(input_ids, inference_params, cfg_scale)

            next_token = sample_from_logits(logits, generated_tokens=delayed_codes[..., :offset], **sampling_params)
            eos_in_cb0 = next_token[:, 0] == self.eos_token_id

            remaining_steps[eos_in_cb0[:, 0]] = torch.minimum(remaining_steps[eos_in_cb0[:, 0]], torch.tensor(9))
            stopping |= eos_in_cb0[:, 0]

            eos_codebook_idx = 9 - remaining_steps
            eos_codebook_idx = torch.clamp(eos_codebook_idx, max=9 - 1)
            for i in range(next_token.shape[0]):
                if stopping[i]:
                    idx = eos_codebook_idx[i].item()
                    next_token[i, :idx] = self.masked_token_id
                    next_token[i, idx] = self.eos_token_id

            frame = delayed_codes[..., offset : offset + 1]
            frame.masked_scatter_(frame == unknown_token, next_token)
            inference_params.seqlen_offset += 1
            inference_params.lengths_per_sample[:] += 1

            remaining_steps -= 1
            step += 1

        out_codes = revert_delay_pattern(delayed_codes)
        out_codes.masked_fill_(out_codes >= 1024, 0)
        out_codes = out_codes[..., : offset - 9]

        self._cg_graph = None  # reset cuda graph to avoid cache changes
        return out_codes
