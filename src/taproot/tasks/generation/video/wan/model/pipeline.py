import torch
import torch.amp as amp
import inspect
import numpy as np

from typing import Any, List, Optional, Sequence, Tuple, Union

from .t5 import T5Encoder
from .wan import WanModel
from .vae import WanVideoVAE

from math import ceil
from diffusers import SchedulerMixin
from transformers import T5Tokenizer # type: ignore[import-untyped]
from contextlib import nullcontext

from taproot.constants import *
from taproot.util import (
    maybe_use_tqdm,
    empty_cache,
    logger,
    make_noise,
    log_duration,
    sliding_3d_windows,
    get_optimized_cfg_alpha,
)

class WanPipeline:
    """
    Video synthesis pipeline using Wan
    """
    def __init__(
        self,
        text_encoder: T5Encoder,
        tokenizer: T5Tokenizer,
        transformer: WanModel,
        vae: WanVideoVAE,
        scheduler: SchedulerMixin,
        dtype: torch.dtype=torch.float16,
        device: Optional[torch.device]=None,
        default_negative_prompt: str="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        stack_conditions: bool=False,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler
        self.dtype = dtype
        self.device = torch.device("cpu") if device is None else device
        self.default_negative_prompt = default_negative_prompt
        self.stack_conditions = stack_conditions

    def get_sampling_sigmas(
        self,
        sampling_steps: int,
        shift: float
    ) -> np.ndarray[Any, Any]:
        """
        Get the sampling sigmas for the inference steps
        :param sampling_steps: Number of sampling steps
        :param shift: Shift value
        :return: Sampling sigmas
        """
        sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
        sigma = (shift * sigma / (1 + (shift - 1) * sigma))
        return sigma

    def retrieve_timesteps(
        self,
        num_inference_steps: Optional[int]=None,
        device: Optional[torch.device]=None,
        timesteps: Optional[Sequence[int]]=None,
        sigmas: Optional[Sequence[float]]=None,
        **kwargs: Any,
    ) -> Tuple[List[int], int]:
        """
        Retrieve the timesteps for the inference steps

        :param num_inference_steps: Number of inference steps
        :param device: Device to use
        :param timesteps: Timesteps to use
        :param sigmas: Sigmas to use
        :param kwargs: Additional keyword arguments
        :return: Timesteps and number of inference steps
        """
        if not hasattr(self.scheduler, "set_timesteps") or not hasattr(self.scheduler, "timesteps"):
            raise ValueError("The current scheduler class does not support custom timesteps or sigmas schedules.")

        passed_args = [num_inference_steps, timesteps, sigmas]
        num_passed_args = sum([arg is not None for arg in passed_args])

        if num_passed_args != 1:
            raise ValueError(
                "Exactly one of `num_inference_steps`, `timesteps`, or `sigmas` must be passed."
            )

        accepts_shift = "shift" in set(
            inspect.signature(self.scheduler.set_timesteps)
                .parameters
                .keys()
        )

        if not accepts_shift:
            kwargs.pop("shift", None)

        if timesteps is not None:
            accepts_timesteps = (
                "timesteps" in set(
                    inspect.signature(self.scheduler.set_timesteps)
                        .parameters
                        .keys()
                )
            )
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {self.scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            self.scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            timesteps = list(self.scheduler.timesteps)
            num_inference_steps = len(timesteps)
        elif sigmas is not None:
            accept_sigmas = (
                "sigmas" in set(
                    inspect.signature(self.scheduler.set_timesteps)
                        .parameters
                        .keys()
                )
            )
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {self.scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )

            self.scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            timesteps = list(self.scheduler.timesteps)
            num_inference_steps = len(timesteps)
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = list(self.scheduler.timesteps)

        return timesteps, num_inference_steps # type: ignore [return-value]

    def get_strength_adjusted_timesteps(
        self,
        num_inference_steps: int,
        strength: float
    ) -> Tuple[List[int], int]:
        """
        Get the strength adjusted timesteps for the inference steps

        :param num_inference_steps: Number of inference steps
        :param strength: Strength value
        :return: Timesteps and number of inference steps
        """
        initial_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - initial_timestep, 0)
        i_start = t_start * self.scheduler.order # type: ignore[attr-defined]
        timesteps = self.scheduler.timesteps[i_start:] # type: ignore[attr-defined]

        if getattr(self.scheduler, "set_begin_index", None) is not None:
            self.scheduler.set_begin_index(i_start) # type: ignore[attr-defined]

        return timesteps, num_inference_steps - t_start

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: str,
    ) -> torch.Tensor:
        """
        Encode the prompt

        :param prompt: Prompt to encode
        :return: Encoded prompt
        """
        """
        self.text_encoder.model.to(self.device)
        return self.text_encoder([prompt], self.device)[0]
        """
        tokenizer_output = self.tokenizer(
            [prompt],
            padding="max_length",
            truncation=True,
            max_length=self.transformer.text_len,
            add_special_tokens=True,
            return_tensors="pt"
        )
        ids = tokenizer_output["input_ids"]
        mask = tokenizer_output["attention_mask"]
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_len = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask).detach()

        return [u[:v] for u, v in zip(context, seq_len)][0] # type: ignore [no-any-return]

    @torch.no_grad()
    def predict_noise_at_timestep(
        self,
        timestep: torch.Tensor,
        latents: List[torch.Tensor],
        cond: List[torch.Tensor],
        uncond: Optional[List[torch.Tensor]],
        window_size: Optional[int],
        window_stride: Optional[int],
        tile_size: Optional[Union[int, Tuple[int, int]]],
        tile_stride: Optional[Union[int, Tuple[int, int]]],
        guidance_scale: float,
        seq_len: int,
        do_classifier_free_guidance: bool,
        loop: bool,
        tile_horizontal: bool,
        tile_vertical: bool,
        use_cfg_alpha: bool,
    ) -> torch.Tensor:
        """
        Predict noise at a given timestep
        """
        use_multidiffusion = (window_size and window_stride) or (tile_size and tile_stride)
        batch_size = latents[0].shape[0]

        if use_multidiffusion:
            _, num_frames, height, width = latents[0].shape
            if not window_size:
                window_size = num_frames
            if not window_stride:
                window_stride = window_size // 2
            if not tile_size:
                tile_size = (width, height)
            elif not isinstance(tile_size, tuple):
                tile_size = (tile_size, tile_size)
            if not tile_stride:
                tile_stride = (tile_size[0] // 2, tile_size[1] // 2)
            elif not isinstance(tile_stride, tuple):
                tile_stride = (tile_stride, tile_stride)

            window_overlap = window_size - window_stride
            window_size = min(window_size, num_frames)
            tile_stride_width, tile_stride_height = tile_stride

            windows = sliding_3d_windows(
                frames=num_frames * (1 + loop),
                height=height * (1 + tile_vertical),
                width=width * (1 + tile_horizontal),
                frame_window_size=window_size,
                frame_window_stride=window_stride,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            windows = [
                (
                    start % num_frames,
                    end if end <= num_frames else end % num_frames,
                    top % height,
                    bottom if bottom <= height else bottom % height,
                    left % width,
                    right if right <= width else right % width,
                )
                for start, end, top, bottom, left, right in windows
                if start < num_frames and left < width and top < height
            ]

            num_windows = len(windows)
            noise_pred_count = torch.zeros_like(latents[0])
            noise_pred_total = torch.zeros_like(latents[0])

            for i, (start, end, top, bottom, left, right) in maybe_use_tqdm(
                enumerate(windows),
                desc="Diffusing windows",
                total=num_windows
            ):
                is_looped = start >= end
                is_wrap_horizontal = left >= right
                is_wrap_vertical = top >= bottom

                if is_looped:
                    if is_wrap_horizontal:
                        if is_wrap_vertical:
                            # Horizontally wrapped
                            # Vertically wrapped
                            # Temporally wrapped
                            latent_model_input = [
                                torch.cat([
                                    torch.cat([
                                        torch.cat([
                                            l[:, start:, top:, left:],
                                            l[:, :end, top:, left:]
                                        ], dim=1),
                                        torch.cat([
                                            l[:, start:, :bottom, left:],
                                            l[:, :end, :bottom, left:]
                                        ], dim=1)
                                    ], dim=2),
                                    torch.cat([
                                        torch.cat([
                                            l[:, start:, top:, :right],
                                            l[:, :end, top:, :right]
                                        ], dim=1),
                                        torch.cat([
                                            l[:, start:, :bottom, :right],
                                            l[:, :end, :bottom, :right]
                                        ], dim=1)
                                    ], dim=2)
                                ], dim=3)
                                for l in latents
                            ]
                        else:
                            # NOT vertically wrapped
                            # Horizontally wrapped
                            # Temporally wrapped
                            latent_model_input = [
                                torch.cat([
                                    torch.cat([
                                        l[:, start:, top:bottom, left:],
                                        l[:, :end, top:bottom, left:]
                                    ], dim=1),
                                    torch.cat([
                                        l[:, start:, top:bottom, :right],
                                        l[:, :end, top:bottom, :right]
                                    ], dim=1),
                                ], dim=3)
                                for l in latents
                            ]
                    elif is_wrap_vertical:
                        # NOT horizontally wrapped
                        # Vertically wrapped
                        # Temporally wrapped
                        latent_model_input = [
                            torch.cat([
                                torch.cat([
                                    l[:, start:, top:, left:right],
                                    l[:, :end, top:, left:right]
                                ], dim=1),
                                torch.cat([
                                    l[:, start:, :bottom, left:right],
                                    l[:, :end, :bottom, left:right]
                                ], dim=1),
                            ], dim=2)
                            for l in latents
                        ]
                    else:
                        # NOT horizontally wrapped
                        # NOT vertically wrapped
                        # Temporally wrapped
                        latent_model_input = [
                            torch.cat([
                                l[:, start:, top:bottom, left:right],
                                l[:, :end, top:bottom, left:right]
                            ], dim=1)
                            for l in latents
                        ]
                elif is_wrap_horizontal:
                    if is_wrap_vertical:
                        # Horizontally wrapped
                        # Vertically wrapped
                        # NOT temporally wrapped
                        latent_model_input = [
                            torch.cat([
                                torch.cat([
                                    l[:, start:end, top:, left:],
                                    l[:, start:end, :bottom, left:]
                                ], dim=2),
                                torch.cat([
                                    l[:, start:end, top:, :right],
                                    l[:, start:end, :bottom, :right]
                                ], dim=2)
                            ], dim=3)
                            for l in latents
                        ]
                    else:
                        # NOT vertically wrapped
                        # Horizontally wrapped
                        # NOT temporally wrapped
                        latent_model_input = [
                            torch.cat([
                                l[:, start:end, top:bottom, left:],
                                l[:, start:end, top:bottom, :right]
                            ], dim=3)
                            for l in latents
                        ]
                elif is_wrap_vertical:
                    # NOT horizontally wrapped
                    # Vertically wrapped
                    # NOT temporally wrapped
                    latent_model_input = [
                        torch.cat([
                            l[:, start:end, top:, left:right],
                            l[:, start:end, :bottom, left:right]
                        ], dim=2)
                        for l in latents
                    ]
                else:
                    # NOT horizontally wrapped
                    # NOT vertically wrapped
                    # NOT temporally wrapped
                    latent_model_input = [
                        l[:, start:end, top:bottom, left:right]
                        for l in latents
                    ]

                if do_classifier_free_guidance and uncond is not None:
                    if self.stack_conditions:
                        [noise_pred_cond, noise_pred_uncond] = self.transformer(
                            latent_model_input + latent_model_input,
                            t=timestep,
                            context=cond + uncond,
                            seq_len=seq_len
                        )
                    else:
                        noise_pred_cond = self.transformer(
                            latent_model_input,
                            t=timestep,
                            context=cond,
                            seq_len=seq_len
                        )[0]
                        noise_pred_uncond = self.transformer(
                            latent_model_input,
                            t=timestep,
                            context=uncond,
                            seq_len=seq_len
                        )[0]

                    if use_cfg_alpha:
                        alpha = get_optimized_cfg_alpha(
                            noise_pred_cond.view(batch_size, -1),
                            noise_pred_uncond.view(batch_size, -1)
                        )
                        alpha = alpha.view(batch_size, 1, 1, 1)
                        noise_pred = noise_pred_uncond * alpha + guidance_scale * (noise_pred_cond - noise_pred_uncond * alpha)
                    else:
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = self.transformer(
                        latent_model_input,
                        t=timestep,
                        context=cond,
                        seq_len=seq_len
                    )[0]

                window_mask = torch.ones_like(noise_pred)

                if loop or start > 0:
                    window_mask[:, :window_overlap] = torch.linspace(0, 1, window_overlap, device=noise_pred.device).view(1, -1, 1, 1)
                if loop or end < num_frames:
                    window_mask[:, -window_overlap:] = torch.linspace(1, 0, window_overlap, device=noise_pred.device).view(1, -1, 1, 1)
                if tile_vertical or top > 0:
                    window_mask[:, :, :tile_stride_height] = torch.min(
                        window_mask[:, :, :tile_stride_height],
                        torch.linspace(0, 1, tile_stride_height, device=noise_pred.device).view(1, 1, -1, 1)
                    )
                if tile_vertical or bottom < height:
                    window_mask[:, :, -tile_stride_height:] = torch.min(
                        window_mask[:, :, -tile_stride_height:],
                        torch.linspace(1, 0, tile_stride_height, device=noise_pred.device).view(1, 1, -1, 1)
                    )
                if tile_horizontal or left > 0:
                    window_mask[:, :, :, :tile_stride_width] = torch.min(
                        window_mask[:, :, :, :tile_stride_width],
                        torch.linspace(0, 1, tile_stride_width, device=noise_pred.device).view(1, 1, 1, -1)
                    )
                if tile_horizontal or right < width:
                    window_mask[:, :, :, -tile_stride_width:] = torch.min(
                        window_mask[:, :, :, -tile_stride_width:],
                        torch.linspace(1, 0, tile_stride_width, device=noise_pred.device).view(1, 1, 1, -1)
                    )

                noise_pred = noise_pred * window_mask

                if is_looped:
                    start_t = start
                    end_t = num_frames
                    initial_t = end_t - start_t
                    if is_wrap_horizontal:
                        start_x = left
                        end_x = width
                        initial_x = end_x - start_x
                        if is_wrap_vertical:
                            # Horizontally wrapped
                            # Vertically wrapped
                            # Temporally wrapped
                            start_y = top
                            end_y = height
                            initial_y = end_y - start_y

                            noise_pred_total[:, start_t:end_t, start_y:end_y, start_x:end_x] += noise_pred[:, :initial_t, :initial_y, :initial_x]
                            noise_pred_count[:, start_t:end_t, start_y:end_y, start_x:end_x] += window_mask[:, :initial_t, :initial_y, :initial_x]
                            noise_pred_total[:, :end, start_y:end_y, start_x:end_x] += noise_pred[:, initial_t:, :initial_y, :initial_x]
                            noise_pred_count[:, :end, start_y:end_y, start_x:end_x] += window_mask[:, initial_t:, :initial_y, :initial_x]

                            noise_pred_total[:, start_t:end_t, :bottom, start_x:end_x] += noise_pred[:, :initial_t, initial_y:, :initial_x]
                            noise_pred_count[:, start_t:end_t, :bottom, start_x:end_x] += window_mask[:, :initial_t, initial_y:, :initial_x]
                            noise_pred_total[:, :end, :bottom, start_x:end_x] += noise_pred[:, initial_t:, initial_y:, :initial_x]
                            noise_pred_count[:, :end, :bottom, start_x:end_x] += window_mask[:, initial_t:, initial_y:, :initial_x]

                            noise_pred_total[:, start_t:end_t, start_y:end_y, :right] += noise_pred[:, :initial_t, :initial_y, initial_x:]
                            noise_pred_count[:, start_t:end_t, start_y:end_y, :right] += window_mask[:, :initial_t, :initial_y, initial_x:]
                            noise_pred_total[:, :end, start_y:end_y, :right] += noise_pred[:, initial_t:, :initial_y, initial_x:]
                            noise_pred_count[:, :end, start_y:end_y, :right] += window_mask[:, initial_t:, :initial_y, initial_x:]

                            noise_pred_total[:, start_t:end_t, :bottom, :right] += noise_pred[:, :initial_t, initial_y:, initial_x:]
                            noise_pred_count[:, start_t:end_t, :bottom, :right] += window_mask[:, :initial_t, initial_y:, initial_x:]
                            noise_pred_total[:, :end, :bottom, :right] += noise_pred[:, initial_t:, initial_y:, initial_x:]
                            noise_pred_count[:, :end, :bottom, :right] += window_mask[:, initial_t:, initial_y:, initial_x:]
                        else:
                            # Horizontally wrapped
                            # NOT vertically wrapped
                            # Temporally wrapped
                            noise_pred_total[:, start_t:end_t, top:bottom, start_x:end_x] += noise_pred[:, :initial_t, :, :initial_x]
                            noise_pred_count[:, start_t:end_t, top:bottom, start_x:end_x] += window_mask[:, :initial_t, :, :initial_x]
                            noise_pred_total[:, :end, top:bottom, start_x:end_x] += noise_pred[:, initial_t:, :, :initial_x]
                            noise_pred_count[:, :end, top:bottom, start_x:end_x] += window_mask[:, initial_t:, :, :initial_x]

                            noise_pred_total[:, start_t:end_t, top:bottom, :right] += noise_pred[:, :initial_t, :, initial_x:]
                            noise_pred_count[:, start_t:end_t, top:bottom, :right] += window_mask[:, :initial_t, :, initial_x:]
                            noise_pred_total[:, :end, top:bottom, :right] += noise_pred[:, initial_t:, :, initial_x:]
                            noise_pred_count[:, :end, top:bottom, :right] += window_mask[:, initial_t:, :, initial_x:]
                    elif is_wrap_vertical:
                        # NOT horizontally wrapped
                        # Vertically wrapped
                        # Temporally wrapped
                        start_y = top
                        end_y = height
                        initial_y = end_y - start_y

                        noise_pred_total[:, start_t:end_t, start_y:end_y, left:right] += noise_pred[:, :initial_t, :initial_y]
                        noise_pred_count[:, start_t:end_t, start_y:end_y, left:right] += window_mask[:, :initial_t, :initial_y]
                        noise_pred_total[:, :end, start_y:end_y, left:right] += noise_pred[:, initial_t:, :initial_y]
                        noise_pred_count[:, :end, start_y:end_y, left:right] += window_mask[:, initial_t:, :initial_y]

                        noise_pred_total[:, start_t:end_t, :bottom, left:right] += noise_pred[:, :initial_t, initial_y:]
                        noise_pred_count[:, start_t:end_t, :bottom, left:right] += window_mask[:, :initial_t, initial_y:]
                        noise_pred_total[:, :end, :bottom, left:right] += noise_pred[:, initial_t:, initial_y:]
                        noise_pred_count[:, :end, :bottom, left:right] += window_mask[:, initial_t:, initial_y:]
                    else:
                        # NOT horizontally wrapped
                        # NOT vertically wrapped
                        # Temporally wrapped
                        noise_pred_total[:, start_t:end_t, top:bottom, left:right] += noise_pred[:, :initial_t]
                        noise_pred_count[:, start_t:end_t, top:bottom, left:right] += window_mask[:, :initial_t]
                        noise_pred_total[:, :end, top:bottom, left:right] += noise_pred[:, initial_t:]
                        noise_pred_count[:, :end, top:bottom, left:right] += window_mask[:, initial_t:]
                elif is_wrap_horizontal:
                    start_x = left
                    end_x = width
                    initial_x = end_x - start_x
                    if is_wrap_vertical:
                        # Horizontally wrapped
                        # Vertically wrapped
                        # NOT temporally wrapped
                        start_y = top
                        end_y = height
                        initial_y = end_y - start_y

                        noise_pred_total[:, start:end, start_y:end_y, start_x:end_x] += noise_pred[:, :, :initial_y, :initial_x]
                        noise_pred_count[:, start:end, start_y:end_y, start_x:end_x] += window_mask[:, :, :initial_y, :initial_x]
                        noise_pred_total[:, start:end, :bottom, start_x:end_x] += noise_pred[:, :, initial_y:, :initial_x]
                        noise_pred_count[:, start:end, :bottom, start_x:end_x] += window_mask[:, :, initial_y:, :initial_x]

                        noise_pred_total[:, start:end, start_y:end_y, :right] += noise_pred[:, :, :initial_y, initial_x:]
                        noise_pred_count[:, start:end, start_y:end_y, :right] += window_mask[:, :, :initial_y, initial_x:]
                        noise_pred_total[:, start:end, :bottom, :right] += noise_pred[:, :, initial_y:, initial_x:]
                        noise_pred_count[:, start:end, :bottom, :right] += window_mask[:, :, initial_y:, initial_x:]
                    else:
                        # NOT vertically wrapped
                        # Horizontally wrapped
                        # NOT temporally wrapped
                        noise_pred_total[:, start:end, top:bottom, start_x:end_x] += noise_pred[:, :, :, :initial_x]
                        noise_pred_count[:, start:end, top:bottom, start_x:end_x] += window_mask[:, :, :, :initial_x]

                        noise_pred_total[:, start:end, top:bottom, :right] += noise_pred[:, :, :, initial_x:]
                        noise_pred_count[:, start:end, top:bottom, :right] += window_mask[:, :, :, initial_x:]
                elif is_wrap_vertical:
                    # NOT horizontally wrapped
                    # Vertically wrapped
                    # NOT temporally wrapped
                    start_y = top
                    end_y = height
                    initial_y = end_y - start_y

                    noise_pred_total[:, start:end, start_y:end_y, left:right] += noise_pred[:, :, :initial_y]
                    noise_pred_count[:, start:end, start_y:end_y, left:right] += window_mask[:, :, :initial_y]

                    noise_pred_total[:, start:end, :bottom, left:right] += noise_pred[:, :, initial_y:]
                    noise_pred_count[:, start:end, :bottom, left:right] += window_mask[:, :, initial_y:]
                else:
                    # NOT horizontally wrapped
                    # NOT vertically wrapped
                    # NOT temporally wrapped
                    noise_pred_total[:, start:end, top:bottom, left:right] += noise_pred
                    noise_pred_count[:, start:end, top:bottom, left:right] += window_mask

            noise_pred = torch.where(
                noise_pred_count > 0,
                noise_pred_total / noise_pred_count,
                noise_pred_total
            )
        else:
            latent_model_input = latents

            if do_classifier_free_guidance and uncond is not None:
                if self.stack_conditions:
                    [noise_pred_cond, noise_pred_uncond] = self.transformer(
                        latent_model_input + latent_model_input,
                        t=timestep,
                        context=cond + uncond,
                        seq_len=seq_len
                    )
                else:
                    noise_pred_cond = self.transformer(
                        latent_model_input,
                        t=timestep,
                        context=cond,
                        seq_len=seq_len
                    )[0]
                    noise_pred_uncond = self.transformer(
                        latent_model_input,
                        t=timestep,
                        context=uncond,
                        seq_len=seq_len
                    )[0]

                if use_cfg_alpha:
                    alpha = get_optimized_cfg_alpha(
                        noise_pred_cond.view(batch_size, -1),
                        noise_pred_uncond.view(batch_size, -1)
                    )
                    alpha = alpha.view(batch_size, 1, 1, 1)
                    noise_pred = noise_pred_uncond * alpha + guidance_scale * (noise_pred_cond - noise_pred_uncond * alpha)
                else:
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.transformer(
                    latent_model_input,
                    t=timestep,
                    context=cond,
                    seq_len=seq_len
                )[0]

        return noise_pred # type: ignore[no-any-return]

    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str]=None,
        num_frames: int=81,
        width: int=832,
        height: int=480,
        shift: float=5.0,
        video: Optional[torch.Tensor]=None,
        strength: float=0.6,
        use_cfg_alpha: bool=False,
        num_zero_steps: int=0,
        guidance_scale: float=5.0,
        guidance_end: Optional[float]=None,
        num_inference_steps: int=50,
        window_size: Optional[int]=None,
        window_stride: Optional[int]=None,
        tile_size: Optional[Union[str, int, Tuple[int, int]]]=None,
        tile_stride: Optional[Union[str, int, Tuple[int, int]]]=None,
        generator: Optional[torch.Generator]=None,
        loop: bool=False,
        tile_horizontal: bool=False,
        tile_vertical: bool=False,
        tile_vae: bool=False,
        use_tqdm: bool=True,
        cpu_offload: bool=True,
        flash_fix: bool=True,
    ) -> torch.Tensor:
        """
        Generate video frames from the prompt.

        :param prompt: Prompt to generate video frames from
        :param negative_prompt: Negative prompt to generate video frames from
        :param num_frames: Number of frames to generate
        :param width: Width of the video in pixels
        :param height: Height of the video in pixels
        :param shift: Shift value
        :param num_inference_steps: Number of inference steps
        :param guidance_scale: Guidance scale
        :param generator: Generator to use for reproducibility
        :return: Video frames [3, T, H, W]
        """
        if video is not None:
            assert strength > 0.0, "A positive strength value must be provided when passing a video."
            if strength == 1.0:
                # The input video will not be used
                video = None
            else:
                if video.ndim == 5:
                    video = video[0]

                num_frames, _, height, width = video.shape

        encoded_video: Optional[torch.Tensor] = None

        if video is not None:
            if cpu_offload:
                with log_duration("onloading vae"):
                    self.vae.to(self.device)

            if loop and flash_fix:
                # Prepend first 15 frames in reverse to avoid VAE warmup flash
                video = torch.cat([
                    torch.flip(video[:15], [0]),
                    video
                ], dim=0)

            with log_duration("encoding video"):
                encoded_video = self.vae.encode(
                    [video.permute(1, 0, 2, 3).to(self.device, dtype=self.dtype)],
                    tiled=tile_vae,
                    device=self.device,
                )[0]

            if loop and flash_fix:
                encoded_video = encoded_video[:, 4:]

            if cpu_offload:
                with log_duration("offloading vae"):
                    self.vae.to("cpu")
                    empty_cache()

            encoded_shape = encoded_video.shape
        else:
            encoded_shape = self.vae.get_target_shape( # type: ignore[assignment]
                num_frames=num_frames,
                height=height,
                width=width,
            )

        e_d, e_t, e_h, e_w = encoded_shape
        p_t, p_h, p_w = self.transformer.patch_size
        seq_len = ceil(
            (e_h * e_w) / (p_h * p_w) * e_t
        )

        if guidance_end is None:
            guidance_end = 1.0

        # Adjust windows based on latent spatiotemporal compression
        temporal_comp, spatial_comp, _ = self.vae.stride
        if window_size is not None:
            window_size = int(window_size) // temporal_comp
        if window_stride is not None:
            window_stride = int(window_stride) // temporal_comp
        if tile_size is not None:
            if isinstance(tile_size, tuple):
                tile_size = (tile_size[0] // spatial_comp, tile_size[1] // spatial_comp)
            elif isinstance(tile_size, str):
                if ":" in tile_size:
                    tile_size = tuple(map(int, tile_size.split(":"))) # type: ignore[assignment]
                    tile_size = (tile_size[0] // spatial_comp, tile_size[1] // spatial_comp) # type: ignore[operator]
                else:
                    tile_size = int(tile_size) // spatial_comp
            else:
                tile_size = tile_size // spatial_comp
        if tile_stride is not None:
            if isinstance(tile_stride, tuple):
                tile_stride = (tile_stride[0] // spatial_comp, tile_stride[1] // spatial_comp)
            elif isinstance(tile_stride, str):
                if ":" in tile_stride:
                    tile_stride = tuple(map(int, tile_stride.split(":"))) # type: ignore[assignment]
                    tile_stride = (tile_stride[0] // spatial_comp, tile_stride[1] // spatial_comp) # type: ignore[operator]
                else:
                    tile_stride = int(tile_stride) // spatial_comp
            else:
                tile_stride = tile_stride // spatial_comp

        do_classifier_free_guidance = guidance_scale > 1.0

        # Get timesteps
        timesteps, num_inference_steps = self.retrieve_timesteps(
            num_inference_steps=num_inference_steps,
            device=self.device,
            shift=shift,
        )

        if encoded_video is not None and strength is not None:
            timesteps, num_inference_steps = self.get_strength_adjusted_timesteps(
                num_inference_steps=num_inference_steps,
                strength=strength,
            )

        guidance_end_step = int(guidance_end * num_inference_steps) - 1

        # Encode prompts
        if cpu_offload:
            with log_duration("onloading text encoder"):
                self.text_encoder.to(self.device)

        with log_duration("encoding prompt"):
            cond = [self.encode_prompt(prompt).to(self.dtype)]

        if do_classifier_free_guidance:
            with log_duration("encoding negative prompt"):
                uncond = [self.encode_prompt(negative_prompt or self.default_negative_prompt).to(self.dtype)]
        else:
            uncond = None

        if cpu_offload:
            with log_duration("offloading text encoder"):
                self.text_encoder.to("cpu")
                empty_cache()

        noise = make_noise(
            batch_size=1,
            channels=e_d,
            frames=e_t,
            height=e_h,
            width=e_w,
            reschedule_window_size=window_size,
            reschedule_window_stride=window_stride,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )[0]

        if hasattr(self.transformer, "no_sync"):
            sync_context = self.transformer.no_sync
        else:
            sync_context = nullcontext # type: ignore[assignment]

        # Denoising loop
        with amp.autocast("cuda", dtype=self.dtype), torch.no_grad(), sync_context(): # type: ignore[attr-defined,operator]
            if cpu_offload:
                with log_duration("onloading transformer"):
                    self.transformer.to(self.device)

            if encoded_video is not None:
                latents = [
                    self.scheduler.add_noise( # type: ignore[attr-defined]
                        encoded_video,
                        noise,
                        timesteps[:1]
                    )
                ]
            else:
                latents = [noise]

            for i, t in maybe_use_tqdm(
                enumerate(timesteps),
                desc="Diffusing",
                use_tqdm=use_tqdm,
                total=num_inference_steps
            ):
                timestep = torch.stack([t])

                if i < num_zero_steps:
                    noise_pred = torch.zeros_like(latents[0])
                else:
                    noise_pred = self.predict_noise_at_timestep(
                        timestep=timestep,
                        latents=latents,
                        cond=cond,
                        uncond=uncond,
                        window_size=window_size,
                        window_stride=window_stride,
                        tile_size=tile_size,
                        tile_stride=tile_stride,
                        guidance_scale=guidance_scale,
                        seq_len=seq_len,
                        loop=loop,
                        tile_horizontal=tile_horizontal,
                        tile_vertical=tile_vertical,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        use_cfg_alpha=use_cfg_alpha,
                    )

                temp_x0 = self.scheduler.step( # type: ignore[attr-defined]
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    generator=generator,
                    return_dict=False
                )[0]
                latents = [temp_x0.squeeze(0)]

                if i >= guidance_end_step and do_classifier_free_guidance:
                    logger.debug(f"Disabling guidance at step {i}")
                    do_classifier_free_guidance = False

            # Decode
            if cpu_offload:
                with log_duration("offloading transformer"):
                    self.transformer.to("cpu")
                    empty_cache()

                with log_duration("onloading vae"):
                    self.vae.to(self.device)

            if loop and flash_fix:
                # The beginning ~13 frames will always have a noticeable jump as the VAE warms up
                # To make perfect loops, we re-add the beginning to the end of the video, then blend
                # in the repeated frames with the original frames.
                latents = [
                    torch.cat(
                        [l, l[:, :4]],
                        dim=1
                    )
                    for l in latents
                ]

            if tile_vertical:
                # Duplication vertically x3
                latents = [
                    torch.cat(
                        [l, l, l],
                        dim=2
                    )
                    for l in latents
                ]

            if tile_horizontal:
                # Duplication horizontally x3
                latents = [
                    torch.cat(
                        [l, l, l],
                        dim=3
                    )
                    for l in latents
                ]

            videos = self.vae.decode(
                latents,
                device=self.device,
                tiled=tile_vae,
            )

            if loop and flash_fix:
                for i in range(len(videos)):
                    mask = torch.ones((13,), device=videos[i].device, dtype=videos[i].dtype)
                    mask[-4:] = torch.linspace(1, 0, 4, device=videos[i].device, dtype=videos[i].dtype)
                    mask = mask.view(1, -1, 1, 1)
                    repeated = videos[i][:, -13:]
                    original = videos[i][:, :13]
                    videos[i][:, :13] = repeated * mask + original * (1 - mask)
                    videos[i] = videos[i][:, :-13]

        if cpu_offload:
            with log_duration("offloading vae"):
                self.vae.to("cpu")
                empty_cache()

        return videos[0]
