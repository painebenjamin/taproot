import torch
import torch.amp as amp
import inspect
import numpy as np

from typing import Any, List, Optional, Sequence, Tuple

from .t5 import T5Encoder
from .wan import WanModel
from .vae import WanVideoVAE

from math import ceil
from diffusers import SchedulerMixin
from transformers import T5Tokenizer # type: ignore[import-untyped]
from contextlib import nullcontext

from taproot.util import maybe_use_tqdm, empty_cache

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
        dtype: torch.dtype=torch.bfloat16,
        device: Optional[torch.device]=None,
        default_negative_prompt: str="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
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

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str]=None,
        num_frames: int=81,
        width: int=832,
        height: int=480,
        shift: float=5.0,
        guidance_scale: float=5.0,
        num_inference_steps: int=50,
        generator: Optional[torch.Generator]=None,
        use_tqdm: bool=True,
        cpu_offload: bool=True,
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
        # Standardize args
        encoded_shape = self.vae.get_target_shape(
            num_frames=num_frames,
            height=height,
            width=width,
        )
        e_d, e_t, e_h, e_w = encoded_shape
        p_t, p_h, p_w = self.transformer.patch_size
        seq_len = ceil(
            (e_h * e_w) / (p_h * p_w) * e_t
        )

        use_classifier_free_guidance = guidance_scale > 1.0

        # Get timesteps
        timesteps, num_inference_steps = self.retrieve_timesteps(
            num_inference_steps=num_inference_steps,
            device=self.device,
            shift=shift,
        )

        # Encode prompts
        if cpu_offload:
            self.text_encoder.to(self.device)

        cond = [self.encode_prompt(prompt).to(self.dtype)]
        if use_classifier_free_guidance:
            uncond = [self.encode_prompt(negative_prompt or self.default_negative_prompt).to(self.dtype)]

        if cpu_offload:
            self.text_encoder.to("cpu")
            empty_cache()

        # Denoising loop
        noise = torch.randn(*encoded_shape, generator=generator, dtype=torch.float32)
        noise = noise.to(self.device, dtype=self.dtype)

        if hasattr(self.transformer, "no_sync"):
            sync_context = self.transformer.no_sync
        else:
            sync_context = nullcontext

        with amp.autocast("cuda", dtype=self.dtype), torch.no_grad(), sync_context(): # type: ignore[attr-defined]
            if cpu_offload:
                self.transformer.to(self.device)

            latents = [noise]

            for t in maybe_use_tqdm(
                timesteps,
                use_tqdm=use_tqdm,
                total=num_inference_steps
            ):
                timestep = torch.stack([t])
                latent_model_input = latents

                noise_pred_cond = self.transformer(
                    latent_model_input,
                    t=timestep,
                    context=cond,
                    seq_len=seq_len
                )[0]
                if use_classifier_free_guidance:
                    noise_pred_uncond = self.transformer(
                        latent_model_input,
                        t=timestep,
                        context=uncond,
                        seq_len=seq_len
                    )[0]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                temp_x0 = self.scheduler.step( # type: ignore[attr-defined]
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    generator=generator,
                    return_dict=False
                )[0]
                latents = [temp_x0.squeeze(0)]

            # Decode
            if cpu_offload:
                self.transformer.to("cpu")
                empty_cache()
                self.vae.to(self.device)

            videos = self.vae.decode(latents)

        if cpu_offload:
            self.vae.to("cpu")
            empty_cache()

        return videos[0]
