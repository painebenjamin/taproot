from __future__ import annotations

import random

from dataclasses import dataclass
from typing import Optional, Union, Tuple, Callable, Any, Literal, Dict, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import (
        Tensor,
        Generator,
        device as Device,
        dtype as DType
    )

NOISE_METHOD_LITERAL = Literal[
    "default", "crosshatch", "simplex",
    "perlin", "brownian_fractal", "white",
    "grey", "pink", "green", "blue",
    "violet", "velvet", "random_mix"
]
POWER_TYPE_LITERAL = Literal[
    "white", "grey", "pink",
    "green", "blue", "violet",
    "velvet", "random_mix", "brownian_fractal"
]

__all__ = [
    "NOISE_METHOD_LITERAL",
    "POWER_TYPE_LITERAL",
    "NoiseMaker",
    "make_noise",
    "reschedule_noise",
]

@dataclass
class NoiseMaker:
    """
    A class to make noise
    """
    batch_size: int
    channels: int
    height: int
    width: int
    frames: Optional[int] = None
    generator: Optional[Generator] = None
    device: Optional[Device] = None
    dtype: Optional[DType] = None

    @property
    def shape(self) -> Union[
        Tuple[int, int, int, int],
        Tuple[int, int, int, int, int],
    ]:
        """
        Gets the shape of the return tensor
        """
        if self.frames:
            return (
                self.batch_size,
                self.channels,
                self.frames,
                self.height,
                self.width,
             )
        return (
            self.batch_size,
            self.channels,
            self.height,
            self.width,
         )

    def default(self) -> Tensor:
        """
        Uses the default torch.rand
        """
        import torch
        return torch.randn(
            self.shape,
            generator=self.generator,
            dtype=self.dtype,
            layout=torch.strided
        ).to(self.device)

    def power(
        self,
        alpha: float = 1.0,
        scale: float = 1.0,
        modulator: float = 0.1,
        noise_type: POWER_TYPE_LITERAL = "brownian_fractal"
    ) -> Tensor:
        """
        Calculates power law noise
        """
        import torch
        from .power import PowerLawNoise
        frames = 1 if self.frames is None else self.frames
        shape = (
            frames,
            self.batch_size,
            self.height,
            self.width,
            self.channels,
        )
        noise = torch.ones(shape, dtype=torch.float32, device="cpu").cpu()
        power_generator = PowerLawNoise()
        for i in range(frames):
            noise[i, :, :, :, 0:min(self.channels, 3)] = power_generator(
                batch_size=self.batch_size,
                width=self.width,
                height=self.height,
                alpha=alpha,
                scale=scale,
                modulator=modulator,
                noise_type=noise_type,
                generator=self.generator,
            )[:, :, :, 0:min(self.channels, 3)]

        from einops import rearrange # type: ignore[import-not-found, unused-ignore]
        noise = rearrange(noise, "f b h w c -> b c f h w")
        if self.frames is None:
            noise = noise[:, :, 0, :, :]
        return noise.to(self.device, dtype=self.dtype)

    def simplex(self) -> Tensor:
        """
        Calculates simplex noise
        """
        import torch
        import numpy as np
        import opensimplex # type: ignore[import-not-found,unused-ignore]
        frames = 1 if self.frames is None else self.frames
        shape = (
            frames,
            self.batch_size,
            self.height,
            self.width,
            self.channels,
        )
        noise = torch.ones(shape, dtype=torch.float32, device="cpu").cpu()
        opensimplex.seed(int(torch.randint(2**32, (1,), generator=self.generator)[0]))
        for i in range(frames):
            noise[i, :, :, :, :] = torch.from_numpy(
                opensimplex.noise4array(
                    np.arange(self.channels),
                    np.arange(self.width),
                    np.arange(self.height),
                    np.arange(self.batch_size),
                )
            )
        from einops import rearrange
        noise = rearrange(noise, "f b h w c -> b c f h w")
        if self.frames is None:
            noise = noise[:, :, 0, :, :]
        return noise.to(self.device, dtype=self.dtype)

    def crosshatch(
        self,
        frequency: int=320,
        octaves: int=12,
        persistence: float=1.5,
        angle_degrees: int=45,
        brightness: float=0.0,
        contrast: float=0.0,
        blur: int=1,
        color_tolerance: float=0.01,
        num_colors: int=32,
        clamp_min: float=0.0,
        clamp_max: float=1.0,
    ) -> Tensor:
        """
        Calculates crosshatch noise
        """
        import torch
        from .crosshatch import CrossHatchPowerFractal
        frames = 1 if self.frames is None else self.frames
        shape = (
            frames,
            self.batch_size,
            self.height,
            self.width,
            self.channels,
        )
        noise = torch.ones(shape, dtype=torch.float32, device="cpu").cpu()
        crosshatch = CrossHatchPowerFractal(
            width=self.width,
            height=self.height,
            frequency=frequency,
            octaves=octaves,
            persistence=persistence,
            num_colors=num_colors,
            color_tolerance=color_tolerance,
            angle_degrees=angle_degrees,
            blur=blur,
            brightness=brightness,
            contrast=contrast,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
        for i in range(frames):
            noise[i, :, :, :, 0:min(self.channels, 3)] = crosshatch(
                batch_size=self.batch_size,
                generator=self.generator,
            )[:, :, :, 0:min(self.channels, 3)]

        from einops import rearrange # type: ignore[import-not-found, unused-ignore]
        noise = rearrange(noise, "f b h w c -> b c f h w")
        if self.frames is None:
            noise = noise[:, :, 0, :, :]
        return noise.to(self.device, dtype=self.dtype)

    def perlin(
        self,
        evolution_factor: float=0.1,
        octaves: int=4,
        persistence: float=0.5,
        lacunarity: float=2.0,
        exponent: float=4.0,
        scale: int=4,
        brightness: float=0.0,
        contrast: float=0.0,
        min_clamp: float=0.0,
        max_clamp: float=1.0,
    ) -> Tensor:
        """
        Calculates perlin noise
        """
        import torch
        from .perlin import PerlinPowerFractal
        frames = 1 if self.frames is None else self.frames
        shape = (
            frames,
            self.channels,
            self.batch_size,
            self.height,
            self.width,
        )
        noise = torch.ones(shape, dtype=torch.float32, device="cpu").cpu()
        perlin = PerlinPowerFractal(self.width, self.height)

        for i in range(frames):
            for j in range(self.channels):
                noise[i, j, :, :, :] = perlin(
                    batch_size=self.batch_size,
                    X=0,
                    Y=0,
                    Z=0,
                    frame=i,
                    evolution_factor=evolution_factor,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    exponent=exponent,
                    scale=scale,
                    brightness=brightness,
                    contrast=contrast,
                    min_clamp=min_clamp,
                    max_clamp=max_clamp
                )[:, :, :, 0]

        from einops import rearrange
        noise = rearrange(noise, "f c b h w -> b c f h w")
        if self.frames is None:
            noise = noise[:, :, 0, :, :]
        return noise.to(self.device, dtype=self.dtype)

    @classmethod
    def get_method_by_name(cls, method: NOISE_METHOD_LITERAL) -> Callable[[Any], Tensor]:
        """
        Gets the callable method by name
        """
        if method == "default":
            return cls.default
        elif method == "crosshatch":
            return cls.crosshatch
        elif method == "simplex":
            return cls.simplex
        elif method == "perlin":
            return cls.perlin
        return cls.power

    @classmethod
    def get_method_kwargs(
        cls,
        method: NOISE_METHOD_LITERAL,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Gets keyword arguments for the callable method
        """
        import inspect
        key_names = set(inspect.signature(cls.get_method_by_name(method)).parameters.keys())
        method_kwargs = dict([
            (key, value)
            for key, value in kwargs.items()
            if key in key_names
        ])
        if "noise_type" in key_names:
            method_kwargs["noise_type"] = method
        return method_kwargs

def reschedule_noise(
    noise: Tensor,
    window_size: int,
    window_stride: int,
    generator: Optional[Generator] = None
) -> Tensor:
    """
    Reschedules noise scross animation frames for more consistent diffusion-based animations
    See https://arxiv.org/abs/2310.15169
    """
    import torch
    _, _, frames, _, _ = noise.shape

    for frame_index in range(window_size, frames, window_stride):
        start_index = max(0, frame_index - window_size)
        end_index = min(frames, start_index + window_stride)
        window_length = end_index - start_index

        if window_length == 0:
            break

        list_indices = list(range(start_index, end_index))
        indices = torch.LongTensor(list_indices).to(noise.device)
        shuffled_indices = indices[torch.randperm(window_length, generator=generator)]

        current_start = frame_index
        current_end = min(frames, current_start + window_length)

        if current_end == current_start + window_length:
            # Fits perfectly in window
            noise[:, :, current_start:current_end] = noise[:, :, shuffled_indices]
        else:
            # Need to wrap around
            prefix_length = current_end - current_start
            shuffled_indices = shuffled_indices[:prefix_length]
            noise[:, :, current_start:current_end] = noise[:, :, shuffled_indices]

    return noise

def make_noise(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    frames: Optional[int] = None,
    reschedule_window_size: Optional[int] = None,
    reschedule_window_stride: Optional[int] = None,
    generator: Optional[Generator] = None,
    device: Optional[Device] = None,
    dtype: Optional[DType] = None,
    method: NOISE_METHOD_LITERAL = "default",
    **kwargs: Any
) -> Tensor:
    """
    Executes the passed method
    """
    noise_maker = NoiseMaker(
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        frames=frames,
        generator=generator,
        device=device,
        dtype=dtype
    )
    make_noise_method = noise_maker.get_method_by_name(method)
    noise = make_noise_method(
        noise_maker,
        **noise_maker.get_method_kwargs(method, **kwargs)
    )
    if frames and reschedule_window_size and reschedule_window_stride:
        noise = reschedule_noise(
            noise,
            reschedule_window_size,
            reschedule_window_stride,
            generator=generator
        )
    return noise
