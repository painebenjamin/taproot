# Adapted from https://raw.githubusercontent.com/WASasquatch/PowerNoiseSuite/main/modules/latent_noise.py
from __future__ import annotations

from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from ..normalization_util import normalize

class PerlinPowerFractal:
    """
    Generate a batch of images with a Perlin power fractal effect.
    """
    def __init__(self, width: int, height: int) -> None:
        """
        Initialize the PerlinPowerFractal.
        """
        self.width = width
        self.height = height

    def __call__(
        self,
        batch_size: int,
        X: int,
        Y: int,
        Z: int,
        frame: int,
        device: Union[str, torch.device]="cpu",
        generator: Optional[torch.Generator]=None,
        evolution_factor: float=0.1,
        octaves: int=4,
        persistence: float=0.5,
        lacunarity: float=2.0,
        exponent: float=4.0,
        scale: int=100,
        brightness: float=0.0,
        contrast: float=0.0,
        min_clamp: float=0.0,
        max_clamp: float=1.0
    ) -> torch.Tensor:
        """
        Generate a batch of images with Perlin power fractal effect.
        """
        import torch

        def fade(t: torch.Tensor) -> torch.Tensor:
            """
            Fade function.
            """
            return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3 # type: ignore[no-any-return]

        def lerp(t: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """
            Linear interpolation.
            """
            return a + t * (b - a)

        def grad(
            hsh: torch.Tensor,
            x: torch.Tensor,
            y: torch.Tensor,
            z: torch.Tensor
        ) -> torch.Tensor:
            """
            Gradient function.
            """
            h = hsh & 15
            u = torch.where(h < 8, x, y)
            v = torch.where(h < 4, y, torch.where((h == 12) | (h == 14), x, z))
            return torch.where(h & 1 == 0, u, -u) + torch.where(h & 2 == 0, v, -v)

        def noise(
            x: torch.Tensor,
            y: torch.Tensor,
            z: torch.Tensor,
            p: torch.Tensor
        ) -> torch.Tensor:
            """
            Noise function.
            """
            X = (x.floor() % 255).to(torch.int32)
            Y = (y.floor() % 255).to(torch.int32)
            Z = (z.floor() % 255).to(torch.int32)

            x -= x.floor()
            y -= y.floor()
            z -= z.floor()

            u = fade(x)
            v = fade(y)
            w = fade(z)

            A = p[X] + Y
            AA = p[A] + Z
            AB = p[A + 1] + Z
            B = p[X + 1] + Y
            BA = p[B] + Z
            BB = p[B + 1] + Z

            r = lerp(
                w,
                lerp(
                    v,
                    lerp(
                        u,
                        grad(p[AA], x, y, z),
                        grad(p[BA], x - 1, y, z)
                    ),
                    lerp(
                        u,
                        grad(p[AB], x, y - 1, z),
                        grad(p[BB], x - 1, y - 1, z)
                    )
                ),
                lerp(
                    v,
                    lerp(
                        u,
                        grad(p[AA + 1], x, y, z - 1),
                        grad(p[BA + 1], x - 1, y, z - 1)
                    ),
                    lerp(
                        u,
                        grad(p[AB + 1], x, y - 1, z - 1),
                        grad(p[BB + 1], x - 1, y - 1, z - 1)
                    )
                )
            )

            return r

        p = torch.randperm(max(self.width, self.height) ** 2, dtype=torch.int32, device=device, generator=generator)
        p = torch.cat((p, p))

        noise_map = torch.zeros(batch_size, self.height, self.width, dtype=torch.float32, device=device)

        x_t = torch.arange(self.width, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) + X
        y_t = torch.arange(self.height, dtype=torch.float32, device=device).unsqueeze(1).unsqueeze(0) + Y
        z_t = evolution_factor * torch.arange(batch_size, dtype=torch.float32, device=device).unsqueeze(1).unsqueeze(1) + Z + frame

        for octave in range(octaves):
            frequency = lacunarity ** octave
            amplitude = persistence ** octave

            nx = x_t / scale * frequency
            ny = y_t / scale * frequency
            nz = (z_t + frame * evolution_factor) / scale * frequency

            noise_values = noise(nx, ny, nz, p) * (amplitude ** exponent)

            noise_map += noise_values.squeeze(-1) * amplitude

        noise_map = normalize(noise_map, min_clamp, max_clamp)

        latent = (noise_map + brightness) * (1.0 + contrast)
        latent = normalize(latent)
        latent = latent.unsqueeze(-1)

        return latent
