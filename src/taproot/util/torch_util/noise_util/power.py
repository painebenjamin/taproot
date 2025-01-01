# Adapted from https://raw.githubusercontent.com/WASasquatch/PowerNoiseSuite/main/modules/latent_noise.py
from __future__ import annotations

from typing import Optional, Union, List, Any, Callable, Tuple, TYPE_CHECKING
from typing_extensions import Literal

if TYPE_CHECKING:
    import torch

from ..normalization_util import normalize

NOISE_TYPE_LITERAL = Literal["white", "grey", "pink", "green", "blue", "random_mix", "brownian_fractal", "velvet", "violet"]

class PowerLawNoise:
    """
    Generate various types of power-law noise.
    """
    def __init__(self, device: Union[str, torch.device] = "cpu") -> None:
        """
        Initialize the PowerLawNoise.

        Args:
            device (str, optional): The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
        """
        self.device = device
        
    @staticmethod
    def get_noise_types() -> List[NOISE_TYPE_LITERAL]:
        """
        Return the valid noise types

        Returns:
            (list): a list of noise types to use for noise_type parameter
        """
        return ["white", "grey", "pink", "green", "blue", "random_mix", "brownian_fractal", "velvet", "violet"]

    @classmethod
    def get_random_noise_type(cls, generator: Optional[torch.Generator] = None) -> NOISE_TYPE_LITERAL:
        """
        Returns a random noise type
        """
        noise_types = cls.get_noise_types()
        return noise_types[int(torch.randint(0, len(noise_types), (1,), generator=generator)[0])]

    def get_generator(self, noise_type: NOISE_TYPE_LITERAL) -> Callable[..., torch.Tensor]:
        """
        Return the noise generator function for the specified noise type.
        """
        if noise_type == "white":
            return self.white_noise
        elif noise_type == "grey":
            return self.grey_noise
        elif noise_type == "pink":
            return self.pink_noise
        elif noise_type == "green":
            return self.green_noise
        elif noise_type == "blue":
            return self.blue_noise
        elif noise_type == "velvet":
            return self.velvet_noise
        elif noise_type == "violet":
            return self.violet_noise
        elif noise_type == "random_mix":
            return self.mix_noise
        elif noise_type == "brownian_fractal":
            return self.brownian_fractal_noise
        raise ValueError(f"`noise_type` is invalid. Valid types are {', '.join(self.get_noise_types())}")

    def white_noise(
        self,
        batch_size: int,
        width: int,
        height: int,
        scale: float,
        alpha: float=0.0,
        generator: Optional[torch.Generator]=None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Generate white noise with a power-law distribution.
        """
        scale = scale
        noise_real = torch.randn((batch_size, 1, height, width), device=self.device, generator=generator)
        noise_power_law = torch.sign(noise_real) * torch.abs(noise_real) ** alpha
        noise_power_law *= scale
        return noise_power_law.to(self.device) # type: ignore[no-any-return]

    def grey_noise(
        self,
        batch_size: int,
        width: int,
        height: int,
        scale: float,
        alpha: float=1.0,
        generator: Optional[torch.Generator]=None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Generate grey noise with a flat power spectrum and modulation.
        """
        scale = scale
        noise_real = torch.randn((batch_size, 1, height, width), device=self.device, generator=generator)
        modulation = torch.abs(noise_real) ** (alpha - 1)
        noise_modulated = noise_real * modulation
        noise_modulated *= scale
        return noise_modulated.to(self.device) # type: ignore[no-any-return]

    def blue_noise(
        self,
        batch_size: int,
        width: int,
        height: int,
        scale: float,
        alpha: float=2.0,
        generator: Optional[torch.Generator]=None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Generate blue noise using the power spectrum method.
        """
        import torch
        import torch.fft as fft
        noise = torch.randn(batch_size, 1, height, width, device=self.device, generator=generator)
        
        freq_x = fft.fftfreq(width, 1.0)
        freq_y = fft.fftfreq(height, 1.0)
        Fx, Fy = torch.meshgrid(freq_x, freq_y, indexing="ij")
        
        power = (Fx**2 + Fy**2)**(alpha / 2.0)
        power[0, 0] = 1.0
        power = power.unsqueeze(0).expand(batch_size, 1, width, height).permute(0, 1, 3, 2).to(device=self.device)
        
        noise_fft = fft.fftn(noise)
        power = power.to(noise_fft)
        noise_fft = noise_fft / torch.sqrt(power)
        
        noise_real = fft.ifftn(noise_fft).real
        noise_real = noise_real - noise_real.min()
        noise_real = noise_real / noise_real.max()
        noise_real = noise_real * scale
        
        return noise_real.to(self.device) # type: ignore[no-any-return]

    def green_noise(
        self,
        batch_size: int,
        width: int,
        height: int,
        scale: float,
        alpha: float=1.5,
        generator: Optional[torch.Generator]=None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Generate green noise using the power spectrum method.
        """
        import torch
        import torch.fft as fft
        noise = torch.randn(batch_size, 1, height, width, device=self.device, generator=generator)
        
        freq_x = fft.fftfreq(width, 1.0)
        freq_y = fft.fftfreq(height, 1.0)
        Fx, Fy = torch.meshgrid(freq_x, freq_y, indexing="ij")
        
        power = (Fx**2 + Fy**2)**(alpha / 2.0)
        power[0, 0] = 1.0
        power = power.unsqueeze(0).expand(batch_size, 1, width, height).permute(0, 1, 3, 2).to(device=self.device)
        
        noise_fft = fft.fftn(noise)
        power = power.to(noise_fft)
        noise_fft = noise_fft / torch.sqrt(power)
        
        noise_real = fft.ifftn(noise_fft).real
        noise_real = noise_real - noise_real.min()
        noise_real = noise_real / noise_real.max()
        noise_real = noise_real * scale
        
        return noise_real.to(self.device) # type: ignore[no-any-return]
        
    def pink_noise(
        self,
        batch_size: int,
        width: int,
        height: int,
        scale: float,
        alpha: float=1.0,
        generator: Optional[torch.Generator]=None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Generate pink noise using the power spectrum method.
        """
        import torch
        import torch.fft as fft

        noise = torch.randn(batch_size, 1, height, width, device=self.device, generator=generator)
        
        freq_x = fft.fftfreq(width, 1.0)
        freq_y = fft.fftfreq(height, 1.0)
        Fx, Fy = torch.meshgrid(freq_x, freq_y, indexing="ij")
        
        power = (Fx**2 + Fy**2)**(alpha / 2.0)
        power[0, 0] = 1.0
        power = power.unsqueeze(0).expand(batch_size, 1, width, height).permute(0, 1, 3, 2).to(device=self.device)

        noise_fft = fft.fftn(noise)
        noise_fft = noise_fft / torch.sqrt(power.to(noise_fft.dtype))

        noise_real = fft.ifftn(noise_fft).real
        noise_real = noise_real - noise_real.min()
        noise_real = noise_real / noise_real.max()
        noise_real = noise_real * scale
        
        return noise_real.to(self.device) # type: ignore[no-any-return]

    def velvet_noise(
        self,
        batch_size: int,
        width: int,
        height: int,
        scale: float,
        alpha: float=1.0,
        generator: Optional[torch.Generator]=None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Generate true Velvet noise with specified width and height using PyTorch.
        """
        import torch

        white_noise = torch.randn((batch_size, 1, height, width), device=self.device, generator=generator)
        velvet_noise = torch.sign(white_noise) * torch.abs(white_noise) ** (1 / alpha)
        velvet_noise = velvet_noise / torch.max(torch.abs(velvet_noise))

        return velvet_noise # type: ignore[no-any-return]

    def violet_noise(
        self,
        batch_size: int,
        width: int,
        height: int,
        scale: float,
        alpha: float=1.0,
        generator: Optional[torch.Generator]=None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Generate true Violet noise with specified width and height using PyTorch.
        """
        import torch

        white_noise = torch.randn((batch_size, 1, height, width), device=self.device, generator=generator)
        violet_noise = torch.sign(white_noise) * torch.abs(white_noise) ** (alpha / 2.0)
        violet_noise = violet_noise / torch.max(torch.abs(violet_noise))

        return violet_noise # type: ignore[no-any-return]

    def brownian_fractal_noise(
        self,
        batch_size: int,
        width: int,
        height: int,
        scale: float,
        alpha: float=1.0,
        generator: Optional[torch.Generator]=None,
        modulator: float=1.0,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Generate Brownian fractal noise using the power spectrum method.
        """
        import torch
        import torch.nn.functional as F
        import torch.fft as fft

        def add_particles_to_grid(
            grid: torch.Tensor,
            particle_x: torch.Tensor,
            particle_y: torch.Tensor
        ) -> None:
            """
            Add particles to the grid
            """
            for x, y in zip(particle_x, particle_y):
                grid[y, x] = 1

        def move_particles(
            particle_x: torch.Tensor,
            particle_y: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Move particles
            """
            dx = torch.randint(-1, 2, (batch_size, n_particles), device=self.device, generator=generator)
            dy = torch.randint(-1, 2, (batch_size, n_particles), device=self.device, generator=generator)
            particle_x = torch.clamp(particle_x + dx, 0, width - 1)
            particle_y = torch.clamp(particle_y + dy, 0, height - 1)
            return particle_x, particle_y

        n_iterations = int(5000 * modulator)
        fy = fft.fftfreq(height).unsqueeze(1) ** 2
        fx = fft.fftfreq(width) ** 2
        f = fy + fx
        power = torch.sqrt(f) ** alpha
        power[0, 0] = 1.0

        grid = torch.zeros(height, width, dtype=torch.uint8, device=self.device)

        n_particles = n_iterations // 10 
        particle_x = torch.randint(0, int(width), (batch_size, n_particles), device=self.device, generator=generator)
        particle_y = torch.randint(0, int(height), (batch_size, n_particles), device=self.device, generator=generator)

        neighborhood = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.uint8, device=self.device)

        for _ in range(n_iterations):
            add_particles_to_grid(grid, particle_x, particle_y)
            particle_x, particle_y = move_particles(particle_x, particle_y)

        brownian_tree = grid.clone().detach().float().to(self.device)
        brownian_tree = brownian_tree / brownian_tree.max()
        brownian_tree = F.interpolate(brownian_tree.unsqueeze(0).unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False)
        brownian_tree = brownian_tree.squeeze(0).squeeze(0)

        fy = fft.fftfreq(height).unsqueeze(1) ** 2
        fx = fft.fftfreq(width) ** 2
        f = fy + fx
        power = torch.sqrt(f) ** alpha
        power[0, 0] = 1.0
        noise_real = brownian_tree * scale

        amplitude = 1.0 / (scale ** (alpha / 2.0))
        noise_real *= amplitude

        noise_fft = fft.fftn(noise_real.to(self.device))
        noise_fft = noise_fft / power.to(self.device)
        noise_real = fft.ifftn(noise_fft).real
        noise_real *= scale

        return noise_real.unsqueeze(0).unsqueeze(0)

    def mix_noise(
        self,
        batch_size: int,
        width: int,
        height: int,
        scale: float,
        alpha: float=2.0,
        generator: Optional[torch.Generator]=None,
        modulator: float=1.0,
        num_noises: int=3,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Mix noise randomly from a number of noise types.
        """
        import torch
        noise_types: List[NOISE_TYPE_LITERAL] = []

        while len(noise_types) < num_noises:
            noise_type = self.get_random_noise_type(generator=generator)
            if noise_type != "random_mix":
                noise_types.append(noise_type)

        scales = [scale] * num_noises
        noise_alpha = 0.5 + (float(torch.rand((1,), generator=generator)[0]) * 1.5)

        mixed_noise = torch.zeros(batch_size, 1, height, width, device=self.device)

        for noise_type in noise_types:
            noise = self.get_generator(noise_type)(
                batch_size=batch_size,
                width=width,
                height=height,
                scale=scale,
                alpha=noise_alpha,
                modulator=modulator,
                generator=generator
            ).to(self.device)
            mixed_noise += noise

        return mixed_noise

    def __call__(
        self,
        batch_size: int,
        width: int,
        height: int,
        scale: float,
        alpha: float=2.0,
        noise_type: NOISE_TYPE_LITERAL="white",
        generator: Optional[torch.Generator]=None,
        modulator: float=1.0,
        num_noises: int=3,
        num_channels: int=3,
    ) -> torch.Tensor:
        """
        Generate a noise image with options for type, frequency, and generator
        """
        if noise_type not in self.get_noise_types():
            raise ValueError(f"`noise_type` is invalid. Valid types are {', '.join(self.get_noise_types())}")

        import torch
        channels: List[torch.Tensor] = []
        for i in range(num_channels):
            noise = normalize(
                self.get_generator(noise_type)(
                    batch_size=batch_size,
                    width=width,
                    height=height,
                    scale=scale,
                    generator=generator,
                    alpha=alpha,
                    modulator=modulator,
                    num_noises=num_noises
                )
            )
            channels.append(noise)

        noise_image = torch.cat(channels, dim=1)
        noise_image = (noise_image - noise_image.min()) / (noise_image.max() - noise_image.min())
        noise_image = noise_image.permute(0, 2, 3, 1).float()

        return noise_image
