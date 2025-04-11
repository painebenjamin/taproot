# First adapted from https://github.com/yl4579/StyleTTS2/blob/main/Modules/istftnet.py and https://github.com/yl4579/StyleTTS2/blob/main/Modules/utils.py
# Second adapted from https://huggingface.co/hexgrad/Kokoro-82M/raw/main/istftnet.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, cast

from scipy.signal import get_window # type: ignore[import-untyped]
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

from .util import init_weights
from .layers import (
    AdaINResBlock,
    AdaINResBlock1D
)

class TorchSTFT(torch.nn.Module):
    """
    PyTorch implementation of the short-time Fourier transform (STFT).
    """
    def __init__(
        self,
        filter_length: int=800,
        hop_length: int=200,
        win_length: int=800,
        window: str="hann"
    ) -> None:
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.from_numpy(
            get_window(window, win_length, fftbins=True).astype(np.float32)
        )

    def transform(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform the input tensor into the time-frequency domain.

        :param input_data: The input tensor.
        :return: The magnitude and phase of the STFT.
        """
        forward_transform = torch.stft(
            input_data,
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window.to(input_data.device),
            return_complex=True
        )

        return (
            torch.abs(forward_transform).to(input_data.dtype),
            torch.angle(forward_transform).to(input_data.dtype)
        )

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        Inverse the magnitude and phase back into the time domain.

        :param magnitude: The magnitude of the STFT.
        :param phase: The phase of the STFT.
        :return: The time-domain signal.
        """
        input_dtype = magnitude.dtype
        # Always use float32 for istft
        magnitude = magnitude.to(torch.float32)
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j), # convert polar to complex
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window.to(magnitude.device)
        )

        # unsqueeze to stay consistent with conv_transpose1d implementation
        return inverse_transform.unsqueeze(-2).to(input_dtype)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the STFT. 

        :param input_data: The input tensor.
        :return: The time-frequency reconstruction of the transformed input.
        """
        magnitude, phase = self.transform(input_data)
        reconstruction = self.inverse(magnitude, phase)
        return reconstruction
    
class SineGenerator(torch.nn.Module):
    """
    Sine-waveform generator.
    """
    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: float,
        harmonic_num: int=0,
        sine_amp: float=0.1,
        noise_std: float=0.003,
        voiced_threshold: float=0.0,
        flag_for_pulse: bool=False
    ) -> None:
        """
        :param sampling_rate: sampling rate in Hz
        :param harmonic_num: number of harmonic overtones
        :param sine_amp: amplitude of sine-wavefrom
        :param noise_std: std of Gaussian noise
        :param voiced_thoreshold: f0 threshold for U/V classification
        :param flag_for_pulse: this SinGen is used inside PulseGen
               when flag_for_pulse is True, the first time step of
               a voiced segment is always sin(np.pi) or cos(0)
        """
        super(SineGenerator, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = sampling_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale

    def sample_to_uv(self, f0: torch.Tensor) -> torch.Tensor:
        """
        Convert f0 tensor to voiced/unvoiced tensor.

        :param f0: The fundamental frequency tensor.
        :return: The voiced/unvoiced tensor.
        """
        uv = (f0 > self.voiced_threshold).to(f0.dtype)
        return uv

    def sample_to_sine(self, f0_values: torch.Tensor) -> torch.Tensor:
        """ 
        Convert f0 tensor to sine waveforms.

        :param f0_values: (batchsize, length, dim) where dim indicates
            fundamental tone and overtones
        :return: sine waveforms (batchsize, length, dim)
        """
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(
            f0_values.shape[0],
            f0_values.shape[2],
            device=f0_values.device,
            dtype=f0_values.dtype
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            rad_values = torch.nn.functional.interpolate(
                rad_values.transpose(1, 2), 
                scale_factor=1/self.upsample_scale, 
                mode="linear"
            ).transpose(1, 2)
            phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            phase = torch.nn.functional.interpolate(
                phase.transpose(1, 2) * self.upsample_scale, 
                scale_factor=self.upsample_scale,
                mode="linear"
            ).transpose(1, 2)
            sines = torch.sin(phase)
        else:
            # identify the last time step in unvoiced segments
            uv = self.sample_to_uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)

            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(
        self,
        f0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param f0: (batchsize=1, length, dim=1)
        :returns: (sine_tensor, uv, noise)
        """
        f0_buf = torch.zeros(
            f0.shape[0],
            f0.shape[1],
            self.dim,
            device=f0.device,
            dtype=f0.dtype
        )

        # fundamental component
        fn = torch.multiply(
            f0,
            torch.Tensor([
                [range(1, self.harmonic_num + 2)]
            ]).to(f0.device, f0.dtype)
        )

        # generate sine waveforms
        sine_waves = self.sample_to_sine(fn) * self.sine_amp

        # generate uv signal
        uv = self.sample_to_uv(f0)

        # noise: for unvoiced should be similar to sine_amp
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)

        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves = sine_waves * uv + noise

        return sine_waves, uv, noise

class SourceModuleHnNSF(torch.nn.Module):
    """
    SourceModule for hn-nsf
    """
    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_num: int=0,
        sine_amp: float=0.1,
        add_noise_std: float=0.003,
        voiced_threshold: float=0.0
    ) -> None:
        """
        :param sampling_rate: sampling_rate in Hz
        :param upsample_scale: upsample scale
        :param harmonic_num: number of harmonic above f0
        :param sine_amp: amplitude of sine source signal
        :param add_noise_std: std of additive Gaussian noise
        :param voiced_threshold: threhold to set U/V given f0 
        """
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGenerator(
            sampling_rate,
            upsample_scale,
            harmonic_num,
            sine_amp,
            add_noise_std,
            voiced_threshold
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: (batchsize, length, 1)
        :returns: (sine_merge, noise, uv)
        """
        # source for harmonic branch
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)

        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv

class Generator(torch.nn.Module):
    """
    Generator for hn-nsf
    """
    def __init__(
        self,
        style_dim: int,
        resblock_kernel_sizes: Tuple[int, ...],
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...],
        upsample_rates: Tuple[int, ...],
        upsample_kernel_sizes: Tuple[int, ...],
        upsample_initial_channel: int,
        gen_istft_n_fft: int,
        gen_istft_hop_size: int,
        lrelu_slope: float = 0.1
    ) -> None:
        """
        :param style_dim: dimension of style input
        :param resblock_kernel_sizes: kernel sizes for resblocks
        :param resblock_dilation_sizes: dilation sizes for resblocks
        :param upsample_rates: upsample rates for upsample layers
        :param upsample_kernel_sizes: kernel sizes for upsample layers
        :param upsample_initial_channel: initial channel for upsample layers
        :param gen_istft_n_fft: n_fft for istft
        :param gen_istft_hop_size: hop_size for istft
        :param lrelu_slope: slope for leaky relu
        """
        super(Generator, self).__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.post_n_fft = gen_istft_n_fft
        self.lrelu_slope = lrelu_slope

        # pre-processing layers
        self.stft = TorchSTFT(
            filter_length=gen_istft_n_fft,
            hop_length=gen_istft_hop_size,
            win_length=gen_istft_n_fft
        )
        self.m_source = SourceModuleHnNSF(
            sampling_rate=24000,
            upsample_scale=int(np.prod(upsample_rates) * gen_istft_hop_size),
            harmonic_num=8,
            voiced_threshold=10
        )

        # upsampling layers
        self.f0_upsamp = torch.nn.Upsample(
            scale_factor=int(np.prod(upsample_rates)) * gen_istft_hop_size
        )
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2
                    )
                )
            )

        # resblocks and noise layers
        self.resblocks = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(AdaINResBlock(ch, k, d, style_dim))

            c_cur = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                stride_f0 = int(np.prod(upsample_rates[i + 1:]))
                self.noise_convs.append(
                    Conv1d(
                        gen_istft_n_fft + 2,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=(stride_f0 + 1) // 2
                    )
                )
                self.noise_res.append(
                    AdaINResBlock(c_cur, 7, (1,3,5), style_dim)
                )
            else:
                self.noise_convs.append(
                    Conv1d(gen_istft_n_fft + 2, c_cur, kernel_size=1)
                )
                self.noise_res.append(
                    AdaINResBlock(c_cur, 11, (1,3,5), style_dim)
                )

        # post-processing layers
        self.conv_post = weight_norm(
            Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3)
        )
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))

        # initialize weights
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def remove_weight_norm(self) -> None:
        """
        Remove weight normalization from the convolutional layers.
        """
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()

        remove_weight_norm(self.conv_pre) # type: ignore[type-var]
        remove_weight_norm(self.conv_post)

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        f0: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x: (batchsize, length, dim)
        :param s: (batchsize, style_dim)
        :param f0: (batchsize, length)
        :returns: (batchsize, length)
        """
        with torch.no_grad():
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
            har_source, noi_source, uv = self.m_source(f0)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = self.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)

            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            x = x + x_source
            xs: Optional[torch.Tensor] = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x, s)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x, s)

            xs = cast(torch.Tensor, xs)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        return self.stft.inverse(spec, phase)

class Decoder(nn.Module):
    """
    Decoder for the hn-nsf model.
    """
    def __init__(
        self,
        in_dim: int=512,
        out_dim: int=80, 
        style_dim: int=64,
        f0_channels: int=512,
        resblock_kernel_sizes: Tuple[int, ...]=(3,7,11),
        upsample_rates: Tuple[int, ...]=(10, 6),
        upsample_initial_channel: int=512,
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...]=((1,3,5), (1,3,5), (1,3,5)),
        upsample_kernel_sizes: Tuple[int, ...]=(20, 12),
        gen_istft_n_fft: int=20,
        gen_istft_hop_size: int=5
    ) -> None:
        """
        :param in_dim: The number of input channels.
        :param f0_channels: The number of channels for the F0 curve.
        :param style_dim: The dimension of the style input.
        :param out_dim: The number of output channels.
        :param resblock_kernel_sizes: The kernel sizes for the resblocks.
        :param upsample_rates: The upsample rates for the upsample layers.
        :param upsample_initial_channel: The initial channel for the upsample layers.
        :param resblock_dilation_sizes: The dilation sizes for the resblocks.
        :param upsample_kernel_sizes: The kernel sizes for the upsample layers.
        :param gen_istft_n_fft: The n_fft for the ISTFT.
        :param gen_istft_hop_size: The hop_size for the ISTFT.
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.encode = AdaINResBlock1D(in_dim + 2, 1024, style_dim)
        self.decode = nn.ModuleList([
            AdaINResBlock1D(1024 + 2 + 64, 1024, style_dim),
            AdaINResBlock1D(1024 + 2 + 64, 1024, style_dim),
            AdaINResBlock1D(1024 + 2 + 64, 1024, style_dim),
            AdaINResBlock1D(1024 + 2 + 64, 512, style_dim, upsample="half")
        ])

        self.f0_conv = weight_norm(
            nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
        )

        self.n_conv = weight_norm(
            nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
        )

        self.asr_res = nn.Sequential(
            weight_norm(nn.Conv1d(512, 64, kernel_size=1)),
        )

        self.generator = Generator(
            style_dim=style_dim,
            resblock_kernel_sizes=resblock_kernel_sizes,
            upsample_rates=upsample_rates, 
            upsample_initial_channel=upsample_initial_channel,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_kernel_sizes=upsample_kernel_sizes,
            gen_istft_n_fft=gen_istft_n_fft,
            gen_istft_hop_size=gen_istft_hop_size
        )

    def forward(
        self,
        asr: torch.Tensor,
        f0_curve: torch.Tensor,
        n: torch.Tensor,
        s: torch.Tensor
    ) -> torch.Tensor:
        """
        :param asr: (batchsize, length, dim)
        :param f0_curve: (batchsize, length)
        :param n: (batchsize, length)
        :param s: (batchsize, style_dim)
        :returns: (batchsize, length)
        """
        f0 = self.f0_conv(f0_curve.unsqueeze(1))
        n = self.n_conv(n.unsqueeze(1))

        x = torch.cat([asr, f0, n], dim=1)
        x = self.encode(x, s)

        asr_res = self.asr_res(asr)
        res = True
        for block in self.decode:
            if res:
                x = torch.cat(
                    [x, asr_res, f0, n],
                    dim=1
                )
            x = block(x, s)
            if block.upsample_type is not None:
                res = False

        x = self.generator(x, s, f0_curve)
        return x
