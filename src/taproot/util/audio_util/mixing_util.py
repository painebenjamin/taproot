from __future__ import annotations

from typing import List, Optional, Tuple, Union, Any, TYPE_CHECKING

from ..introspection_util import is_torch_tensor, is_numpy_array

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

__all__ = [
    "sine_wave",
    "concatenate_audio",
    "trim_silence"
]

def sine_wave(
    frequency: float,
    duration: float,
    sample_rate: int,
) -> Tensor:
    """
    Generate a sine wave.

    >>> sine_wave(440, 1, 4)
    tensor([ 0.0000e+00, -8.6604e-01,  8.6606e-01,  2.7341e-05])

    :param frequency: Frequency of the sine wave in Hz.
    :param duration: Duration of the sine wave in seconds.
    :param sample_rate: Sample rate of the sine wave.
    :return: A tensor of shape (1, num_samples) containing the sine wave.
    """
    import torch
    num_samples = int(duration * sample_rate)
    t = torch.linspace(0, duration, num_samples, dtype=torch.float32)
    return torch.sin(2 * torch.pi * frequency * t)

def concatenate_audio(
    audio: List[Tensor],
    cross_fade_samples: Optional[int]=None,
    cross_fade_duration: Optional[float]=None,
    sample_rate: Optional[int]=None,
) -> Tensor:
    """
    Concatenate audios with cross fade.

    >>> import torch
    >>> a = torch.tensor([1., 2., 3.])
    >>> b = torch.tensor([4., 5., 6.])
    >>> concatenate_audio([a, b], cross_fade_samples=3)
    tensor([1.7500, 3.5000, 5.2500])
    >>> concatenate_audio([a, b], cross_fade_duration=2, sample_rate=1)
    tensor([1.0000, 2.6667, 4.3333, 6.0000])
    >>> concatenate_audio([a, b])
    tensor([1., 2., 3., 4., 5., 6.])
    >>> concatenate_audio([a.unsqueeze(0), b.unsqueeze(0)], cross_fade_samples=1)
    tensor([[1.0000, 2.0000, 3.5000, 5.0000, 6.0000]])

    :param audio: A list of tensors of shape (num_samples) or (c, num_samples) containing the audio to concatenate.
    :param cross_fade_samples: Number of samples to cross fade. Takes precedence over cross_fade_duration.
    :param cross_fade_duration: Duration of the cross fade in seconds.
    :param sample_rate: Sample rate of the audio. Required if cross_fade_duration is provided.
    :return: A tensor of shape (num_samples) or (c, num_samples) containing the concatenated audio.
    """
    import torch
    num_audio = len(audio)
    if num_audio == 0:
        raise ValueError("At least one audio must be provided.")
    elif num_audio == 1:
        return audio[0]
    if cross_fade_samples is None and cross_fade_duration is not None:
        assert sample_rate is not None, "sample_rate must be provided if cross_fade_duration is provided."
        cross_fade_samples = int(cross_fade_duration * sample_rate)
    if cross_fade_samples is None:
        cross_fade_samples = 0

    if cross_fade_samples <= 0:
        # No fading necessary, just concatenate
        return torch.cat(audio, dim=-1)

    # Calculate total number of samples after applying cross fade
    total_samples = 0
    for i, wav in enumerate(audio):
        wav_samples = wav.size(-1)
        if i == 0:
            total_samples += wav_samples
        else:
            cross_fade_samples = min(cross_fade_samples, wav_samples)
            total_samples += wav_samples - cross_fade_samples

    audio_shape: Tuple[int, ...] = (total_samples,)
    if audio[0].ndim == 2:
        audio_shape = (audio[0].size(0), total_samples)

    # Allocate memory for the concatenated audio
    concatenated_audio = torch.zeros(
        audio_shape,
        dtype=torch.float32,
        device=audio[0].device
    )

    # Concatenate audio with cross fade
    start = 0
    for i, wav in enumerate(audio):
        wav_samples = wav.size(-1)
        wav_mask = torch.ones_like(wav).to(torch.float32)
        fade_samples_start = 0
        fade_samples_end = 0

        if i > 0:
            fade_samples_start = min(cross_fade_samples, wav_samples)
            wav_mask[..., :fade_samples_start] = torch.linspace(0, 1, fade_samples_start + 2, device=wav.device)[1:-1]

        if i < num_audio - 1:
            fade_samples_end = min(audio[i + 1].size(-1), cross_fade_samples)
            wav_mask[..., -fade_samples_end:] = torch.linspace(1, 0, fade_samples_end + 2, device=wav.device)[1:-1]

        end = start + wav_samples
        concatenated_audio[..., start:end] = concatenated_audio[..., start:end] + (wav * wav_mask)
        start += wav_samples - fade_samples_end

    return concatenated_audio

def trim_silence(
    audio: Union[Tensor, np.ndarray[Any, Any]],
    leading: bool=True,
    trailing: bool=True,
    threshold: float=1e-3,
    raise_when_all_silent: bool=True,
) -> Union[Tensor, np.ndarray[Any, Any]]:
    """
    Trim leading and trailing silence from audio.

    >>> import torch
    >>> trim_silence(torch.tensor([0., 0., 0., 1., 0., 0., 0.]))
    tensor([1.])
    >>> trim_silence(torch.tensor([0., 0., 0., 1., 0., 0., 0.]), leading=False)
    tensor([0., 0., 0., 1.])
    >>> trim_silence(torch.tensor([0., 0., 0., 1., 0., 0., 0.]), trailing=False)
    tensor([1., 0., 0., 0.])
    >>> trim_silence(torch.tensor([0., 0., 0.4, 1., 0.4, 0., 0.]), threshold=0.5)
    tensor([1.])
    >>> trim_silence(torch.tensor([[0., 0., 1., 0., 0.], [0., 0., 0., 0., 0.]]))
    tensor([[1.],
            [0.]])
    >>> import numpy as np
    >>> trim_silence(np.array([0., 0., 0., 1., 0., 0., 0.]))
    array([1.])
    >>> trim_silence(np.array([[0., 0., 1., 0., 0.], [0., 0., 0., 0., 0.]]))
    array([[1.],
           [0.]])

    :param audio: A tensor of shape (num_samples) or (c, num_samples) or a numpy array of shape (num_samples) or (c, num_samples) containing the audio to trim.
    :param leading: Whether to trim leading silence.
    :param trailing: Whether to trim trailing silence.
    :param threshold: Threshold below which audio is considered silent.
    :param raise_when_all_silent: Whether to raise an exception when all audio is silent.
    :return: A tensor of shape (num_samples) or (c, num_samples) or a numpy array of shape (num_samples) or (c, num_samples) containing the trimmed audio.
    """
    start_sample_index = 0
    end_sample_index = len(audio)
    if is_torch_tensor(audio):
        import torch

        abs_audio = torch.abs(audio)
        non_silent = (abs_audio > threshold).nonzero(as_tuple=True)

        if non_silent[-1].numel() == 0:  # All values are below the threshold
            if raise_when_all_silent:
                raise ValueError("All audio is silent.")
            return audio[..., :0]  # Return empty tensor

        if leading:
            start_sample_index = non_silent[-1].min().item() # type: ignore[assignment]

        if trailing:
            end_sample_index = non_silent[-1].max().item() + 1 # type: ignore[assignment]
        else:
            end_sample_index = len(audio)

        return audio[..., start_sample_index:end_sample_index]

    elif is_numpy_array(audio):
        import numpy as np

        abs_audio_np = np.abs(audio)
        non_silent_np = np.nonzero(abs_audio_np > threshold)

        if len(non_silent_np[-1]) == 0:  # All values are below the threshold
            if raise_when_all_silent:
                raise ValueError("All audio is silent.")
            return audio[..., :0]  # Return empty array

        if leading:
            start_sample_index = np.min(non_silent_np[-1]) # type: ignore[assignment]
        else:
            start_sample_index = 0

        if trailing:
            end_sample_index = np.max(non_silent_np[-1]) + 1 # type: ignore[assignment]
        else:
            end_sample_index = len(audio)

        return audio[..., start_sample_index:end_sample_index]

    else:
        raise ValueError("audio must be a torch tensor or numpy array.")
