from __future__ import annotations

import warnings

from typing import Tuple, List, Union, Dict, Sequence, Optional, TYPE_CHECKING

from ..package_util import binary_is_available

if TYPE_CHECKING:
    import torch

__all__ = [
    "EqualizerType",
    "CompandType",
    "compand_presets",
    "equalizer_presets",
    "standardize_equalizer",
    "standardize_compand",
    "sox_is_available",
    "pitch_shift_audio",
    "equalize_audio",
    "rms_normalize_audio",
]

"""
Equalizer format is (freq, width, gain.) Frequency is always in Hz, and gain is in dB.
The width parameter can be specified in one of the following ways:
    q: The number is a Q factor. (The Q factor is defined as the center frequency divided by the 3 dB bandwidth.)
       So for a filter centered at 250 Hz, a Q of 1.0 gives a 3 dB bandwidth of roughly 250 Hz.
    o: The number is in octaves. For example, 1.0o means the filter’s bandwidth spans one octave.
    h: The number specifies the half-power (or -3 dB) bandwidth directly.
    k: The number is given in kilohertz.
"""
EqualizerType = Union[
    str,
    Tuple[
        Union[str, int], # freq(Hz)
        Union[str, float, Tuple[float, str]], # width(+unit)
        Union[str, float], # gain(dB)
    ]
]
CompandType = Union[ # Dynamic-range compression/expansion
    str,
    Tuple[
        Union[str, Tuple[float, ...]], # attack1,decay1{,attack2,decay2}
        Union[str, Tuple[int, ...], Tuple[int, Tuple[int, ...]]], # [soft-knee-dB:]in-dB1[,out-dB1]{,in-dB2,out-dB2}
    ],
    Tuple[
        Union[str, Tuple[float, ...]], # attack1,decay1{,attack2,decay2}
        Union[str, Tuple[int, ...], Tuple[int, Tuple[int, ...]]], # [soft-knee-dB:]in-dB1[,out-dB1]{,in-dB2,out-dB2}
        Union[str, float], # gain
    ],
    Tuple[
        Union[str, Tuple[float, ...]], # attack1,decay1{,attack2,decay2}
        Union[str, Tuple[int, ...], Tuple[int, Tuple[int, ...]]], # [soft-knee-dB:]in-dB1[,out-dB1]{,in-dB2,out-dB2}
        Union[str, float], # gain
        Union[str, int], # initial-volume-dB
    ],
    Tuple[
        Union[str, Tuple[float, ...]], # attack1,decay1{,attack2,decay2}
        Union[str, Tuple[int, ...], Tuple[int, Tuple[int, ...]]], # [soft-knee-dB:]in-dB1[,out-dB1]{,in-dB2,out-dB2}
        Union[str, float], # gain
        Union[str, int], # initial-volume-dB
        Union[str, float], # delay
    ],
]

# Presets
compand_presets: Dict[str, Sequence[CompandType]] = {
    "voice": [
        ((0.3, 0.8), (6, (-80, -70, -30)), -8., -90, 0.2),
    ],
    "music": [
        ((0.3, 1.0), (6, (-70, -60, -20)), -5., -90, 0.2),
    ],
    "movie": [
        ((0.0, 1.0), (6, (-70, -43, -20)), -6., -90, 0.1),
    ],
}
equalizer_presets: Dict[str, Sequence[EqualizerType]] = {
    "voice": [
        (250, (1.0, "q"), 0.5),
        (4000, (1.0, "q"), -6.0),
    ]
}

def standardize_equalizer(equalizer: EqualizerType) -> List[str]:
    """
    Standardize the equalizer settings into a string list that can be passed to the sox command.

    :param equalizer: The equalizer settings to standardize.
    :return: The standardized equalizer settings.
    """
    equalizer_tuple: EqualizerType
    if isinstance(equalizer, str):
        equalizer_tuple = equalizer.split(" ") # type: ignore[assignment]
        assert len(equalizer_tuple) == 3, f"Invalid equalizer format: {equalizer}"
    else:
        equalizer_tuple = equalizer

    freq, width, gain = equalizer_tuple # type: ignore[misc]
    if isinstance(width, tuple):
        width = f"{width[0]}{width[1]}"

    return [str(freq), str(width), str(gain)]

def standardize_compand(compand: CompandType) -> List[str]:
    """
    Standardize the compand settings into a string list that can be passed to the sox command.

    :param compand: The compand settings to standardize.
    :return: The standardized compand settings.
    """
    compand_tuple: CompandType
    if isinstance(compand, str):
        compand_tuple = compand.split(" ") # type: ignore[assignment]
        assert len(compand_tuple) in {2, 3, 4, 5}, f"Invalid compand format: {compand}"
    else:
        compand_tuple = compand

    if len(compand_tuple) == 2:
        attack_decay, in_out = compand_tuple # type: ignore[misc]
        gain = None
        initial_volume = None
        delay = None
    elif len(compand_tuple) == 3:
        attack_decay, in_out, gain = compand_tuple # type: ignore[misc]
        initial_volume = None
        delay = None
    elif len(compand_tuple) == 4:
        attack_decay, in_out, gain, initial_volume = compand_tuple # type: ignore[misc]
        delay = None
    else:
        attack_decay, in_out, gain, initial_volume, delay = compand_tuple # type: ignore[misc]

    attack_decay_list: List[str] = []
    if isinstance(attack_decay, str):
        attack_decay_list = attack_decay.split(",")
    else:
        attack_decay_list = [str(a_d) for a_d in attack_decay]

    assert len(attack_decay_list) in {2, 4}, f"Invalid attack/decay format: {attack_decay}"
    attack_decay_str = ",".join([
        "{:.1f}".format(float(a_d))
        for a_d in attack_decay_list
    ])

    soft_knee: Optional[str] = None
    in_out_list: List[str] = []
    if isinstance(in_out, str):
        if ":" in in_out:
            soft_knee, in_out = in_out.split(":")
        in_out_list = in_out.split(",")
    elif isinstance(in_out, tuple) and isinstance(in_out[-1], tuple):
        soft_knee, in_out = in_out # type: ignore[assignment]
        in_out_list = [str(io) for io in in_out] # type: ignore[union-attr]
    else:
        in_out_list = [str(io) for io in in_out]

    assert len(in_out_list) in {1, 3}, f"Invalid in/out format: {in_out}"
    in_out_db_str = ",".join([
        "{:d}".format(int(io))
        for io in in_out_list
    ])
    if soft_knee is not None:
        in_out_str = f"{int(soft_knee):d}:{in_out_db_str}"
    else:
        in_out_str = in_out_db_str

    gain_str: Optional[str] = None
    if gain is not None:
        gain_str = "{:d}".format(int(gain))

    initial_volume_str: Optional[str] = None
    if initial_volume is not None:
        initial_volume_str = "{:d}".format(int(initial_volume))

    delay_str: Optional[str] = None
    if delay is not None:
        delay_str = "{:.1f}".format(float(delay))

    arg_list: List[str] = [attack_decay_str, in_out_str]
    if gain_str is not None:
        arg_list.append(gain_str)
        if initial_volume_str is not None:
            arg_list.append(initial_volume_str)
            if delay_str is not None:
                arg_list.append(delay_str)

    return arg_list

def sox_is_available() -> bool:
    """
    Check if sox is available.

    :return: True if sox is available, False otherwise.
    """
    return binary_is_available("sox")

def pitch_shift_audio(
    audio_tensor: torch.Tensor,
    sample_rate: int,
    pitch_shift: Union[str, float],
    raise_on_error: bool=True,
) -> Tuple[torch.Tensor, int]:
    """
    Pitch shift audio with the given pitch shift value.

    :param audio_tensor: The audio tensor to pitch shift.
    :param sample_rate: The sample rate of the audio tensor.
    :param pitch_shift: The pitch shift value to use.
    :return: The pitch shifted audio tensor and the sample rate.
    """
    if not sox_is_available():
        warnings.warn("sox is not available, pitch shift will not be applied.")
        return audio_tensor, sample_rate

    from torchaudio.sox_effects import apply_effects_tensor # type: ignore[import-untyped]

    effects_chain: List[List[str]] = [
        ["pitch", str(pitch_shift)],
    ]

    try:
        # apply_effects_tensor requires 2d, but we return the same shape as input
        unsqueeze_squeeze = audio_tensor.ndim == 1
        squeeze_unsqueeze = audio_tensor.ndim == 3

        if unsqueeze_squeeze:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif squeeze_unsqueeze:
            audio_tensor = audio_tensor.squeeze(0)

        enhanced_audio, enhanced_sample_rate = apply_effects_tensor(audio_tensor, sample_rate, effects_chain)

        if unsqueeze_squeeze:
            enhanced_audio = enhanced_audio.squeeze(0)
        elif squeeze_unsqueeze:
            enhanced_audio = enhanced_audio.unsqueeze(0)

        return enhanced_audio, enhanced_sample_rate
    except Exception as e:
        if raise_on_error:
            raise e
        return audio_tensor, sample_rate

def equalize_audio(
    audio_tensor: torch.Tensor,
    sample_rate: int,
    norm_level: Optional[int]=-8,
    high_pass_hz: Optional[int]=80,
    low_pass_hz: Optional[int]=6000,
    preset: Optional[str]=None,
    equalizers: Optional[Sequence[EqualizerType]]=None,
    compand: Optional[Sequence[CompandType]]=None,
    raise_on_error: bool=True,
) -> Tuple[torch.Tensor, int]:
    """
    Equalize audio with the given equalizers and compand settings, in the following order:

    norm -> highpass -> equalizers -> lowpass -> compand -> norm

    Any of the settings can be omitted, in which case the corresponding step is skipped.

    :param audio_tensor: The audio tensor to equalize.
    :param sample_rate: The sample rate of the audio tensor.
    :param norm_level: The normalization level in dB. If None, no normalization is applied.
    :param high_pass_hz: The high pass filter frequency in Hz. If None, no high pass filter is applied.
    :param low_pass_hz: The low pass filter frequency in Hz. If None, no low pass filter is applied.
    :param preset: The preset to use for the equalizer and compand settings.
    :param equalizers: The equalizer settings to use.
    :param compand: The compand settings to use.
    :return: The equalized audio tensor and the sample rate.
    """
    if not sox_is_available():
        warnings.warn("sox is not available, equalization will not be applied.")
        return audio_tensor, sample_rate

    from torchaudio.sox_effects import apply_effects_tensor

    if preset is not None:
        if equalizers is None:
            equalizers = equalizer_presets.get(preset, [])
        if compand is None:
            compand = compand_presets.get(preset, [])

    effects_chain: List[List[str]] = []

    if norm_level is not None:
        effects_chain.append(["norm", str(norm_level)])
    if high_pass_hz is not None:
        effects_chain.append(["highpass", str(high_pass_hz)])
    if equalizers is not None:
        for equalizer_params in equalizers:
            effects_chain.append(["equalizer", *standardize_equalizer(equalizer_params)])
    if low_pass_hz is not None:
        effects_chain.append(["lowpass", str(low_pass_hz)])
    if compand is not None:
        for compand_params in compand:
            effects_chain.append(["compand", *standardize_compand(compand_params)])
    if norm_level is not None:
        effects_chain.append(["norm", str(norm_level)])

    try:
        # apply_effects_tensor requires 2d, but we return the same shape as input
        unsqueeze_squeeze = audio_tensor.ndim == 1
        squeeze_unsqueeze = audio_tensor.ndim == 3

        if unsqueeze_squeeze:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif squeeze_unsqueeze:
            audio_tensor = audio_tensor.squeeze(0)

        enhanced_audio, enhanced_sample_rate = apply_effects_tensor(audio_tensor, sample_rate, effects_chain)

        if unsqueeze_squeeze:
            enhanced_audio = enhanced_audio.squeeze(0)
        elif squeeze_unsqueeze:
            enhanced_audio = enhanced_audio.unsqueeze(0)

        return enhanced_audio, enhanced_sample_rate
    except Exception as e:
        if raise_on_error:
            raise e
        return audio_tensor, sample_rate

def rms_normalize_audio(
    audio_tensor: torch.Tensor,
    target_rms: float=0.1
) -> torch.Tensor:
    """
    Normalize the audio tensor to the target RMS value.

    :param audio_tensor: The audio tensor to normalize.
    :param target_rms: The target RMS value to normalize to.
    :return: The normalized audio tensor.
    """
    import torch
    rms = torch.sqrt(torch.mean(torch.square(audio_tensor)))
    if rms == 0:
        return audio_tensor
    return audio_tensor * (target_rms / rms) # type: ignore[no-any-return]
