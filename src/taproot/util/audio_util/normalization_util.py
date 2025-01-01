# Many of these methods are adapted from audiocraft under MIT License
# See https://github.com/facebookresearch/audiocraft/blob/main/LICENSE
from __future__ import annotations

import io
import tempfile
import subprocess

from typing import Tuple, Optional, List, Union, TYPE_CHECKING
from typing_extensions import Literal

from pathlib import Path
from dataclasses import dataclass

from ..log_util import logger
from ..download_util import retrieve_uri
from ..introspection_util import is_numpy_array, is_torch_tensor

if TYPE_CHECKING:
    import torch
    from ...hinting import AudioType

__all__ = [
    "AudioFileInfo",
    "EncodedAudioProxy",
    "audio_to_bct_tensor",
    "is_multi_audio",
    "convert_audio_channels",
    "convert_audio",
    "normalize_loudness",
    "clip_wav",
    "normalize_audio",
    "f32_pcm",
    "i16_pcm",
    "compress_audio",
    "get_mp3",
    "get_aac",
    "get_av_info",
    "get_soundfile_info",
    "audio_info",
    "audio_read",
    "av_read_audio",
    "pipe_to_ffmpeg",
    "audio_write"
]

@dataclass(frozen=True)
class AudioFileInfo:
    sample_rate: int
    duration: float
    channels: int

class EncodedAudioProxy:
    """
    A proxy to audio that is already encoded in a specific format.
    This avoids unnecessary re-encoding of audio data.
    """
    def __init__(
        self,
        data: bytes,
        format: str,
        sample_rate: Optional[int]=None,
        duration: Optional[float]=None,
        channels: Optional[int]=None
    ) -> None:
        self.data = data
        self.format = format
        self.passed_sample_rate = sample_rate
        self.passed_duration = duration
        self.passed_channels = channels

    def save(self, path: str) -> None:
        """
        Save the audio data to a file.
        """
        with open(path, 'wb') as f:
            f.write(self.data)

    @property
    def audio_info(self) -> AudioFileInfo:
        """
        Get audio information from the encoded audio data.
        """
        if not hasattr(self, "_audio_info"):
            # Write to file then read
            with tempfile.NamedTemporaryFile(mode='wb', suffix=f".{self.format}") as f:
                f.write(self.data)
                f.flush()
                self._audio_info = audio_info(f.name)
        return self._audio_info

    @property
    def channels(self) -> int:
        """
        Get the number of channels in the audio.
        """
        if self.passed_channels is not None:
            return self.passed_channels
        return self.audio_info.channels

    @property
    def sample_rate(self) -> int:
        """
        Get the sample rate of the audio.
        """
        if self.passed_sample_rate is not None:
            return self.passed_sample_rate
        return self.audio_info.sample_rate

    @property
    def duration(self) -> float:
        """
        Get the duration of the audio.
        """
        if self.passed_duration is not None:
            return self.passed_duration
        return self.audio_info.duration

def is_multi_audio(input_data: AudioType) -> bool:
    """
    Check if the input data is:
    - a list of audio data
    - a list of URIs
    - a 3-dimensional numpy array
    - a 3-dimensional torch tensor
    """
    if isinstance(input_data, list):
        return True
    if is_numpy_array(input_data) and len(input_data.shape) == 3:
        return True
    if is_torch_tensor(input_data) and len(input_data.shape) == 3:
        return True
    return False

def audio_to_bct_tensor(
    input_data: AudioType,
    sample_rate: Optional[int] = None,
    target_sample_rate: Optional[int] = None
) -> Tuple[torch.Tensor, Optional[int]]:
    """
    Convert audio data to a torch tensor of waveform samples
    in channel-first format with a batch dimension.
    """
    import torch
    import torchaudio # type: ignore[import-untyped]
    if isinstance(input_data, list):
        # Recursive case
        recursed_data: List[Tuple[torch.Tensor, int]] = [
            audio_to_bct_tensor(input_datum, sample_rate) # type: ignore[misc]
            for input_datum in input_data
        ]
        sample_rates = [sr for _, sr in recursed_data if sr is not None]
        if len(sample_rates) > 0 and sample_rate is None:
            sample_rate = sample_rates[0]
        return torch.cat([datum for datum, _ in recursed_data], dim=0), sample_rate
    if isinstance(input_data, EncodedAudioProxy):
        audio_data = io.BytesIO(input_data.data)
        waveform, sample_rate = torchaudio.load(audio_data)
        if sample_rate is None:
            sample_rate = input_data.sample_rate
    elif isinstance(input_data, str):
        audio_data = retrieve_uri(input_data) # type: ignore[assignment]
        waveform, sample_rate = torchaudio.load(audio_data)
    elif isinstance(input_data, (bytes, bytearray)):
        audio_data = io.BytesIO(input_data)
        waveform, sample_rate = torchaudio.load(audio_data)
    elif is_numpy_array(input_data):
        # Assume input_data is a numpy array of waveform samples
        if sample_rate is None:
            logger.warning("No sample rate provided for numpy array input. Assuming 44100 Hz.")
            sample_rate = 44100
        waveform = torch.from_numpy(input_data)
    elif is_torch_tensor(input_data):
        # Assume input_data is a torch tensor of waveform samples
        if sample_rate is None:
            logger.warning("No sample rate provided for torch tensor input. Assuming 44100 Hz.")
            sample_rate = 44100
        waveform = input_data
    else:
        raise ValueError(f"Unsupported input type {type(input_data).__name__}")

    # Normalize the waveform to the range [-1, 1]
    if waveform.dtype is torch.int16:
        waveform = waveform.float() / 32768.0
    elif waveform.dtype is torch.int8:
        waveform = (waveform.float() - 128) / 128.0

    if sample_rate is None:
        raise ValueError("No sample rate provided and could not infer from input data. Please provide a sample rate.")

    if target_sample_rate is not None and sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)
        sample_rate = target_sample_rate

    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0) # Add channel dimension
    if len(waveform.shape) == 2:
        waveform = waveform.unsqueeze(0) # Add batch dimension

    return waveform, sample_rate

def convert_audio_channels(
    wav: torch.Tensor,
    channels: int = 2
) -> torch.Tensor:
    """
    Convert audio to the given number of channels.

    Args:
        wav (torch.Tensor): Audio wave of shape [B, C, T].
        channels (int): Expected number of channels as output.
    Returns:
        torch.Tensor: Downmixed or unchanged audio wave [B, C, T].
    """
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, and the stream has multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file has
        # a single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file has
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav

def convert_audio(
    wav: torch.Tensor,
    from_rate: float,
    to_rate: float,
    to_channels: int
) -> torch.Tensor:
    """
    Convert audio to new sample rate and number of audio channels.
    """
    import julius # type: ignore[import-not-found,unused-ignore]
    wav = julius.resample_frac(wav, int(from_rate), int(to_rate))
    wav = convert_audio_channels(wav, to_channels)
    return wav

def normalize_loudness(
    wav: torch.Tensor,
    sample_rate: int,
    loudness_headroom_db: float = 14.0,
    loudness_compressor: bool = False,
    energy_floor: float = 2e-3
) -> torch.Tensor:
    """
    Normalize an input signal to a user loudness in dB LKFS.
    Audio loudness is defined according to the ITU-R BS.1770-4 recommendation.

    Args:
        wav (torch.Tensor): Input multichannel audio data.
        sample_rate (int): Sample rate.
        loudness_headroom_db (float): Target loudness of the output in dB LUFS.
        loudness_compressor (bool): Uses tanh for soft clipping.
        energy_floor (float): anything below that RMS level will not be rescaled.
    Returns:
        torch.Tensor: Loudness normalized output data.
    """
    import torch
    import torchaudio
    energy = wav.pow(2).mean().sqrt().item()
    if energy < energy_floor:
        return wav
    transform = torchaudio.transforms.Loudness(sample_rate)
    input_loudness_db = transform(wav).item()
    # calculate the gain needed to scale to the desired loudness level
    delta_loudness = -loudness_headroom_db - input_loudness_db
    gain = 10.0 ** (delta_loudness / 20.0)
    output = gain * wav
    if loudness_compressor:
        output = torch.tanh(output)
    assert output.isfinite().all(), (input_loudness_db, wav.pow(2).mean().sqrt())
    return output # type: ignore[no-any-return]

def clip_wav(
    wav: torch.Tensor,
    log_clipping: bool = False,
    stem_name: Optional[str] = None
) -> None:
    """
    Utility function to clip the audio with logging if specified.
    """
    max_scale = wav.abs().max()
    if log_clipping and max_scale > 1:
        from ..log_util import logger
        clamp_prob = (wav.abs() > 1).float().mean().item()
        logger.debug(
            f"CLIPPING {stem_name or ''} happening with probability {clamp_prob:%} (a bit of clipping is okay): maximum scale: {max_scale.item()}"
        )
    wav.clamp_(-1, 1)

def normalize_audio(
    wav: torch.Tensor,
    normalize: bool = True,
    strategy: Optional[Literal["clip", "peak", "rms", "loudness", "none"]] = "peak",
    peak_clip_headroom_db: float = 1.0,
    rms_headroom_db: float = 18.0,
    loudness_headroom_db: float = 14.0,
    loudness_compressor: bool = False,
    log_clipping: bool = False,
    sample_rate: Optional[int] = None,
    stem_name: Optional[str] = None
) -> torch.Tensor:
    """
    Normalize the audio according to the prescribed strategy (see after).

    Args:
        wav (torch.Tensor): Audio data.
        normalize (bool): if `True` (default), normalizes according to the prescribed
            strategy (see after). If `False`, the strategy is only used in case clipping
            would happen.
        strategy (str): Can be either 'clip', 'peak', or 'rms'. Default is 'peak',
            i.e. audio is normalized by its largest value. RMS normalizes by root-mean-square
            with extra headroom to avoid clipping. 'clip' just clips.
            # SIC: original comment omits 'loudness' and 'none' strategies
        peak_clip_headroom_db (float): Headroom in dB when doing 'peak' or 'clip' strategy.
        rms_headroom_db (float): Headroom in dB when doing 'rms' strategy. This must be much larger
            than the `peak_clip` one to avoid further clipping.
        loudness_headroom_db (float): Target loudness for loudness normalization.
        loudness_compressor (bool): If True, uses tanh based soft clipping.
        log_clipping (bool): If True, basic logging on stderr when clipping still
            occurs despite strategy (only for 'rms').
        sample_rate (int): Sample rate for the audio data (required for loudness).
        stem_name (str, optional): Stem name for clipping logging.
    Returns:
        torch.Tensor: Normalized audio.
    """
    scale_peak = 10 ** (-peak_clip_headroom_db / 20)
    scale_rms = 10 ** (-rms_headroom_db / 20)
    if strategy == 'peak':
        rescaling = (scale_peak / wav.abs().max())
        if normalize or rescaling < 1:
            wav = wav * rescaling
    elif strategy == 'clip':
        wav = wav.clamp(-scale_peak, scale_peak)
    elif strategy == 'rms':
        mono = wav.mean(dim=0)
        rescaling = scale_rms / mono.pow(2).mean().sqrt()
        if normalize or rescaling < 1:
            wav = wav * rescaling
        clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    elif strategy == 'loudness':
        assert sample_rate is not None, "Loudness normalization requires sample rate."
        wav = normalize_loudness(wav, sample_rate, loudness_headroom_db, loudness_compressor)
        clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    else:
        assert wav.abs().max() < 1
        assert strategy is None or strategy == 'none', f"Unexpected strategy: '{strategy}'"
    return wav

def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """
    Convert audio to float 32 bits PCM format.
    Args:
        wav (torch.tensor): Input wav tensor
    Returns:
        same wav in float32 PCM format
    """
    import torch
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / 2**15
    elif wav.dtype == torch.int32:
        return wav.float() / 2**31
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")

def i16_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to int 16 bits PCM format.

    ..Warning:: There exist many formula for doing this conversion. None are perfect
    due to the asymmetry of the int16 range. One either have possible clipping, DC offset,
    or inconsistencies with f32_pcm. If the given wav doesn't have enough headroom,
    it is possible that `i16_pcm(f32_pcm)) != Identity`.
    Args:
        wav (torch.tensor): Input wav tensor
    Returns:
        same wav in float16 PCM format
    """
    import torch
    if wav.dtype.is_floating_point:
        assert wav.abs().max() <= 1
        candidate = (wav * 2 ** 15).round()
        if candidate.max() >= 2 ** 15:  # clipping would occur
            candidate = (wav * (2 ** 15 - 1)).round()
        return candidate.short()
    else:
        assert wav.dtype == torch.int16
        return wav

def compress_audio(
    wav: torch.Tensor,
    sr: int,
    target_format: Literal["mp3", "ogg", "flac"] = "mp3",
    bits_per_sample: float = 128.0
) -> Tuple[torch.Tensor, int]:
    """Convert audio wave form to a specified lossy format: mp3, ogg, flac

    Args:
        wav (torch.Tensor): Input wav tensor.
        sr (int): Sampling rate.
        target_format (str): Compression format (e.g., 'mp3').
        bits_per_sample (float): Bitrate for the compression.

    Returns:
        Tuple of compressed WAV tensor and sampling rate.
    """
    try:
        import torchaudio
        # Create a virtual file instead of saving to disk
        buffer = io.BytesIO()

        torchaudio.save(
            buffer, wav, sr, format=target_format, bits_per_sample=bits_per_sample
        )
        # Move to the beginning of the file
        buffer.seek(0)
        compressed_wav, sr = torchaudio.load(buffer)
        return compressed_wav, sr

    except RuntimeError:
        from ..log_util import logger
        logger.warning(
            f"compression failed skipping compression: {format} {bits_per_sample}"
        )
        return wav, sr

def get_mp3(
    wav_tensor: torch.Tensor,
    sr: int,
    bits_per_sample: float = 128.0
) -> torch.Tensor:
    """Convert a batch of audio files to MP3 format, maintaining the original shape.

    This function takes a batch of audio files represented as a PyTorch tensor, converts
    them to MP3 format using the specified bitrate, and returns the batch in the same
    shape as the input.

    Args:
        wav_tensor (torch.Tensor): Batch of audio files represented as a tensor.
            Shape should be (batch_size, channels, length).
        sr (int): Sampling rate of the audio.
        bits_per_sample (float): Bitrate for the MP3 conversion.

    Returns:
        torch.Tensor: Batch of audio files converted to MP3 format, with the same
            shape as the input tensor.
    """
    import torch
    device = wav_tensor.device
    batch_size, channels, original_length = wav_tensor.shape

    # Flatten tensor for conversion and move to CPU
    wav_tensor_flat = wav_tensor.view(1, -1).cpu()

    # Convert to MP3 format with specified bitrate
    wav_tensor_flat, _ = compress_audio(wav_tensor_flat, sr, bits_per_sample=bits_per_sample)

    # Reshape back to original batch format and trim or pad if necessary
    wav_tensor = wav_tensor_flat.view(batch_size, channels, -1)
    compressed_length = wav_tensor.shape[-1]
    if compressed_length > original_length:
        wav_tensor = wav_tensor[:, :, :original_length]  # Trim excess frames
    elif compressed_length < original_length:
        padding = torch.zeros(
            batch_size, channels, original_length - compressed_length, device=device
        )
        wav_tensor = torch.cat((wav_tensor, padding), dim=-1)  # Pad with zeros

    # Move tensor back to the original device
    return wav_tensor.to(device)

def get_aac(
    wav_tensor: torch.Tensor,
    sr: int,
    bits_per_sample: float = 128.0,
    lowpass_freq: Optional[int] = None,
) -> torch.Tensor:
    """Converts a batch of audio tensors to AAC format and then back to tensors.

    This function first saves the input tensor batch as WAV files, then uses FFmpeg to convert
    these WAV files to AAC format. Finally, it loads the AAC files back into tensors.

    Args:
        wav_tensor (torch.Tensor): A batch of audio files represented as a tensor.
                                   Shape should be (batch_size, channels, length).
        sr (int): Sampling rate of the audio.
        bits_per_sample (float): Bitrate for the AAC conversion.
        lowpass_freq (Optional[int]): Frequency for a low-pass filter. If None, no filter is applied.

    Returns:
        torch.Tensor: Batch of audio files converted to AAC and back, with the same
                      shape as the input tensor.
    """
    import torch
    import torchaudio

    device = wav_tensor.device
    batch_size, channels, original_length = wav_tensor.shape

    # Flatten tensor for conversion and move to CPU
    wav_tensor_flat = wav_tensor.view(1, -1).cpu()

    with tempfile.NamedTemporaryFile(
        suffix=".wav"
    ) as f_in, tempfile.NamedTemporaryFile(suffix=".aac") as f_out:
        input_path, output_path = f_in.name, f_out.name

        # Save the tensor as a WAV file
        torchaudio.save(input_path, wav_tensor_flat, sr, backend="ffmpeg")

        # Prepare FFmpeg command for AAC conversion
        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ar",
            str(sr),
            "-b:a",
            f"{bits_per_sample:.0f}k",
            "-c:a",
            "aac",
        ]
        if lowpass_freq is not None:
            command += ["-cutoff", str(lowpass_freq)]
        command.append(output_path)

        try:
            # Run FFmpeg and suppress output
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Load the AAC audio back into a tensor
            aac_tensor, _ = torchaudio.load(output_path, backend="ffmpeg")
        except Exception as exc:
            raise RuntimeError(
                "Failed to run command " ".join(command)} "
                "(Often this means ffmpeg is not installed or the encoder is not supported, "
                "make sure you installed an older version ffmpeg<5)"
            ) from exc

    original_length_flat = batch_size * channels * original_length
    compressed_length_flat = aac_tensor.shape[-1]

    # Trim excess frames
    if compressed_length_flat > original_length_flat:
        aac_tensor = aac_tensor[:, :original_length_flat]

    # Pad the shortedn frames
    elif compressed_length_flat < original_length_flat:
        padding = torch.zeros(
            1, original_length_flat - compressed_length_flat, device=device
        )
        aac_tensor = torch.cat((aac_tensor, padding), dim=-1)

    # Reshape and adjust length to match original tensor
    wav_tensor = aac_tensor.view(batch_size, channels, -1)
    compressed_length = wav_tensor.shape[-1]

    assert compressed_length == original_length, (
        "AAC-compressed audio does not have the same frames as original one. "
        "One reason can be ffmpeg is not  installed and used as proper backed "
        "for torchaudio, or the AAC encoder is not correct. Run "
        "`torchaudio.utils.ffmpeg_utils.get_audio_encoders()` and make sure we see entry for"
        "AAC in the output."
    )
    return wav_tensor.to(device)

def get_av_info(filepath: Union[str, Path]) -> AudioFileInfo:
    """
    Get audio file information using PyAV bindings.
    Args:
        filepath (str or Path): Path to audio file.
    Returns:
        AudioFileInfo: Audio file information.
    """
    import av
    with av.open(str(filepath)) as af: # type: ignore[attr-defined,import-not-found,unused-ignore]
        stream = af.streams.audio[0]
        sample_rate = stream.codec_context.sample_rate
        duration = float(stream.duration * stream.time_base) # type: ignore[operator]
        channels = stream.channels
    return AudioFileInfo(sample_rate, duration, channels)

def get_soundfile_info(filepath: Union[str, Path]) -> AudioFileInfo:
    """
    Get audio file information using soundfile.
    Args:
        filepath (str or Path): Path to audio file.
    Returns:
        AudioFileInfo: Audio file information.
    """
    import soundfile # type: ignore[import-untyped]
    info = soundfile.info(filepath)
    return AudioFileInfo(info.samplerate, info.duration, info.channels)

def audio_info(filepath: Union[str, Path]) -> AudioFileInfo:
    # torchaudio no longer returns useful duration informations for some formats like mp3s.
    filepath = Path(filepath)
    if filepath.suffix in ['.flac', '.ogg']:  # TODO: Validate .ogg can be safely read with av_info
        # ffmpeg has some weird issue with flac.
        return get_soundfile_info(filepath)
    else:
        return get_av_info(filepath)

def av_read_audio(
    filepath: Union[str, Path],
    seek_time: float = 0,
    duration: float = -1.
) -> Tuple[torch.Tensor, int]:
    """
    FFMPEG-based audio file reading using PyAV bindings.
    Soundfile cannot read mp3 and av_read_audio is more efficient than torchaudio.

    Args:
        filepath (str or Path): Path to audio file to read.
        seek_time (float): Time at which to start reading in the file.
        duration (float): Duration to read from the file. If set to -1, the whole file is read.
    Returns:
        tuple of torch.Tensor, int: Tuple containing audio data and sample rate
    """
    import av
    with av.open(str(filepath)) as af: # type: ignore[attr-defined,import-not-found,unused-ignore]
        stream = af.streams.audio[0]
        sr = stream.codec_context.sample_rate
        num_frames = int(sr * duration) if duration >= 0 else -1
        frame_offset = int(sr * seek_time)
        # we need a small negative offset otherwise we get some edge artifact
        # from the mp3 decoder.
        af.seek(int(max(0, (seek_time - 0.1)) / stream.time_base), stream=stream) # type: ignore[operator]
        frames = []
        length = 0
        for frame in af.decode(streams=stream.index):
            current_offset = int(frame.rate * frame.pts * frame.time_base) # type: ignore[union-attr,operator]
            strip = max(0, frame_offset - current_offset)
            buf = torch.from_numpy(frame.to_ndarray()) # type: ignore[union-attr]
            if buf.shape[0] != stream.channels:
                buf = buf.view(-1, stream.channels).t()
            buf = buf[:, strip:]
            frames.append(buf)
            length += buf.shape[1]
            if num_frames > 0 and length >= num_frames:
                break
        assert frames
        # If the above assert fails, it is likely because we seeked past the end of file point,
        # in which case ffmpeg returns a single frame with only zeros, and a weird timestamp.
        # This will need proper debugging, in due time.
        wav = torch.cat(frames, dim=1)
        assert wav.shape[0] == stream.channels
        if num_frames > 0:
            wav = wav[:, :num_frames]
    return f32_pcm(wav), sr

def audio_read(
    filepath: Union[str, Path],
    seek_time: float = 0.,
    duration: float = -1.0,
    pad: bool = False
) -> Tuple[torch.Tensor, int]:
    """
    Read audio by picking the most appropriate backend tool based on the audio format.

    Args:
        filepath (str or Path): Path to audio file to read.
        seek_time (float): Time at which to start reading in the file.
        duration (float): Duration to read from the file. If set to -1, the whole file is read.
        pad (bool): Pad output audio if not reaching expected duration.
    Returns:
        tuple of torch.Tensor, int: Tuple containing audio data and sample rate.
    """
    fp = Path(filepath)
    if fp.suffix in ['.flac', '.ogg']:  # TODO: check if we can safely use av_read_audio for .ogg
        # There is some bug with ffmpeg and reading flac
        import soundfile
        import numpy as np
        import torch
        info = get_soundfile_info(filepath)
        frames = -1 if duration <= 0 else int(duration * info.sample_rate)
        frame_offset = int(seek_time * info.sample_rate)
        wav, sr = soundfile.read(filepath, start=frame_offset, frames=frames, dtype=np.float32)
        assert info.sample_rate == sr, f"Mismatch of sample rates {info.sample_rate} {sr}"
        wav = torch.from_numpy(wav).t().contiguous()
        if len(wav.shape) == 1:
            wav = torch.unsqueeze(wav, 0)
    else:
        wav, sr = av_read_audio(filepath, seek_time, duration)
    if pad and duration > 0:
        import torchaudio.functional as F # type: ignore[import-untyped]
        expected_frames = int(duration * sr)
        wav = F.pad(wav, (0, expected_frames - wav.shape[-1]))
    return wav, sr

def pipe_to_ffmpeg(
    out_path: Union[str, Path],
    wav: torch.Tensor,
    sample_rate: int,
    flags: List[str]
) -> None:
    # ffmpeg is always installed and torchaudio is a bit unstable lately, so let's bypass it entirely.
    assert wav.dim() == 2, wav.shape
    command = [
        'ffmpeg',
        '-loglevel', 'error',
        '-y', '-f', 'f32le', '-ar', str(sample_rate), '-ac', str(wav.shape[0]),
        '-i', '-'
    ] + flags + [str(out_path)]
    input_ = f32_pcm(wav).t().detach().cpu().numpy().tobytes()
    subprocess.run(command, input=input_, check=True)

def audio_write(
    stem_name: Union[str, Path],
    wav: torch.Tensor,
    sample_rate: int,
    format: str = 'wav',
    mp3_rate: int = 320,
    ogg_rate: Optional[int] = None,
    normalize: bool = True,
    strategy: Optional[Literal["clip", "peak", "rms", "loudness", "none"]] = "peak",
    peak_clip_headroom_db: float = 1.0,
    rms_headroom_db: float = 18.0,
    loudness_headroom_db: float = 14.0,
    loudness_compressor: bool = False,
    log_clipping: bool = True,
    make_parent_dir: bool = True,
    add_suffix: bool = True
) -> Path:
    """
    Convenience function for saving audio to disk. Returns the filename the audio was written to.

    Args:
        stem_name (str or Path): Filename without extension which will be added automatically.
        wav (torch.Tensor): Audio data to save.
        sample_rate (int): Sample rate of audio data.
        format (str): Either "wav", "mp3", "ogg", or "flac".
        mp3_rate (int): kbps when using mp3s.
        ogg_rate (int): kbps when using ogg/vorbis. If not provided, let ffmpeg decide for itself.
        normalize (bool): if `True` (default), normalizes according to the prescribed
            strategy (see after). If `False`, the strategy is only used in case clipping
            would happen.
        strategy (str): Can be either 'clip', 'peak', or 'rms'. Default is 'peak',
            i.e. audio is normalized by its largest value. RMS normalizes by root-mean-square
            with extra headroom to avoid clipping. 'clip' just clips.
            # SIC: original comment omits 'loudness' and 'none' strategies
        peak_clip_headroom_db (float): Headroom in dB when doing 'peak' or 'clip' strategy.
        rms_headroom_db (float): Headroom in dB when doing 'rms' strategy. This must be much larger
            than the `peak_clip` one to avoid further clipping.
        loudness_headroom_db (float): Target loudness for loudness normalization.
        loudness_compressor (bool): Uses tanh for soft clipping when strategy is 'loudness'.
         when strategy is 'loudness' log_clipping (bool): If True, basic logging on stderr when clipping still
            occurs despite strategy (only for 'rms').
        make_parent_dir (bool): Make parent directory if it doesn't exist.
    Returns:
        Path: Path of the saved audio.
    """
    assert wav.dtype.is_floating_point, "wav is not floating point"
    if wav.dim() == 1:
        wav = wav[None]
    elif wav.dim() > 2:
        raise ValueError("Input wav should be at most 2 dimension.")
    assert wav.isfinite().all()
    wav = normalize_audio(
        wav, normalize, strategy, peak_clip_headroom_db,
        rms_headroom_db, loudness_headroom_db, loudness_compressor,
        log_clipping=log_clipping, sample_rate=sample_rate,
        stem_name=str(stem_name)
    )
    if format == 'mp3':
        suffix = '.mp3'
        flags = ['-f', 'mp3', '-c:a', 'libmp3lame', '-b:a', f'{mp3_rate}k']
    elif format == 'wav':
        suffix = '.wav'
        flags = ['-f', 'wav', '-c:a', 'pcm_s16le']
    elif format == 'ogg':
        suffix = '.ogg'
        flags = ['-f', 'ogg', '-c:a', 'libvorbis']
        if ogg_rate is not None:
            flags += ['-b:a', f'{ogg_rate}k']
    elif format == 'flac':
        suffix = '.flac'
        flags = ['-f', 'flac']
    else:
        raise RuntimeError(f"Invalid format {format}. Only wav or mp3 are supported.")
    if not add_suffix:
        suffix = ''
    path = Path(str(stem_name) + suffix)
    if make_parent_dir:
        path.parent.mkdir(exist_ok=True, parents=True)
    try:
        pipe_to_ffmpeg(path, wav, sample_rate, flags)
    except Exception:
        if path.exists():
            # we do not want to leave half written files around.
            path.unlink()
        raise
    return path
