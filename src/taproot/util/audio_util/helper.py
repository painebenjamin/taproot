from __future__ import annotations

import os

from typing import TYPE_CHECKING, Iterable, Tuple, Optional, List, Dict, Callable, Any, Union

from ..misc_util import reiterator

if TYPE_CHECKING:
    from moviepy.editor import ( # type: ignore[import-not-found,import-untyped,unused-ignore]
        AudioClip,
        AudioFileClip,
        CompositeAudioClip
    )

__all__ = ["Audio"]

class Audio:
    """
    A class to handle audio frames
    """
    def __init__(
        self,
        frames: Iterable[Tuple[float, ...]],
        rate: Optional[int]=None
    ) -> None:
        self.frames = reiterator(frames)
        self.rate = rate

    @property
    def frames_as_list(self) -> List[Tuple[float, ...]]:
        """
        Returns the frames as a list
        """
        return [frame for frame in self.frames] # type: ignore

    def get_clip(
        self,
        rate: Optional[int]=None,
        maximum_seconds: Optional[float]=None
    ) -> AudioClip:
        """
        Gets the moviepy audioclip
        """
        if not rate:
            rate = self.rate
        if not rate:
            rate = 44100

        from moviepy.editor import AudioClip

        all_frames: List[Tuple[float, ...]] = [frame for frame in self.frames] # type: ignore
        if maximum_seconds is not None:
            all_frames = all_frames[:int(maximum_seconds*rate)]

        total_frames = len(all_frames)
        duration = total_frames / rate

        def get_frame(time: float) -> Tuple[float, ...]:
            if isinstance(time, int) or isinstance(time, float):
                return all_frames[int(total_frames*time)]
            return [
                all_frames[int(t*rate)]
                for t in time
            ]

        return AudioClip(get_frame, duration=duration, fps=rate)

    def get_composite_clip(
        self,
        rate: Optional[int]=None,
        maximum_seconds: Optional[float]=None
    ) -> CompositeAudioClip:
        """
        Gets the moviepy composite audioclip
        """
        from moviepy.editor import CompositeAudioClip
        return CompositeAudioClip([
            self.get_clip(rate=rate, maximum_seconds=maximum_seconds)
        ])

    def save(
        self,
        path: str,
        rate: Optional[int]=None,
        maximum_seconds: Optional[float]=None
     ) -> int:
        """
        Saves the audio frames to file
        """
        if not rate:
            rate = self.rate
        if not rate:
            rate = 44100
        if path.startswith("~"):
            path = os.path.expanduser(path)
        clip = self.get_clip(rate=rate, maximum_seconds=maximum_seconds)
        clip.write_audiofile(path)
        if not os.path.exists(path):
            raise IOError(f"Nothing was written to {path}.")
        size = os.path.getsize(path)
        if size == 0:
            raise IOError(f"Nothing was written to {path}.")
        return size

    def get_normalized_frequencies(
        self,
        samples_per_second: int=8,
        maximum_samples: Optional[int]=None,
        rate: Optional[int]=None,
        low_filter: float = 0.0,
        num_frequency_bands: int = 256,
    ) -> Tuple[List[int], List[List[Tuple[float, ...]]]]:
        """
        Gets frequencies after normalizing
        """
        amplitude_dicts = [
            f for f in self.analyze(
                samples_per_second=samples_per_second,
                maximum_samples=maximum_samples,
                rate=rate,
                num_frequency_bands=num_frequency_bands
            )
        ]
        frequencies = list(amplitude_dicts[0].keys())
        num_frequency_bands = len(frequencies)
        low_pass = lambda f: 0 if f < low_filter else f
        amplitudes = [
            [
                tuple([low_pass(channel_value) for channel_value in amplitude_dict[freq]]) # type: ignore[no-untyped-call]
                for freq in frequencies
            ]
            for amplitude_dict in amplitude_dicts
            if len(list(amplitude_dict.keys())) == num_frequency_bands
        ]

        return [int(f) for f in frequencies], amplitudes

    def analyze(
        self,
        samples_per_second: int=60,
        maximum_samples: Optional[int]=None,
        rate: Optional[int]=None,
        num_frequency_bands: int=512,
    ) -> Iterable[Dict[int, List[float]]]:
        """
        Analyzes the audio stream
        """
        from .stream_analyzer import StreamAnalyzer
        if rate is None:
            rate = self.rate
        if rate is None:
            rate = 44100

        analyzers: List[StreamAnalyzer] = []

        samples_per_yield = int(rate / samples_per_second)
        samples: List[Tuple[float, ...]] = [(0.0,)] * samples_per_yield
        current_samples = 0
        yielded_samples = 0

        def get_frequencies() -> Dict[int, List[float]]:
            num_channels = len(samples[0])
            xf = None
            xy = []
            for i in range(num_channels):
                these_samples = [
                    samples[j][i]
                    for j in range(current_samples)
                ]
                if len(analyzers) <= i:
                    analyzers.append(
                        StreamAnalyzer(
                            rate=rate,
                            fft_window_size=samples_per_yield,
                            smoothing_length_ms=1000//samples_per_second,
                            n_frequency_bins=num_frequency_bands
                        )
                    )
                fftx, fft, fbc, fbe = analyzers[i](these_samples) # type: ignore[arg-type]
                if xf is None:
                    xf = fbc
                xy.append(fbe)

            return dict(zip(
                [int(f) for f in xf], # type: ignore[union-attr]
                [list(f) for f in zip(*xy)]
            ))

        for frame in self.frames: # type: ignore[attr-defined]
            samples[current_samples] = frame
            current_samples += 1
            if current_samples >= samples_per_yield:
                yield get_frequencies()
                yielded_samples += 1
                current_samples = 0
                if maximum_samples is not None and yielded_samples >= maximum_samples:
                    break
        if current_samples != 0:
            if maximum_samples is None or yielded_samples < maximum_samples:
                yield get_frequencies()

    @classmethod
    def file_to_frames(
        cls,
        path: str,
        skip_frames: Optional[int] = None,
        maximum_frames: Optional[int] = None,
        on_open: Optional[Callable[[AudioFileClip], None]] = None
    ) -> Iterable[Tuple[float, ...]]:
        """
        Starts an audio capture and yields tuples for each frame.
        """
        from moviepy.editor import AudioFileClip
        if path.startswith("~"):
            path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise IOError(f"Audio at path {path} not found or inaccessible")

        clip = AudioFileClip(path)
        if on_open is not None:
            on_open(clip)

        total_frames = 0
        for i, frame in enumerate(clip.iter_frames()):
            if skip_frames is not None and i < skip_frames:
                continue
            if maximum_frames is not None and total_frames + 1 > maximum_frames:
                break
            yield frame
            total_frames += 1

        if total_frames == 0:
            raise IOError(f"No frames were read from audio at path {path}")

    @classmethod
    def from_file(
        cls,
        path: str,
        skip_frames: Optional[int] = None,
        maximum_frames: Optional[int] = None,
        on_open: Optional[Callable[[AudioFileClip], None]] = None
    ) -> Audio:
        """
        Uses Audio.frames_from_file and instantiates an Audio object.
        """
        rate: Optional[int] = None

        def get_rate_on_open(clip: AudioFileClip) -> None:
            nonlocal rate
            rate = clip.fps
            if on_open is not None:
                on_open(clip)

        frames=cls.file_to_frames(
            path=path,
            skip_frames=skip_frames,
            maximum_frames=maximum_frames,
            on_open=get_rate_on_open
        )
        return cls(frames=frames, rate=rate)

    @classmethod
    def combine(
        cls,
        *audios: Union[str, List[Tuple[float, ...]]],
        **kwargs: Any
    ) -> Audio:
        """
        Combines multiple audio chunks
        """
        num_audios = len(audios)
        rate: Optional[int] = kwargs.get("rate", None)
        silence: Optional[float] = kwargs.get("silence", None)

        def maybe_get_rate_on_open(clip: AudioFileClip) -> None:
            nonlocal rate
            if rate is None:
                rate = clip.fps

        def iterate_audio() -> Iterable[Tuple[float, ...]]:
            nonlocal rate
            channels: Optional[int] = None
            for i, audio in enumerate(audios):
                if isinstance(audio, str):
                    frames = cls.file_to_frames(
                        path=audio,
                        on_open=maybe_get_rate_on_open
                    )
                else:
                    frames = audio
                for frame in frames:
                    if channels is None:
                        channels = len(frame)
                    yield frame
                if i < num_audios - 1 and silence:
                    silence_frames = (rate if rate else 44100) * silence
                    silence_channels = (channels if channels else 1)
                    for j in range(int(silence_frames)):
                        yield (0.0,) * silence_channels

        return cls(frames=iterate_audio(), rate=rate)
