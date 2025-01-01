from __future__ import annotations

from typing import Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "mel_spectrogram",
    "mel_spectrogram_windowed",
    "save_spectrograms"
]

def mel_spectrogram_windowed(
    audio: np.ndarray[Any, Any],
    sample_rate: int=16000,
    window_length: int=25,
    window_step: int=10,
    n_channels: int=40
) -> np.ndarray[Any, Any]:
    """
    Compute the mel spectrogram of an audio signal.
    Mel is a scale that is more perceptually relevant than the linear scale.
    """
    import numpy as np
    import librosa
    frames = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_channels,
        n_fft=int(sample_rate * window_length / 1000),
        hop_length=int(sample_rate * window_step / 1000),
    )
    return frames.astype(np.float32).T

def mel_spectrogram(
    audio: np.ndarray[Any, Any],
    sample_rate: int=16000,
    num_samples_per_fft: int=4096,
    num_samples_between_frames: int=128,
    duration: float=8
) -> np.ndarray[Any, Any]:
    """
    Get the mel-spectrogram from the raw audio.
    """
    import numpy as np
    import librosa
    spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=num_samples_per_fft,
        hop_length=num_samples_between_frames,
    )
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db # type: ignore[no-any-return,unused-ignore]

def save_spectrograms(
    audios: List[np.ndarray[Any, Any]],
    path: str,
    names: List[str],
    sample_rate: int=16000,
    num_samples_per_fft: int=4096,
    num_samples_between_frames: int=128,
    duration: float=8
) -> None:
    """Plot a spectrogram for an audio file.

    Args:
        audios: List of audio spectrograms
        sr (int): Sampling rate of the audio file. Default is 22050 Hz.
        path (str): Path to the plot file.
        names: name of each spectrogram plot
        n_fft (int): Number of samples per FFT. Default is 2048.
        hop_length (int): Number of samples between successive frames. Default is 512.
        dur (float): Maxium duration to plot the spectrograms

    Returns:
        None (plots the spectrogram using matplotlib)
    """
    import numpy as np
    import librosa
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import librosa.display

    if not names:
        names = ["Ground Truth", "Audio Watermarked", "Watermark"]
    audios = [wav[: int(duration * sample_rate)] for wav in audios]  # crop
    assert len(names) == len(audios), f"There are {len(audios)} wavs but {len(names)} names ({names})"

    # Set matplotlib stuff
    BIGGER_SIZE = 10
    SMALLER_SIZE = 8
    linewidth = 234.8775  # linewidth in pt

    plt.rc("font", size=BIGGER_SIZE, family="serif")  # controls default text sizes
    plt.rcParams["font.family"] = "DeJavu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALLER_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)
    height = 1.6 * linewidth / 72.0
    fig, ax = plt.subplots(
        nrows=len(audios),
        ncols=1,
        sharex=True,
        figsize=(linewidth / 72.0, height),
    )
    fig.tight_layout()

    # Plot the spectrogram

    for i, audio in enumerate(audios):
        spectrogram_db = mel_spectrogram(
            audio,
            sample_rate=sample_rate,
            num_samples_per_fft=num_samples_per_fft,
            num_samples_between_frames=num_samples_between_frames,
            duration=duration,
        )

        if i == 0:
            cax = fig.add_axes(
                [
                    ax[0].get_position().x1 + 0.01,  # type: ignore
                    ax[-1].get_position().y0,
                    0.02,
                    ax[0].get_position().y1 - ax[-1].get_position().y0,
                ]
            )
            fig.colorbar(
                mpl.cm.ScalarMappable(
                    norm=mpl.colors.Normalize(
                        np.min(spectrogram_db), np.max(spectrogram_db)
                    ),
                    cmap="magma",
                ),
                ax=ax,
                orientation="vertical",
                format="%+2.0f dB",
                cax=cax,
            )
        librosa.display.specshow(
            spectrogram_db,
            sr=sample_rate,
            hop_length=num_samples_between_frames,
            x_axis="time",
            y_axis="mel",
            ax=ax[i],
        )
        ax[i].set(title=names[i])
        ax[i].yaxis.set_label_text(None)
        ax[i].label_outer()

    fig.savefig(path, bbox_inches="tight")
    plt.close()
