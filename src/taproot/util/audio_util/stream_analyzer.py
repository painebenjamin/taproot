import math

from typing import Any, Tuple, TYPE_CHECKING

from ..log_util import logger
from ..numpy_util import NumpyDataBuffer

if TYPE_CHECKING:
    import numpy as np
    
class StreamAnalyzer:
    """
    The StreamAnalyzer class processes real-time audio data through FFT and provides
    methods for extracting and smoothing frequency components.
    """
    def __init__(self, rate: float, fft_window_size: int, smoothing_length_ms: int = 50, n_frequency_bins: int = 256) -> None:
        """
        Initializes the StreamAnalyzer with custom settings for audio analysis.

        Args:
            rate (float): The sample rate of the audio signal.
            fft_window_size (int): The size of the window used for FFT calculations.
            smoothing_length_ms (int): The length of the smoothing window in milliseconds.
            n_frequency_bins (int): The number of frequency bins to use.
        """
        import numpy as np
        self.n_frequency_bins = n_frequency_bins
        self.rate = rate
        self.rolling_stats_window_s = 20
        self.equalizer_strength = 0.20
        self.apply_frequency_smoothing = True

        if self.apply_frequency_smoothing:
            self.filter_width = self._round_up_to_even(0.03 * self.n_frequency_bins) - 1

        self.fft_window_size = fft_window_size
        self.fft_window_size_ms = 1000 * self.fft_window_size / self.rate
        self.fft = np.ones(int(self.fft_window_size / 2), dtype=float)
        self.fftx = np.arange(int(self.fft_window_size / 2), dtype=float) * self.rate / self.fft_window_size

        self.smoothing_length_ms = smoothing_length_ms
        if self.smoothing_length_ms > 0:
            self.smoothing_kernel = self._get_smoothing_filter(self.fft_window_size_ms, self.smoothing_length_ms)
            self.feature_buffer = NumpyDataBuffer(len(self.smoothing_kernel), len(self.fft), dtype=np.float32, data_dimensions=2)

        self.fftx_bin_indices = self._calculate_bin_indices(self.n_frequency_bins, len(self.fftx))
        self.frequency_bin_energies = np.zeros(self.n_frequency_bins)
        self.frequency_bin_centres = np.zeros(self.n_frequency_bins)
        self.fftx_indices_per_bin = [np.where(self.fftx_bin_indices == bin_index)[0] for bin_index in range(self.n_frequency_bins)]
        for bin_index in range(self.n_frequency_bins):
            fftx_frequencies_this_bin = self.fftx[self.fftx_indices_per_bin[bin_index]]
            self.frequency_bin_centres[bin_index] = np.mean(fftx_frequencies_this_bin)

        self.fft_fps = 30
        self.log_features = False
        self.num_ffts = 0
        self.strongest_frequency = 0

        self.power_normalization_coefficients = np.logspace(np.log2(1), np.log2(np.log2(self.rate / 2)), len(self.fftx), endpoint=True, base=2, dtype=None)
        self.rolling_stats_window_n = self.rolling_stats_window_s * self.fft_fps
        self.rolling_bin_values = NumpyDataBuffer(self.rolling_stats_window_n, self.n_frequency_bins, start_value=25000)
        self.bin_mean_values = np.ones(self.n_frequency_bins)

    """Private Static Methods"""

    @staticmethod
    def _get_fft(
        data: np.ndarray[Any, Any],
        rate: float,
        chunk_size: int,
        log_scale: bool = False
    ) -> np.ndarray[Any, Any]:
        """
        Calculates the Fast Fourier Transform (FFT) of the given data.
        """
        import numpy as np
        data *= np.hamming(len(data))
        try:
            fft = np.abs(np.fft.rfft(data)[1:])
        except:
            fft = np.fft.fft(data)
            left, right = np.split(np.abs(fft), 2)
            fft = np.add(left, right[::-1])

        if log_scale:
            try:
                fft = 20 * np.log10(fft)
            except Exception as e:
                logger.error(f'Log(FFT) failed: {str(e)}')

        return fft # type: ignore[no-any-return,unused-ignore]

    @staticmethod
    def _round_up_to_even(f: float) -> int:
        """Rounds up the given float to the nearest even integer."""
        return int(math.ceil(f / 2.) * 2)

    @staticmethod
    def _gaussian_kernel1d(sigma: float, truncate: float = 2.0) -> np.ndarray[Any, Any]:
        """
        Generates a Gaussian kernel for smoothing.
        """
        import numpy as np
        sigma2 = sigma * sigma
        radius = int(truncate * sigma + 0.5)
        x = np.arange(-radius, radius + 1)
        phi_x = np.exp(-0.5 / sigma2 * x ** 2)
        phi_x /= np.sum(phi_x)
        return phi_x # type: ignore[no-any-return,unused-ignore]

    @staticmethod
    def _get_smoothing_filter(fft_window_size_ms: float, filter_length_ms: float) -> np.ndarray[Any, Any]:
        """
        Calculates a smoothing filter based on FFT window size and filter length.
        """
        import numpy as np
        buffer_length = StreamAnalyzer._round_up_to_even(filter_length_ms / fft_window_size_ms) + 1
        filter_sigma = buffer_length / 3
        filter_weights = StreamAnalyzer._gaussian_kernel1d(filter_sigma)[:, np.newaxis]

        max_index = np.argmax(filter_weights)
        filter_weights = filter_weights[:max_index + 1]
        filter_weights /= np.mean(filter_weights)

        return filter_weights

    @staticmethod
    def _calculate_bin_indices(n_frequency_bins: int, fft_size: int) -> np.ndarray[Any, Any]:
        """
        Calculates the indices for frequency bins.
        """
        bin_indices = np.logspace(np.log2(fft_size), 0, fft_size, endpoint=True, base=2, dtype=None) - 1
        bin_indices = -np.round((bin_indices - np.max(bin_indices)) * -1 / (fft_size / n_frequency_bins), 0).astype(int)
        bin_indices = np.minimum(np.arange(len(bin_indices)), bin_indices - np.min(bin_indices))
        return bin_indices

    """Public Methods"""

    def update_rolling_stats(self) -> None:
        """
        Updates the rolling statistics for frequency bin energies.
        """
        import numpy as np
        self.rolling_bin_values.append_data(self.frequency_bin_energies)
        self.bin_mean_values = np.mean(self.rolling_bin_values.get_buffer_data(), axis=0)
        self.bin_mean_values = np.maximum((1 - self.equalizer_strength) * np.mean(self.bin_mean_values), self.bin_mean_values)

    def update_features(self, latest_data_window: np.ndarray[Any, Any]) -> None:
        """
        Processes the latest data window to update FFT and frequency features.
        """
        import numpy as np
        self.fft = self._get_fft(latest_data_window, self.rate, self.fft_window_size, log_scale=self.log_features)
        self.fft *= self.power_normalization_coefficients
        self.num_ffts += 1

        if self.smoothing_length_ms > 0:
            self.feature_buffer.append_data(self.fft)
            buffered_features = self.feature_buffer.get_most_recent(len(self.smoothing_kernel))
            if len(buffered_features) == len(self.smoothing_kernel):
                buffered_features = self.smoothing_kernel * buffered_features
                self.fft = np.mean(buffered_features, axis=0)

        self.strongest_frequency = self.fftx[np.argmax(self.fft)]
        for bin_index in range(self.n_frequency_bins):
            self.frequency_bin_energies[bin_index] = np.mean(self.fft[self.fftx_indices_per_bin[bin_index]])

    def __call__(self, data_window: np.ndarray[Any, Any]) -> Tuple[
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ]:
        """
        Updates the analyzer with a new data window and returns the processed FFT data.
        """
        import numpy as np
        self.update_features(data_window)
        self.update_rolling_stats()

        self.frequency_bin_energies = np.nan_to_num(self.frequency_bin_energies, copy=True)
        if self.apply_frequency_smoothing and self.filter_width > 3:
            from scipy.signal import savgol_filter # type: ignore[import-untyped]
            self.frequency_bin_energies = savgol_filter(self.frequency_bin_energies, self.filter_width, 3)

        self.frequency_bin_energies /= self.bin_mean_values
        self.frequency_bin_energies = np.clip(self.frequency_bin_energies, 0, 1)
        return self.fftx, self.fft, self.frequency_bin_centres, self.frequency_bin_energies
