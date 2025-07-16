import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class DatasetConfig:
    name: str
    signal_path: str
    trigger_path: str
    fs: float
    stft_f_max_hz: float
    nperseg: int
    noverlap: int
    binary_threshold: Optional[float] = None
    # Motif analysis parameters
    target_length_points: Optional[int] = None
    target_count: Optional[int] = None
    threshold: Optional[float] = None
    downsample_factor: Optional[int] = 9000
    use_center: Optional[float] = None

def calculate_stft(signal_data: np.ndarray, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate STFT of the signal data.
    
    Args:
        signal_data: Input signal array
        config: Dataset configuration containing STFT parameters
        
    Returns:
        Tuple of (frequencies, time, STFT_matrix)
    """
    print("Calculating STFT...")
    f, t, Zxx = signal.stft(signal_data, fs=config.fs, nperseg=config.nperseg, noverlap=config.noverlap)
    return f, t, Zxx

def process_spectrogram(f: np.ndarray, Zxx: np.ndarray, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process spectrogram and create vertical projection.
    
    Args:
        f: Frequency array from STFT
        Zxx: STFT matrix
        config: Dataset configuration containing frequency limits
        
    Returns:
        Tuple of (sliced_frequencies, magnitude_spectrogram_dB, vertical_projection)
    """
    print("Processing spectrogram and creating projection...")
    freq_slice = np.where(f <= config.stft_f_max_hz)[0]
    f_sliced = f[freq_slice]
    Zxx_sliced = Zxx[freq_slice, :]
    Zxx_mag_db = 20 * np.log10(np.abs(Zxx_sliced) + 1e-9)
    vertical_projection = np.sum(np.abs(Zxx_sliced), axis=0)
    return f_sliced, Zxx_mag_db, vertical_projection
