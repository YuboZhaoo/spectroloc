import numpy as np
import os
import time
from scipy import signal
from scipy.spatial.distance import cosine
from dataclasses import dataclass
from typing import Optional, Tuple
from .proj import DatasetConfig, calculate_stft

@dataclass
class AnalysisResult:
    """Analysis result data class for template matching"""
    signal_data: np.ndarray
    template: Optional[np.ndarray]
    match_score: Optional[np.ndarray]
    t_corr: Optional[np.ndarray]
    total_samples: int

def load_signal_data(config: DatasetConfig) -> Optional[np.ndarray]:
    """Load signal data from file.
    
    Args:
        config: Dataset configuration containing file path
        
    Returns:
        Signal data array or None if file not found
    """
    t_start = time.time()
    print(f"Loading signal data for {config.name}...")
    
    if os.path.exists(config.signal_path):
        signal_data = np.load(config.signal_path)
        print(f"Signal loaded successfully. Total points: {len(signal_data)}")
        print(f"--> Data loading finished. Time elapsed: {time.time() - t_start:.2f} seconds")
        return signal_data
    else:
        print(f"Error: Signal file not found at '{config.signal_path}'")
        return None

def process_spectrogram_for_template(f: np.ndarray, Zxx: np.ndarray, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Process spectrogram data and create binary projection for template extraction.
    
    Args:
        f: Frequency array from STFT
        Zxx: STFT matrix
        config: Dataset configuration containing frequency limits and threshold
        
    Returns:
        Tuple of (vertical_projection, binary_projection)
    """
    print("Processing spectrogram and creating projection...")
    t_start = time.time()
    freq_slice = np.where(f <= config.stft_f_max_hz)[0]
    Zxx_sliced = Zxx[freq_slice, :]
    vertical_projection = np.sum(np.abs(Zxx_sliced), axis=0)
    
    if config.binary_threshold is not None:
        binary_projection = (vertical_projection > config.binary_threshold).astype(int)
    else:
        # Use median as default threshold
        threshold = np.median(vertical_projection)
        binary_projection = (vertical_projection > threshold).astype(int)
    
    print(f"--> Spectrogram processing finished. Time elapsed: {time.time() - t_start:.2f} seconds")
    return vertical_projection, binary_projection

def extract_best_template(
    binary_projection: np.ndarray, 
    t: np.ndarray, 
    signal_data: np.ndarray, 
    config: DatasetConfig
) -> Optional[np.ndarray]:
    """Extract the most representative template from high-energy segments.
    
    Args:
        binary_projection: Binary projection indicating high-energy regions
        t: Time array from STFT
        signal_data: Original signal data
        config: Dataset configuration
        
    Returns:
        Best template or None if extraction fails
    """
    print("Extracting best matching template...")
    t_start = time.time()

    # Boundary identification
    diffs = np.diff(binary_projection, prepend=0, append=0)
    starts_t_idx, ends_t_idx = np.where(diffs == 1)[0], np.where(diffs == -1)[0]

    if len(ends_t_idx) > 0 and len(starts_t_idx) > 0 and ends_t_idx[0] < starts_t_idx[0]:
        ends_t_idx = ends_t_idx[1:]
    if len(starts_t_idx) > len(ends_t_idx):
        starts_t_idx = starts_t_idx[:len(ends_t_idx)]

    num_templates = len(starts_t_idx)
    if num_templates == 0:
        print("No complete template pairs found.")
        return None

    # Extract raw template segments
    raw_templates, lengths = [], []
    for i in range(num_templates):
        start_time_s, end_time_s = t[starts_t_idx[i]], t[ends_t_idx[i] - 1]
        start_signal_idx = int(start_time_s * config.fs)
        end_signal_idx = int(end_time_s * config.fs)
        template_signal = signal_data[start_signal_idx:end_signal_idx]
        if len(template_signal) > 0:
            raw_templates.append(template_signal)
            lengths.append(len(template_signal))

    if not raw_templates:
        print("All extracted templates were empty.")
        return None

    # Filter out abnormal length segments
    lengths = np.array(lengths)
    median_len = np.median(lengths)
    lower_bound, upper_bound = 0.8 * median_len, 1.2 * median_len
    filtered_indices = [i for i, L in enumerate(lengths) if lower_bound <= L <= upper_bound]
    filtered_templates = [raw_templates[i] for i in filtered_indices]

    if len(filtered_templates) < 2:
        print("Not enough valid-length templates for similarity comparison.")
        return None

    print(f"→ {len(filtered_templates)} templates retained after length filtering.")

    # Uniform length for similarity comparison
    min_len = min(len(tpl) for tpl in filtered_templates)
    trimmed_templates = [tpl[:min_len] for tpl in filtered_templates]
    N = len(trimmed_templates)

    sim_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            sim = 1 - cosine(trimmed_templates[i], trimmed_templates[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    avg_sim = np.mean(sim_matrix, axis=1)

    best_filtered_index = int(np.argmax(avg_sim))
    best_template = trimmed_templates[best_filtered_index]

    print(f"✓ Selected template (avg similarity = {avg_sim[best_filtered_index]:.4f})")
    print(f"✓ Template length: {len(best_template)} points")
    print(f"✓ Template extraction completed (time: {time.time() - t_start:.2f}s)")

    return best_template

def calculate_match_correlation(
    signal_data: np.ndarray, 
    template: np.ndarray, 
    config: DatasetConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate template matching correlation score using FFT.
    
    Args:
        signal_data: Original signal data
        template: Template to match against
        config: Dataset configuration
        
    Returns:
        Tuple of (time_array, normalized_correlation)
    """
    print("Calculating sliding window matching score (correlation)...")
    t_start = time.time()
    
    # Normalize template and signal for better correlation stability
    template_norm = (template - np.mean(template)) / np.std(template)
    signal_norm = (signal_data - np.mean(signal_data)) / np.std(signal_data)
    
    correlation = signal.correlate(signal_norm, template_norm, mode='valid', method='fft')
    
    # Normalize to -1 to 1 range while preserving correlation sign
    correlation_abs_max = np.max(np.abs(correlation))
    if correlation_abs_max > 0:
        correlation_normalized = correlation / correlation_abs_max
    else:
        correlation_normalized = np.zeros_like(correlation)
    
    # Generate corresponding time axis, note correlation starts at 0
    t_corr = np.arange(len(correlation)) / config.fs
    print(f"--> Match correlation finished. Time elapsed: {time.time() - t_start:.2f} seconds")
    print(f"--> Correlation range: [{np.min(correlation):.6f}, {np.max(correlation):.6f}] -> [-1, 1]")
    return t_corr, correlation_normalized

def analyze_dataset_template_matching(config: DatasetConfig) -> Optional[AnalysisResult]:
    """Complete template matching analysis pipeline.
    
    Args:
        config: Dataset configuration
        
    Returns:
        AnalysisResult containing all analysis outputs
    """
    # print(f"\n{'='*25} Analyzing dataset: {config.name} {'='*25}")
    
    # 1. Data loading
    signal_data = load_signal_data(config)
    if signal_data is None:
        return None

    # 2. STFT calculation
    f_stft, t_stft, Zxx = calculate_stft(signal_data, config)
    
    # 3. Spectrogram processing and projection
    vertical_projection, binary_projection = process_spectrogram_for_template(f_stft, Zxx, config)
    
    # 4. Template extraction
    template = extract_best_template(binary_projection, t_stft, signal_data, config)
    
    # 5. Match score calculation
    match_score, t_corr = None, None
    if template is not None and len(template) > 0:
        t_corr, match_score = calculate_match_correlation(signal_data, template, config)
    
    return AnalysisResult(
        signal_data=signal_data,
        template=template,
        match_score=match_score,
        t_corr=t_corr,
        total_samples=len(signal_data)
    )
