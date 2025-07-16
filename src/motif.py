import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import signal
from dataclasses import dataclass
from typing import List, Optional, Tuple
from motiflets.plotting import *
from tslearn.clustering import TimeSeriesKMeans
import time
from scipy.spatial.distance import cosine

from .proj import DatasetConfig, calculate_stft

@dataclass
class MotifAnalysisResult:
    """Motif analysis result data class"""
    signal_data: np.ndarray
    best_template: Optional[np.ndarray]
    match_score: Optional[np.ndarray]
    t_corr: Optional[np.ndarray]
    total_samples: int

def load_data(config: DatasetConfig):
    """Load signal and trigger data from files"""
    signal_data = np.load(config.signal_path)
    print(f"Loaded signal, shape: {signal_data.shape}")
    if os.path.exists(config.trigger_path):
        trigger_data = np.load(config.trigger_path)
        print(f"Loaded trigger signal, shape: {trigger_data.shape}")
    else:
        print(f"Warning: Trigger file not found at '{config.trigger_path}', using zeros.")
        trigger_data = np.zeros_like(signal_data)
    return signal_data, trigger_data

def apply_threshold(vertical_projection: np.ndarray, threshold_value: float):
    """Apply threshold to vertical projection"""
    vertical_projection_thresholded = vertical_projection.copy()
    vertical_projection_thresholded[vertical_projection_thresholded < threshold_value] = 0
    return vertical_projection_thresholded

def downsample_projection(vertical_projection_thresholded, t, config: DatasetConfig):
    """Downsample projection for motif analysis"""
    stft_step = config.nperseg - config.noverlap
    stft_step_duration = stft_step / config.fs
    target_time_s = config.target_length_points / config.fs
    target_stft_windows = int(target_time_s / stft_step_duration)
    downsample_step = max(1, len(vertical_projection_thresholded) // (target_stft_windows * config.downsample_factor))
    vertical_projection_ds = vertical_projection_thresholded[::downsample_step]
    t_ds = t[::downsample_step]
    motif_length_ds = int(target_stft_windows / downsample_step)
    if motif_length_ds < 2:
        motif_length_ds = 2
    return vertical_projection_ds, t_ds, motif_length_ds, downsample_step

def normalize_projection(vertical_projection_ds):
    """Normalize projection data"""
    scaler = StandardScaler()
    proj_norm = scaler.fit_transform(vertical_projection_ds.reshape(-1, 1)).flatten()
    proj_norm = proj_norm.reshape(1, -1)
    return proj_norm

def motiflets_analysis(proj_norm, motif_length_ds, target_count):
    """Perform motiflets analysis"""
    ml_instance = Motiflets('proj', proj_norm)
    dists, candidates, elbow_points = ml_instance.fit_k_elbow(target_count+1, motif_length_ds)
    return dists, candidates, elbow_points

def extract_motif_positions(candidates, dists, target_k, downsample_step, motif_length_ds, t):
    """Extract motif positions from candidates"""
    if target_k >= 2 and target_k <= len(candidates):
        target_motif_positions_ds = candidates[target_k]
        target_motif_extent = dists[target_k]
        if target_motif_positions_ds is not None and len(target_motif_positions_ds) > 0 and -1 not in target_motif_positions_ds:
            final_motif_positions_original_time = []
            for ds_start_idx in target_motif_positions_ds:
                original_start_stft_idx = ds_start_idx * downsample_step
                original_end_stft_idx = original_start_stft_idx + motif_length_ds * downsample_step
                original_end_stft_idx = min(original_end_stft_idx, len(t))
                final_motif_positions_original_time.append((original_start_stft_idx, original_end_stft_idx))
            return target_motif_positions_ds, final_motif_positions_original_time, target_motif_extent
        else:
            return np.array([]), [], None
    else:
        return np.array([]), [], None

def calculate_match_correlation(signal_data: np.ndarray, template: np.ndarray, config: DatasetConfig):
    """Calculate template matching correlation score using FFT"""
    print("Calculating sliding window matching score (correlation)...")
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
    
    t_corr = np.arange(len(correlation)) / config.fs
    print(f"--> Correlation range: [{np.min(correlation):.6f}, {np.max(correlation):.6f}] -> [-1, 1]")
    return t_corr, correlation_normalized

def analyze_dataset_motif(config: DatasetConfig) -> Optional[MotifAnalysisResult]:
    """Complete motif analysis pipeline"""
    print(f"Analyzing dataset: {config.name}")
    
    # Load data
    signal_data, trigger_data = load_data(config)
    
    # STFT analysis
    f, t, Zxx = calculate_stft(signal_data, config)
    freq_slice = np.where(f <= config.stft_f_max_hz)[0]
    f_sliced = f[freq_slice]
    Zxx_sliced = Zxx[freq_slice, :]
    vertical_projection = np.sum(np.abs(Zxx_sliced), axis=0)
    
    # Apply threshold
    threshold_value = config.threshold
    print(f"Applying threshold: {threshold_value}")
    vertical_projection_thresholded = apply_threshold(vertical_projection, threshold_value)
    print(f"Non-zero elements after threshold: {np.count_nonzero(vertical_projection_thresholded)} / {len(vertical_projection_thresholded)}")
    
    # Downsample
    vertical_projection_ds, t_ds, motif_length_ds, downsample_step = downsample_projection(vertical_projection_thresholded, t, config)
    print(f"Projection length: {len(vertical_projection_ds)}, motif downsampled length: {motif_length_ds}")
    print(f"Downsample factor: {config.downsample_factor}, downsample step: {downsample_step}")
    
    # Normalize
    proj_norm = normalize_projection(vertical_projection_ds)

    # Motiflets analysis
    dists, candidates, elbow_points = motiflets_analysis(proj_norm, motif_length_ds, config.target_count)

    # print("\n--- Motiflets calculation results ---")
    # print(f"Motiflets distances (dists): {dists}")
    # print(f"Motiflets candidates: {candidates}")
    # print(f"Elbow points: {elbow_points}")
    
    # Extract motif positions
    target_k = config.target_count
    target_motif_positions_ds, final_motif_positions_original_time, target_motif_extent = extract_motif_positions(
        candidates, dists, target_k, downsample_step, motif_length_ds, t)
    
    if len(final_motif_positions_original_time) > 0:
        print(f"\nGet Target K ({target_k}) Motiflet ---")
        # print(f"Motiflet positions (downsampled): {target_motif_positions_ds}")
        # print(f"Motiflet extent: {target_motif_extent:.4f}")
        # print(f"Identified {target_k} motifs original STFT time index ranges: {final_motif_positions_original_time}")
    else:
        print(f"\nNo valid Motiflet found for K={target_k}.")

    # DTW-KMeans clustering on projection curve motif segments
    best_template = None
    t_corr = None
    match_score = None
    
    if len(final_motif_positions_original_time) > 0:
        motif_segments = []
        segment_len = 128  # Downsampled length
        for (start_idx, end_idx) in final_motif_positions_original_time:
            seg = vertical_projection[start_idx:end_idx]
            if len(seg) < 2:
                motif_segments.append(np.zeros(segment_len))
            else:
                seg_ds = np.interp(np.linspace(0, len(seg)-1, segment_len), np.arange(len(seg)), seg)
                motif_segments.append(seg_ds)
        motif_segments = np.array(motif_segments)
        if len(motif_segments) >= 2:
            model = TimeSeriesKMeans(n_clusters=2, metric="euclidean", random_state=0, n_init=5, max_iter=10)
            motif_labels = model.fit_predict(motif_segments)

            # Extract template from major class motifs
            if motif_labels is not None and len(motif_labels) == len(final_motif_positions_original_time):
                counts = np.bincount(motif_labels)
                major_label = np.argmax(counts)
                stft_step = config.nperseg - config.noverlap
                raw_segments = []
                for i, (start_stft_idx, end_stft_idx) in enumerate(final_motif_positions_original_time):
                    if motif_labels[i] == major_label:
                        start_sample = int(start_stft_idx * stft_step)
                        end_sample = int(end_stft_idx * stft_step)
                        end_sample = min(end_sample, len(signal_data))
                        seg = signal_data[start_sample:end_sample]
                        if len(seg) > 0:
                            raw_segments.append(seg)
                
                if len(raw_segments) > 0:
                    # Calculate similarity to select best template
                    print(f"Extracted {len(raw_segments)} segments from major class, calculating similarity to select best template...")
                    
                    # Filter abnormal length segments
                    lengths = np.array([len(seg) for seg in raw_segments])
                    median_len = np.median(lengths)
                    lower_bound, upper_bound = 0.8 * median_len, 1.2 * median_len
                    filtered_indices = [i for i, L in enumerate(lengths) if lower_bound <= L <= upper_bound]
                    filtered_segments = [raw_segments[i] for i in filtered_indices]
                    
                    if len(filtered_segments) >= 2:
                        print(f"→ {len(filtered_segments)} segments retained after length filtering.")
                        
                        # Uniform length for similarity comparison
                        min_len = min(len(seg) for seg in filtered_segments)
                        trimmed_segments = [seg[:min_len] for seg in filtered_segments]
                        N = len(trimmed_segments)
                        
                        # Calculate similarity matrix
                        sim_matrix = np.zeros((N, N))
                        for i in range(N):
                            for j in range(i + 1, N):
                                sim = 1 - cosine(trimmed_segments[i], trimmed_segments[j])
                                sim_matrix[i, j] = sim
                                sim_matrix[j, i] = sim
                        avg_sim = np.mean(sim_matrix, axis=1)
                        
                        best_filtered_index = int(np.argmax(avg_sim))
                        best_index = filtered_indices[best_filtered_index]
                        best_template = trimmed_segments[best_filtered_index]
                        
                        print(f"✓ Selected template index: {best_index} (avg similarity = {avg_sim[best_filtered_index]:.4f})")
                        print(f"✓ Template length: {len(best_template)} points")
                        
                        # Use center portion if specified
                        if hasattr(config, 'use_center') and config.use_center is not None:
                            center_ratio = config.use_center
                            center_start = len(best_template) // 2 - int(len(best_template) * center_ratio / 2)
                            center_end = len(best_template) // 2 + int(len(best_template) * center_ratio / 2)
                            best_template = best_template[center_start:center_end]
                            print(f"Using template center {center_ratio*100:.0f}% portion: length {len(best_template)} points")
                        else:
                            print(f"Using full template: length {len(best_template)} points")
                        
                        # Calculate sliding match score (using major class template)
                        t_corr, match_score = calculate_match_correlation(signal_data, best_template, config)

    # print("\n=== K-Motiflets analysis results ===")
    # print(f"Target motif count: {config.target_count}")
    # print(f"Target length: {config.target_length_points} points ({config.target_length_points/config.fs*1000:.1f} ms)")
    # print(f"Downsample factor: {config.downsample_factor}")
    # print(f"Downsample step: {downsample_step}")
    if target_motif_positions_ds is not None and len(target_motif_positions_ds) > 0:
        print(f"Actually identified motif count (K={config.target_count}): {len(target_motif_positions_ds)}")
        print(f"Motif length (STFT windows): {motif_length_ds}")
        stft_step = config.nperseg - config.noverlap
        stft_step_duration = stft_step / config.fs
        print(f"Motif length (time): {motif_length_ds * stft_step_duration * 1000:.1f} ms")
    else:
        print("No motif identified for target K value")

    return MotifAnalysisResult(
        signal_data=signal_data,
        best_template=best_template,
        match_score=match_score,
        t_corr=t_corr,
        total_samples=len(signal_data)
    )
