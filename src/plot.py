import numpy as np
import matplotlib.pyplot as plt
import os
import time
from typing import List, Optional, Union, TYPE_CHECKING
from .proj import DatasetConfig

if TYPE_CHECKING:
    from .motif import MotifAnalysisResult

from .self_temp import AnalysisResult

def plot_matching_score(config: DatasetConfig, result: Union[AnalysisResult, 'MotifAnalysisResult'], result_dir: str = "./result_stage1"):
    """Plot template matching score."""
    print("Creating matching score plot...")
    t_start = time.time()
    os.makedirs(result_dir, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(12, 4))
    
    # Plot matching score
    if result.match_score is not None and result.t_corr is not None:
        # Calculate matching score corresponding sample indices
        corr_sample_indices = (result.t_corr * config.fs).astype(int)
        plt.plot(corr_sample_indices, result.match_score, color='purple', linewidth=0.9)
        plt.ylabel('Correlation Score')
        plt.title('Template Matching Score', fontsize=13)
    else:
        plt.text(0.5, 0.5, 'No matching score available', 
                horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.title('Template Matching Score (No Data)', fontsize=13)
    
    plt.xlabel('Sample Index')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, result.total_samples)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(result_dir, f'{config.name}_matching_score.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"--> Matching score plot saved to '{output_path}'")
    print(f"--> Plot completed. Plotting time: {time.time() - t_start:.2f}s")
    
    # Save template if available and result has best_template attribute
    if hasattr(result, 'best_template') and getattr(result, 'best_template', None) is not None:
        template_suffix = f"_center{int(config.use_center*100)}" if hasattr(config, 'use_center') and config.use_center is not None else ""
        template_path = os.path.join(result_dir, f"{config.name}_template{template_suffix}.npy")
        np.save(template_path, getattr(result, 'best_template'))
        print(f"Template saved to: '{template_path}'")

def _create_boundaries(arr_centers: np.ndarray) -> np.ndarray:
    """Create boundary array for pcolormesh from center points.
    
    Args:
        arr_centers: Array of center points
        
    Returns:
        Array of boundary points
    """
    if arr_centers is None or arr_centers.size == 0: 
        return np.array([])
    diffs = np.diff(arr_centers) / 2.0
    return np.concatenate(([arr_centers[0] - diffs[0]], arr_centers[:-1] + diffs, [arr_centers[-1] + diffs[-1]]))

def plot_segmentation_results(config: DatasetConfig, signal_data: np.ndarray, t_stft: np.ndarray, 
                            f_sliced: np.ndarray, Zxx_mag_db: np.ndarray, vertical_projection: np.ndarray, 
                            change_points: List[int], trigger_data: Optional[np.ndarray] = None, 
                            t_trigger: Optional[np.ndarray] = None, result_dir: str = "./result/sc1"):
    """Plot segmentation results including raw signal, spectrogram, and projection with change points.
    
    Args:
        config: Dataset configuration
        signal_data: Raw signal data
        t_stft: Time array from STFT
        f_sliced: Frequency array (sliced)
        Zxx_mag_db: Magnitude spectrogram in dB
        vertical_projection: Vertical projection of spectrogram
        change_points: Detected change points
        trigger_data: Optional trigger signal data
        t_trigger: Optional trigger time array
    """
    print("Plotting segmentation results...")

    os.makedirs(result_dir, exist_ok=True)
    
    # Fixed 3 subplots
    n_subplots = 3
    fig, axes = plt.subplots(n_subplots, 1, figsize=(18, 6), sharex=False,
                            gridspec_kw={'height_ratios': [1, 1, 1]}, constrained_layout=True)
    ax_raw, ax_spec, ax_proj_seg = axes

    # 1. Raw signal and trigger signal
    step = max(1, len(signal_data) // 150000)
    idx_raw = np.arange(0, len(signal_data), step)
    ax_raw.plot(idx_raw, signal_data[::step], color='blue', lw=0.7, label='Side-Channel Signal')
    
    # Add trigger signal to raw signal plot
    if trigger_data is not None and t_trigger is not None:
        # Convert trigger signal time axis to sample point index
        idx_trigger = np.arange(len(trigger_data))
        # If trigger signal length differs from raw signal, adjust display range
        if len(trigger_data) <= len(signal_data):
            # Normalize trigger signal amplitude for display in same plot
            trigger_normalized = trigger_data / np.max(np.abs(trigger_data)) * np.max(np.abs(signal_data[::step])) * 0.5
            ax_raw.plot(idx_trigger, trigger_normalized, color='red', lw=0.8, label='Trigger Signal')
        else:
            # If trigger signal is longer, truncate to corresponding length
            trigger_truncated = trigger_data[:len(signal_data)]
            trigger_normalized = trigger_truncated / np.max(np.abs(trigger_truncated)) * np.max(np.abs(signal_data[::step])) * 0.5
            ax_raw.plot(idx_raw, trigger_normalized[::step], color='red', lw=0.8, label='Trigger Signal')
    
    ax_raw.set_ylabel('Amplitude')
    ax_raw.set_title(f"Side-Channel Trace")
    ax_raw.grid(True, ls='--', alpha=0.5)
    ax_raw.legend()

    # 2. STFT spectrogram
    # Spectrogram x-axis is sample index
    stft_idx = np.round(np.linspace(0, len(signal_data)-1, len(t_stft))).astype(int)
    mesh = ax_spec.pcolormesh(_create_boundaries(stft_idx), _create_boundaries(f_sliced / 1e6),
                             Zxx_mag_db, shading='flat', cmap='viridis',
                             vmin=np.percentile(Zxx_mag_db[np.isfinite(Zxx_mag_db)], 5),
                             vmax=np.percentile(Zxx_mag_db[np.isfinite(Zxx_mag_db)], 99), rasterized=True)
    fig.colorbar(mesh, ax=ax_spec, label='Amplitude').ax.tick_params(labelsize=8)
    ax_spec.set_ylabel('Frequency (MHz)')
    ax_spec.set_title("STFT Spectrogram", fontsize=13)
    ax_spec.grid(True, ls='--', alpha=0.5)

    # 3. Projection curve and segmentation results combined
    ax_proj_seg.plot(stft_idx, vertical_projection, color='g', lw=0.8, label='Spectrogram Projection')
    for cp in change_points:
        if cp < len(stft_idx):
            ax_proj_seg.axvline(stft_idx[cp], color='r', ls='--', lw=1.2, alpha=0.8, 
                               label='Detected Change Points' if cp == change_points[0] else '')
    ax_proj_seg.set_ylabel('Amplitude')
    ax_proj_seg.set_title("Spectrogram Projection & Segmentation Result", fontsize=13)
    ax_proj_seg.grid(True, ls='--', alpha=0.5)
    ax_proj_seg.legend()

    # Unify x-axis range for all subplots
    xlim = [0, len(signal_data)]
    for ax in axes:
        ax.set_xlim(xlim)
    axes[-1].set_xlabel('Sample Index')

    output_path = os.path.join(result_dir, f'{config.name}_segmentation.png')
    plt.savefig(output_path, dpi=150)
    print(f"Segmentation result saved to '{output_path}'")
    plt.close(fig)

def plot_downsampled_raw_and_trigger(config: DatasetConfig, signal_data: np.ndarray, 
                                   raw_down: np.ndarray, change_points: List[int],
                                   trigger_data: Optional[np.ndarray] = None, 
                                   t_trigger: Optional[np.ndarray] = None,
                                   result_dir: str = "./result/sc1"):
    """Plot downsampled raw signal with change points and optional trigger signal.
    
    Args:
        config: Dataset configuration
        signal_data: Original signal data
        raw_down: Downsampled raw signal
        change_points: Change points detected on downsampled signal
        trigger_data: Optional trigger signal data
        t_trigger: Optional trigger time array
    """
    # Calculate downsampling indices
    n_proj = len(raw_down)
    n_raw = len(signal_data)
    idx_down = np.round(np.linspace(0, n_raw-1, n_proj)).astype(int)

    # Create plot
    n_subplots = 2 if trigger_data is not None and t_trigger is not None else 1
    fig, axes = plt.subplots(n_subplots, 1, figsize=(14, 6 if n_subplots==2 else 4), sharex=False)
    
    # Handle single vs multiple axes
    if n_subplots == 1:
        ax_raw = axes if not isinstance(axes, np.ndarray) else axes.item()
    else:
        ax_raw = axes[0]
        
    ax_raw.plot(idx_down, raw_down, color='blue', lw=0.7, label='Downsampled Raw')
    for i, cp in enumerate(change_points):
        if cp < len(idx_down):
            ax_raw.axvline(idx_down[cp], color='r', ls='--', lw=1.2, alpha=0.8, 
                          label='Change Point' if i==0 else '')
    ax_raw.set_ylabel('Amplitude')
    ax_raw.set_title('Downsampled Raw Trace & Change Points')
    ax_raw.legend()
    ax_raw.grid(True, ls='--', alpha=0.5)

    if n_subplots == 2 and trigger_data is not None:
        ax_trigger = axes[1]
        idx_trigger = np.arange(len(trigger_data))
        ax_trigger.plot(idx_trigger, trigger_data, color='g', lw=0.8)
        ax_trigger.set_ylabel('Amplitude')
        ax_trigger.set_title('Trigger Signal')
        ax_trigger.grid(True, ls='--', alpha=0.5)

    if n_subplots == 1:
        ax_raw.set_xlabel('Sample Index')
    else:
        axes[1].set_xlabel('Sample Index')
        
    plt.tight_layout()
    
    
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(f'{result_dir}/{config.name}_raw_downsampled_segmentation.png', dpi=150)
    plt.close(fig)
