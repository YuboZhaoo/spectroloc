import numpy as np
import os
import time
from typing import Tuple, List, Optional
from claspy.segmentation import BinaryClaSPSegmentation
from .proj import DatasetConfig, calculate_stft, process_spectrogram

def load_data(config: DatasetConfig) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load signal and trigger data from files.
    
    Args:
        config: Dataset configuration containing file paths
        
    Returns:
        Tuple of (signal_data, trigger_data, trigger_time)
    """
    print(f"Loading signal data for {config.name}...")
    signal_data = None
    trigger_data = None
    t_trigger = None
    
    if os.path.exists(config.signal_path):
        signal_data = np.load(config.signal_path)
        print(f"Signal loaded successfully. Total points: {len(signal_data)}")
    else:
        print(f"Error: Signal file not found at '{config.signal_path}'")
        
    if hasattr(config, 'trigger_path') and config.trigger_path and os.path.exists(config.trigger_path):
        trigger_data = np.load(config.trigger_path)
        t_trigger = np.arange(len(trigger_data)) / config.fs
        print(f"Trigger loaded successfully. Total points: {len(trigger_data)}")
    else:
        print(f"Warning: Trigger file not found at '{getattr(config, 'trigger_path', None)}'")
        
    return signal_data, trigger_data, t_trigger

def analyze_change_points(config: DatasetConfig) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[dict]]:
    """Analyze change points using spectrogram projection and ClaSPy.
    
    Args:
        config: Dataset configuration
        
    Returns:
        Tuple of (vertical_projection, change_points, analysis_data)
        analysis_data contains: {
            'signal_data': signal_data,
            'trigger_data': trigger_data, 
            't_trigger': t_trigger,
            't_stft': t_stft,
            'f_sliced': f_sliced,
            'Zxx_mag_db': Zxx_mag_db,
            'timing': timing_info
        }
    """
    # Load data
    signal_data, trigger_data, t_trigger = load_data(config)
    if signal_data is None:
        return None, None, None
    
    timing = {}
    
    # 1. STFT and projection timing
    start_time = time.time()
    f_stft, t_stft, Zxx = calculate_stft(signal_data, config)
    f_sliced, Zxx_mag_db, vertical_projection = process_spectrogram(f_stft, Zxx, config)
    vertical_projection = vertical_projection.astype(np.float64)  # Ensure type compatibility with claspy
    timing['stft_projection'] = time.time() - start_time
    # print(f"STFT and projection time: {timing['stft_projection']:.4f} seconds")
    
    # 2. ClaSPy segmentation timing
    start_time = time.time()
    clasp = BinaryClaSPSegmentation()
    # print(f"Vertical projection: {len(vertical_projection)}")
    change_points_raw = clasp.fit_predict(vertical_projection)
    # Convert to list of integers - handle different return formats
    if isinstance(change_points_raw, (list, tuple)):
        change_points = []
        for cp_array in change_points_raw:
            if hasattr(cp_array, 'flatten'):
                change_points.extend([int(cp) for cp in cp_array.flatten()])
            else:
                change_points.append(int(cp_array))
    elif hasattr(change_points_raw, 'flatten'):
        change_points = [int(cp) for cp in change_points_raw.flatten()]
    else:
        change_points = [int(change_points_raw)]
    timing['claspy'] = time.time() - start_time
    # print(f"ClaSPy segmentation time: {timing['claspy']:.4f} seconds")
    # print(f"Change points: {change_points}")
    
    analysis_data = {
        'signal_data': signal_data,
        'trigger_data': trigger_data,
        't_trigger': t_trigger,
        't_stft': t_stft,
        'f_sliced': f_sliced,
        'Zxx_mag_db': Zxx_mag_db,
        'timing': timing
    }
    
    return vertical_projection, change_points, analysis_data

def analyze_downsampled_raw(config: DatasetConfig, signal_data: np.ndarray, vertical_projection: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Analyze change points on downsampled raw signal.
    
    Args:
        config: Dataset configuration
        signal_data: Original signal data
        vertical_projection: Vertical projection for length reference
        
    Returns:
        Tuple of (downsampled_raw_signal, change_points)
    """
    # Downsample raw signal to match projection length
    n_proj = len(vertical_projection)
    n_raw = len(signal_data)
    idx_down = np.round(np.linspace(0, n_raw-1, n_proj)).astype(int)
    raw_down = signal_data[idx_down].astype(np.float64)  # Ensure float64
    print(f"[Downsampled Raw] Length after downsampling: {len(raw_down)}")

    # Change point detection on downsampled raw signal
    clasp = BinaryClaSPSegmentation()
    change_points_raw = clasp.fit_predict(raw_down)
    # Convert to list of integers - handle different return formats
    if isinstance(change_points_raw, (list, tuple)):
        change_points = []
        for cp_array in change_points_raw:
            if hasattr(cp_array, 'flatten'):
                change_points.extend([int(cp) for cp in cp_array.flatten()])
            else:
                change_points.append(int(cp_array))
    elif hasattr(change_points_raw, 'flatten'):
        change_points = [int(cp) for cp in change_points_raw.flatten()]
    else:
        change_points = [int(change_points_raw)]
    print(f"[Downsampled Raw] Detected change points: {change_points}")
    
    return raw_down, change_points
