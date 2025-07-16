from typing import List
import time
import matplotlib.pyplot as plt
import os
from src.proj import DatasetConfig
from src.self_temp import analyze_dataset_template_matching, AnalysisResult
from src.plot import plot_matching_score



def get_dataset_configs() -> List[DatasetConfig]:
    return [
        DatasetConfig(
            name="software_aes",
            signal_path="./dataset/semi-loc/sw_aes.npy",
            trigger_path="./dataset/semi-loc/sw_aes_trig.npy", # trigger is never used for analysis

            # STFT setting
            fs=120.0e6,
            stft_f_max_hz=50.0e6,
            nperseg=20000,
            noverlap=int(20000 * 0.25), # all for sc2 is 0.25

            # the only threshold parameter we need for sc2
            binary_threshold=800.0,
        ),
        DatasetConfig(
            name="hardware_aes",
            signal_path="./dataset/semi-loc/hw_aes.npy",
            trigger_path="./dataset/semi-loc/hw_aes_trig.npy",
            fs=500.0e6,
            stft_f_max_hz=100.0e6,
            nperseg=10000,
            noverlap=int(10000 * 0.25),
            
            binary_threshold=490.0
        ),
        DatasetConfig(
            name="aes_random_50",
            signal_path="./dataset/em/aes_rand_50.npy",
            trigger_path="./dataset/em/aes_rand_50_triggers.npy",
            fs=100.0e6,
            stft_f_max_hz=100.0e6,
            nperseg=3000,
            noverlap=int(3000 * 0.25),

            binary_threshold=0.02,
        ),
        DatasetConfig(
            name="aes_random_single",
            signal_path="./dataset/trace_copilot/aes_rand_single.npy",
            trigger_path="./dataset/trace_copilot/aes_rand_single_trig.npy",
            fs=300.0e6,
            stft_f_max_hz=80.0e6,
            nperseg=800,
            noverlap=int(800 * 0.25),

            binary_threshold=27000,
        ),
    ]

def main():
    """Main function to process all datasets and generate template matching results."""
    configs = get_dataset_configs()
    
    for config in configs:
        print(f"\n{'='*25} Processing dataset: {config.name} {'='*25}")
        
        # Analyze dataset for template matching
        start = time.perf_counter()
        result = analyze_dataset_template_matching(config)
        elapsed = time.perf_counter() - start
        print(f"\n\n   ---------------------Full Analysis time for {config.name}: {elapsed:.4f} seconds-------------------\n")
        
        # Plot results
        if result is not None:
            plot_matching_score(config, result, result_dir="./result/sc2")
        else:
            print(f"Failed to analyze dataset: {config.name}")

if __name__ == "__main__":
    main() 