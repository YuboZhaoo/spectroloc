from typing import List
import time
from src.proj import DatasetConfig
from src.motif import analyze_dataset_motif, MotifAnalysisResult
from src.plot import plot_matching_score

def get_dataset_configs() -> List[DatasetConfig]:
    """Get list of dataset configurations for motif analysis"""
    return [
        DatasetConfig(
            name="aes_mixed",
            signal_path="./dataset/semi-loc/mixed_aes.npy",
            trigger_path="./dataset/semi-loc/mixed_aes_trig.npy",
            # STFT setting
            fs=250.0e6,
            stft_f_max_hz=15.0e6,
            nperseg=50000,
            noverlap=int(50000 * 0.95),

            # Motif analysis parameters
            target_length_points=85000,
            target_count=46,
            threshold=600,
        ),
        DatasetConfig(
            name="sha256",
            signal_path='./dataset/semi-loc/mixed_sha256.npy',
            trigger_path='./dataset/semi-loc/mixed_sha256_trig.npy',
            # STFT setting, same as above
            fs=250.0e6,
            stft_f_max_hz=15.0e6,
            nperseg=50000,
            noverlap=int(50000 * 0.95),

            # Motif analysis parameters
            target_length_points=450000,
            target_count=16,
            threshold=500,
            use_center=0.1 # because the target is too large
        ),
    ]

def main():
    """Main function: analyze datasets and plot match scores"""
    configs = get_dataset_configs()
    
    all_results = {}
    for config in configs:
        print(f"\n{'='*25} Processing dataset: {config.name} {'='*25}")
        start = time.perf_counter() 
        analysis_results = analyze_dataset_motif(config)
        elapsed = time.perf_counter() - start
        print(f"\n\n   ---------------------Full Analysis time for {config.name}: {elapsed:.4f} seconds-------------------\n")

        all_results[config.name] = analysis_results
        
    
    for config in configs:
        if all_results[config.name] is not None:
            plot_matching_score(config, all_results[config.name], result_dir="./result/sc3")
        else:
            print(f"Failed to analyze dataset: {config.name}")

if __name__ == "__main__":
    main() 