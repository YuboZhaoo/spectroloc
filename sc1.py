from typing import List
from src.proj import DatasetConfig
from src.cpd import analyze_change_points, analyze_downsampled_raw
from src.plot import plot_segmentation_results, plot_downsampled_raw_and_trigger
import time
def get_dataset_configs() -> List[DatasetConfig]:
    return [
        DatasetConfig(
            name="ecdsa_1",
            signal_path="./dataset/trace_copilot/ecdsa_1.npy",
            trigger_path="./dataset/trace_copilot/ecdsa_1_trig.npy",
            fs=200.0e6,
            stft_f_max_hz=100.0e6,
            nperseg=10000,
            noverlap=int(10000 * 0.25)
        ),
        DatasetConfig(
            name="ecdsa_2",
            signal_path="./dataset/trace_copilot/ecdsa_2.npy",
            trigger_path="./dataset/trace_copilot/ecdsa_2_trig.npy",
            fs=200.0e6,
            stft_f_max_hz=100.0e6,
            nperseg=1000,
            noverlap=int(1000 * 0.25)
        )
    ]

def main():
    configs = get_dataset_configs()
    
    for config in configs:
        print(f"\n{'='*25} Processing dataset: {config.name} {'='*25}")
        
        # Analyze change points using spectrogram projection
        start = time.perf_counter() 
        vertical_projection, change_points, analysis_data = analyze_change_points(config)
        print(f"\n\n   ---------------------Full Analysis time for {config.name}: {time.perf_counter() - start:.4f} seconds-------------------")
        print(f"Change points: {change_points}")



        if vertical_projection is None or change_points is None or analysis_data is None:
            continue
            
        # Plot segmentation results in our paper, this needs some time
        timing = analysis_data['timing']
        start_time = __import__('time').time()
        plot_segmentation_results(
            config, 
            analysis_data['signal_data'], 
            analysis_data['t_stft'], 
            analysis_data['f_sliced'], 
            analysis_data['Zxx_mag_db'], 
            vertical_projection, 
            change_points, 
            analysis_data['trigger_data'], 
            analysis_data['t_trigger'],
            result_dir="./result/sc1"
        )
        timing['plotting'] = __import__('time').time() - start_time
        print(f"Plotting time: {timing['plotting']:.4f} seconds")

        # Analyze same downsampled raw trace change point detection
        # raw_down, raw_change_points = analyze_downsampled_raw(
        #     config, 
        #     analysis_data['signal_data'], 
        #     vertical_projection
        # )

        # print(f"Change points for raw trace: {change_points}")
        
        # plot_downsampled_raw_and_trigger(
        #     config, 
        #     analysis_data['signal_data'], 
        #     raw_down, 
        #     raw_change_points, 
        #     analysis_data['trigger_data'], 
        #     analysis_data['t_trigger']
        # )
        
        # Print timing summary
        # total_time = timing['stft_projection'] + timing['claspy'] + timing['plotting']
        # print(f"Total processing time: {total_time:.4f} seconds")
        # print(f"  - STFT & Projection: {timing['stft_projection']:.4f}s ({timing['stft_projection']/total_time*100:.1f}%)")
        # print(f"  - ClaSPy Segmentation: {timing['claspy']:.4f}s ({timing['claspy']/total_time*100:.1f}%)")
        # print(f"  - Plotting: {timing['plotting']:.4f}s ({timing['plotting']/total_time*100:.1f}%)")

if __name__ == "__main__":
    main()
