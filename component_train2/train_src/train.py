import argparse
import glob
import os
from pathlib import Path
from track_analyzer import TrackAnalyzer  # Import class dari file sebelah
import pandas as pd
import shutil

def parse_args():
     # 1. SETUP ARGUMENT PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Path to input data folder")
    parser.add_argument("--output_data", type=str, help="Path to output data folder")
    return parser.parse_args()

def main_with_tuning():
    args = parse_args()

    # Setup Paths
    input_root = args.input_data
    # Convert string output path ke Path object
    output_root = args.output_data 

    os.makedirs(output_root, exist_ok=True)
    
    print("="*50)
    print("TRACK SEGMENTATION PIPELINE STARTED")
    print(f"Input Path: {input_root}")
    print(f"Output Path: {output_root}")
    print("="*50)
    
    # 2. INISIALISASI ANALYZER
    # Kita oper output_root ke class agar dia tau mau simpan model di mana
    analyzer = TrackAnalyzer(output_dir=output_root)
    
    # 3. DEFINE FILE PATHS
    # Sesuaikan dengan struktur file di Blob Storage kamu
    # Gunakan os.path.join agar aman di Linux/Windows
    elev_path = os.path.join(input_root, 'track_with_elevation.csv')

    root_track_files = glob.glob(os.path.join(args.input_data, "tracks/device_track_*.csv"))
    root_emerg_path = glob.glob(os.path.join(args.input_data, "emergency/emergency_events_*.csv"))

    df_track_list = []
    for filename in root_track_files:
        print(f" - Merging: {os.path.basename(filename)}")
        try:
            df_temp = pd.read_csv(filename)
            df_track_list.append(df_temp)
        except Exception as e:
            print(f"Warning: Gagal baca {filename}: {e}")

    if df_track_list:
        combined_track_df = pd.concat(df_track_list, ignore_index=True)
    else:
        combined_track_df = pd.DataFrame() # Empty fallback

    df_emg_list = []
    for filename in root_emerg_path:
        print(f" - Merging: {os.path.basename(filename)}")
        try:
            df_temp = pd.read_csv(filename)
            df_emg_list.append(df_temp)
        except Exception as e:
            print(f"Warning: Gagal baca {filename}: {e}")

    if df_emg_list:
        combined_emg_df = pd.concat(df_emg_list, ignore_index=True)
    else:
        combined_emg_df = pd.DataFrame() # Empty fallback
    
    # 4. EXECUTE PIPELINE
    try:
        analyzer.load_data(elev_path, combined_track_df, combined_emg_df)
        
        analyzer.generate_segments()
        analyzer.map_data_to_segments()
        analyzer.engineer_features()
        
        # Tuning
        print("\nStarting Hyperparameter Tuning...")
        # Grid kecil untuk demo biar cepat
        
        df_results, best_params = analyzer.tune_hyperparameters()
        
        # Save tuning log
        df_results.to_csv(analyzer.output_dir / "tuning_results.csv", index=False)
        print(f"Tuning log saved.")
        
        # Apply Best Model
        analyzer.apply_best_params(best_params)
        analyzer.evaluate_clustering()
        metrics_dict = analyzer.evaluate_clustering()
        
        # Save Final CSV
        analyzer.save_results()
        analyzer.save_model()
        analyzer.save_metrics(metrics_dict)
        
        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*50)
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {str(e)}")
        raise e

if __name__ == "__main__":
    main_with_tuning()