"""
Azure ML Training Script with Blob Storage Integration
"""
import argparse
import os
import joblib
import mlflow
import mlflow.sklearn
from azure.storage.blob import BlobServiceClient
from datetime import datetime

from track_analyzer import TrackAnalyzer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blob-connection-string", type=str, required=True)
    parser.add_argument("--data-container", type=str, default="raw-data")
    parser.add_argument("--model-container", type=str, default="ml-models")
    parser.add_argument("--date-folder", type=str, required=True,
                       help="Date folder in format YYYY-MM-DD")
    parser.add_argument("--output-path", type=str, default="./outputs")
    parser.add_argument("--enable-tuning", type=str, default="false")
    return parser.parse_args()


def main():
    args = parse_args()
    enable_tuning = args.enable_tuning.lower() == "true"
    
    print("="*70)
    print("AZURE ML TRAINING - TRACK CLUSTERING MODEL")
    print("="*70)
    print(f"Date folder: {args.date_folder}")
    print(f"Enable tuning: {enable_tuning}")
    print("="*70)
    
    mlflow.start_run()
    mlflow.log_param("date_folder", args.date_folder)
    mlflow.log_param("enable_tuning", enable_tuning)
    
    try:
        # Download data from blob
        print("\n[1/7] Downloading data from Blob Storage...")
        local_data_path = "./data"
        download_data_from_blob(
            args.blob_connection_string,
            args.data_container,
            args.date_folder,
            local_data_path
        )
        
        # Initialize analyzer
        analyzer = TrackAnalyzer()
        
        # Load data
        print("\n[2/7] Loading data...")
        analyzer.load_data(
            f"{local_data_path}/track_with_elevation.csv",
            f"{local_data_path}/device_track_1.csv",
            f"{local_data_path}/emergency_events.csv"
        )
        
        # Generate segments
        print("\n[3/7] Generating segments...")
        analyzer.generate_segments()
        
        # Map data
        print("\n[4/7] Mapping data to segments...")
        analyzer.map_data_to_segments()
        
        # Engineer features
        print("\n[5/7] Engineering features...")
        analyzer.engineer_features()
        
        # Training
        print("\n[6/7] Training model...")
        if enable_tuning:
            param_grid = {
                'n_clusters': [3, 4, 5],
                'max_iter': [100, 300],
                'n_init': [10, 20],
                'algorithm': ['lloyd'],
                'scaler': ['standard', 'robust']
            }
            df_results, best_params = analyzer.tune_hyperparameters(param_grid=param_grid)
            analyzer.apply_best_params(best_params)
            
            for key, value in best_params.items():
                if key == 'features':
                    mlflow.log_param(key, ','.join(value))
                else:
                    mlflow.log_param(key, value)
        else:
            analyzer.train_clustering_model(n_clusters=3, max_iter=300)
        
        # Evaluate
        metrics = analyzer.evaluate_clustering()
        mlflow.log_metric("calinski_harabasz", metrics['calinski_harabasz'])
        mlflow.log_metric("davies_bouldin", metrics['davies_bouldin'])
        mlflow.log_metric("silhouette", metrics['silhouette'])
        
        # Save outputs
        print("\n[7/7] Saving outputs...")
        os.makedirs(args.output_path, exist_ok=True)
        
        model_path = f"{args.output_path}/model.joblib"
        scaler_path = f"{args.output_path}/scaler.joblib"
        features_path = f"{args.output_path}/features.joblib"
        
        joblib.dump(analyzer.kmeans, model_path)
        joblib.dump(analyzer.scaler, scaler_path)
        
        feature_info = {
            'feature_cols': analyzer.feature_cols,
            'best_params': analyzer.best_params
        }
        joblib.dump(feature_info, features_path)
        
        # Upload to blob storage
        upload_model_to_blob(
            args.blob_connection_string,
            args.model_container,
            model_path,
            scaler_path,
            features_path
        )
        
        # Log to MLflow
        mlflow.sklearn.log_model(
            analyzer.kmeans,
            "model",
            registered_model_name="track-clustering-model"
        )
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Silhouette Score: {metrics['silhouette']:.4f}")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        mlflow.log_param("status", "failed")
        raise
    finally:
        mlflow.end_run()


if __name__ == "__main__":
    main()