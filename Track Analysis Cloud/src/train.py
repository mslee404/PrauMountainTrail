import argparse
import os
import joblib
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import mlflow
import mlflow.sklearn

# Import your TrackAnalyzer class
from track_analyzer import TrackAnalyzer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, help="Path to input data")
    parser.add_argument("--output-path", type=str, help="Path to output model")
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--enable-tuning", type=bool, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Start MLflow run
    mlflow.start_run()
    
    # Initialize analyzer
    analyzer = TrackAnalyzer()
    
    # Load data
    data_path = args.data_path
    analyzer.load_data(
        f"{data_path}/track_with_elevation.csv",
        f"{data_path}/device_track_1.csv",
        f"{data_path}/emergency_events.csv"
    )
    
    # Generate segments and features
    analyzer.generate_segments()
    analyzer.map_data_to_segments()
    analyzer.engineer_features()
    
    # Training with or without tuning
    if args.enable_tuning:
        print("Running hyperparameter tuning...")
        param_grid = {
            'n_clusters': [3, 4, 5],
            'max_iter': [100, 300],
            'n_init': [10, 20],
            'algorithm': ['lloyd'],
            'scaler': ['standard', 'robust']
        }
        df_results, best_params = analyzer.tune_hyperparameters(param_grid=param_grid)
        analyzer.apply_best_params(best_params)
        
        # Log tuning results
        mlflow.log_params(best_params)
    else:
        print("Training with default parameters...")
        analyzer.train_clustering_model(
            n_clusters=args.n_clusters,
            max_iter=args.max_iter
        )
        mlflow.log_param("n_clusters", args.n_clusters)
        mlflow.log_param("max_iter", args.max_iter)
    
    # Evaluate
    metrics = analyzer.evaluate_clustering()
    
    # Log metrics to MLflow
    mlflow.log_metric("calinski_harabasz", metrics['calinski_harabasz'])
    mlflow.log_metric("davies_bouldin", metrics['davies_bouldin'])
    mlflow.log_metric("silhouette", metrics['silhouette'])
    
    # Save outputs
    os.makedirs(args.output_path, exist_ok=True)
    
    # Save model and scaler
    model_path = f"{args.output_path}/model.joblib"
    scaler_path = f"{args.output_path}/scaler.joblib"
    
    joblib.dump(analyzer.kmeans, model_path)
    joblib.dump(analyzer.scaler, scaler_path)
    
    # Save segmented data
    analyzer.df_seg.to_csv(f"{args.output_path}/df_seg.csv", index=False)
    
    # Log model to MLflow
    mlflow.sklearn.log_model(analyzer.kmeans, "model")
    
    mlflow.end_run()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()