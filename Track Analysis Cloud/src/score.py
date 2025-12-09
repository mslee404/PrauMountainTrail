import json
import joblib
import numpy as np
import pandas as pd
import os


def init():
    """Initialize model and scaler"""
    global model, scaler, feature_cols
    
    # Get model path
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.joblib")
    scaler_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "scaler.joblib")
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Define feature columns (should match training)
    feature_cols = ['slope', 'curvature', 'offtrack_rate', 
                   'density', 'stuck_rate', 'sos_rate']
    
    print("Model and scaler loaded successfully")


def run(raw_data):
    """
    Make predictions on input data
    
    Input format:
    {
        "data": [
            {
                "slope": 0.05,
                "curvature": 0.2,
                "offtrack_rate": 0.1,
                "density": 50,
                "stuck_rate": 0.02,
                "sos_rate": 0.01
            }
        ]
    }
    """
    try:
        # Parse input
        data = json.loads(raw_data)
        df = pd.DataFrame(data['data'])
        
        # Validate features
        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            return json.dumps({
                "error": f"Missing columns: {missing_cols}"
            })
        
        # Select and order features
        X = df[feature_cols]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Predict
        predictions = model.predict(X_scaled)
        
        # Get cluster centers distance (for confidence)
        distances = model.transform(X_scaled)
        min_distances = np.min(distances, axis=1)
        
        # Prepare response
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "cluster": int(pred),
                "distance_to_center": float(min_distances[i]),
                "input": df.iloc[i].to_dict()
            })
        
        return json.dumps({
            "predictions": results
        })
        
    except Exception as e:
        return json.dumps({
            "error": str(e)
        })