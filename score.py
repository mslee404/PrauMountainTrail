import os
import joblib
import pandas as pd

model = None

def init():
    global model
    # Path model biasanya ada di folder /var/azureml-app/ atau di environment variable
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "kmeans_model.pkl")
    scaler_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "scaler_model.pkl")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

def run(mini_batch):
    results = []

    for file_path in mini_batch:
        try:
            df = pd.read_csv(file_path)

            

            clusters = model.predict(df)

            df["cluster"] = clusters

            output_file = file_path.replace(".csv", "_scored.csv")
            df.to_csv(output_file, index=False)

            results.append(output_file)
        except Exception as e:
            # Kalau error, catat
            results.append(f"Error processing {file_path}: {str(e)}")

    return results
