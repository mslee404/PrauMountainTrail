import json
import os
import joblib
import pandas as pd
import pyodbc

# Global objects
model = None
selected_features = None
sql_conn = None


# =========================
# INIT
# =========================
def init():
    global model, selected_features, sql_conn

    # Load model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl")
    model = joblib.load(model_path)

    # Load selected features
    features_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "selected_features.txt")
    with open(features_path, "r") as f:
        selected_features = [line.strip() for line in f.readlines()]

    # SQL Connection (gunakan env var di endpoint)
    sql_conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={os.environ['SQL_SERVER']};"
        f"DATABASE={os.environ['SQL_DATABASE']};"
        f"UID={os.environ['SQL_USERNAME']};"
        f"PWD={os.environ['SQL_PASSWORD']};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
        f"Connection Timeout=30;"
    )

    print("✅ Model & features loaded")
    print("📌 Selected features:", selected_features)


# =========================
# FEATURE LOOKUP
# =========================
def fetch_features_from_sql(latitude, longitude):
    """
    Ambil fitur historis / agregat berdasarkan koordinat
    """
    query = f"""
    SELECT TOP 1
        curvature,
        slope,
        density,
        offtrack_rate,
        stuck_rate,
        sos_rate
    FROM track_cluster
    WHERE lat = ? AND lon = ?
    """

    df = pd.read_sql(query, sql_conn, params=[latitude, longitude])

    if df.empty:
        raise ValueError("No historical features found for given coordinates")

    return df


# =========================
# RUN (INFERENCE)
# =========================
def run(raw_data):
    try:
        data = json.loads(raw_data)

        latitude = data.get("latitude")
        longitude = data.get("longitude")

        if latitude is None or longitude is None:
            return {
                "error": "latitude and longitude are required"
            }

        # 1️⃣ Lookup historical features
        feature_df = fetch_features_from_sql(latitude, longitude)

        # 2️⃣ Select features dynamically (sesuai training)
        X = feature_df[selected_features]

        # 3️⃣ Inference
        cluster_label = model.predict(X)[0]

        return {
            "latitude": latitude,
            "longitude": longitude,
            "cluster": int(cluster_label),
            "used_features": selected_features
        }

    except Exception as e:
        return {
            "error": str(e)
        }
