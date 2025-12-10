import argparse
import json
import os
import time
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import BatchDeployment, BatchEndpoint, Model
from azure.ai.ml.constants import BatchDeploymentOutputAction

def main():
    # 1. SETUP ARGUMEN
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Nama model di Registry")
    parser.add_argument("--endpoint_name", type=str, required=True, help="Nama Endpoint yang mau dituju")
    parser.add_argument("--input_metrics", type=str, help="Folder input berisi metrics.json")
    parser.add_argument("--min_score", type=float, default=0.8, help="Batas minimal Silhouette Score")
    parser.add_argument("--compute_name", type=str, default="cpu-cluster", help="Nama Compute Cluster untuk Endpoint")
    args = parser.parse_args()

    print(f"🚀 Memulai Auto-Deployment untuk {args.model_name} ke {args.endpoint_name}")

    # 2. CEK METRICS (GATEKEEPER)
    # Mencari file metrics.json di folder input
    metrics_path = os.path.join(args.input_metrics, "metrics.json")
    
    # Fallback: Jika file ada di subfolder (karena struktur Azure kadang nambah folder ID)
    if not os.path.exists(metrics_path):
        import glob
        found_files = glob.glob(f"{args.input_metrics}/**/metrics.json", recursive=True)
        if found_files:
            metrics_path = found_files[0]
        else:
            print("⚠️ Warning: metrics.json tidak ditemukan")
            current_score = 1.

    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            current_score = metrics.get('silhouette', 0)
            print(f"📊 Detected Score: {current_score} (Threshold: {args.min_score})")

            if current_score < args.min_score:
                print(f"⛔ STOP: Skor model ({current_score}) di bawah standar ({args.min_score}).")
                print("Deployment dibatalkan untuk menjaga kualitas endpoint.")
                return # KELUAR DARI SCRIPT

    # 3. KONEKSI KE AZURE ML (Menggunakan Identitas Compute Cluster)
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)

    # 4. AMBIL MODEL TERBARU
    # Kita ambil model yang baru saja di-register di step sebelumnya
    model_list = list(ml_client.models.list(name=args.model_name))
    latest_model = model_list[0] # List biasanya urut dari terbaru
    print(f"✅ Menggunakan Model Versi: {latest_model.version}")

    # 5. CEK / BUAT ENDPOINT
    try:
        endpoint = ml_client.batch_endpoints.get(args.endpoint_name)
        print(f"Endpoint {args.endpoint_name} sudah ada.")
    except:
        print(f"Endpoint {args.endpoint_name} belum ada. Membuat baru...")
        endpoint = BatchEndpoint(name=args.endpoint_name)
        ml_client.batch_endpoints.begin_create_or_update(endpoint).result()

    # 6. CONFIG DEPLOYMENT BARU
    # Nama deployment unik, misal: "dep-v5-timestamp"
    timestamp = int(time.time())
    deployment_name = f"dep-v{latest_model.version}-{timestamp}"

    deployment = BatchDeployment(
        name=deployment_name,
        endpoint_name=args.endpoint_name,
        model=latest_model,
        compute=args.compute_name, # Compute cluster yang menjalankan prediksi nanti
        instance_count=1,
        max_concurrency_per_instance=2,
        mini_batch_size=10,
        output_action=BatchDeploymentOutputAction.APPEND_ROW,
        output_file_name="predictions.csv"
    )

    # 7. EKSEKUSI DEPLOYMENT (LRO - Long Running Operation)
    print(f"⏳ Sedang mendeploy {deployment_name}... (Bisa memakan waktu 5-10 menit)")
    ml_client.batch_deployments.begin_create_or_update(deployment).result()

    # 8. UPDATE TRAFFIC (Jadikan Default)
    # Arahkan 100% traffic ke deployment baru ini
    endpoint = ml_client.batch_endpoints.get(args.endpoint_name)
    endpoint.defaults.deployment_name = deployment_name
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()

    print(f"🎉 SUKSES! Endpoint {args.endpoint_name} telah diupdate ke model v{latest_model.version}.")

if __name__ == "__main__":
    main()