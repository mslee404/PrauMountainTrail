from azure.ai.ml import MLClient, dsl, Input, Output
from azure.ai.ml import command
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes

# ==============================================================
# 1. KONFIGURASI (ISI DATA ASLI)
# ==============================================================
API_KEY        = "GANTI_API_KEY_YOUTUBE"
CONN_STR       = "GANTI_CONN_STRING_BLOB"
CONTAINER      = "kel2tcba"
COMPUTE_NAME   = "kel2tcba"
ENDPOINT_NAME  = "ganti-nama-endpoint-anda" # Contoh: indoyt-endpoint

# Environment Anda
ENV_URI        = "azureml:Kel_2_TCBA:3"

# Data Workspace (Cek pojok kanan atas Studio)
SUB_ID         = "c835dc63-ce84-40ce-add2-501209336e4e"
RES_GROUP      = "TCBA_ML_Kel2"
WORKSPACE      = "project-kel2"

# ==============================================================
# 2. DEFINISI KOMPONEN (BLUEPRINTS)
# ==============================================================

# A. Cetakan Job Scraping
scrape_comp = command(
    display_name="1. Scraping YouTube",
    code="./",
    command=f'python scrape_to_blob.py --api_key "{API_KEY}" --conn_str "{CONN_STR}" --container "{CONTAINER}"',
    environment=ENV_URI,
)

# B. Cetakan Job Training
train_comp = command(
    display_name="2. Retraining Model",
    code="./",
    command=f'python retrain.py --conn_str "{CONN_STR}" --container "{CONTAINER}" --model_input ${{{{inputs.model_guru}}}} --model_output ${{{{outputs.model_murid}}}}',
    environment=ENV_URI,
    inputs={"model_guru": Input(type="custom_model")},
    outputs={"model_murid": Output(type="uri_folder", mode="upload")}
)

# C. Cetakan Job Deployment (Install azure-ai-ml dulu biar bisa deploy)
deploy_comp = command(
    display_name="3. Update Endpoint",
    code="./",
    command=f'pip install azure-ai-ml azure-identity && python deploy_auto.py --model_path ${{{{inputs.model_baru}}}} --endpoint_name "{ENDPOINT_NAME}" --subscription_id "{SUB_ID}" --resource_group "{RES_GROUP}" --workspace_name "{WORKSPACE}"',
    environment=ENV_URI,
    inputs={"model_baru": Input(type="uri_folder")}
)

# ==============================================================
# 3. MENYUSUN PIPELINE
# ==============================================================
@dsl.pipeline(
    compute=COMPUTE_NAME,
    description="Full Pipeline: Scrape -> Train -> Deploy"
)
def youtube_full_pipeline(model_input):

    # Langkah 1: Scrape
    step1 = scrape_comp()

    # Langkah 2: Train (Inputnya dari parameter pipeline)
    step2 = train_comp(model_guru=model_input)
    step2.resources = step1.resources # Train nunggu Scrape selesai

    # Langkah 3: Deploy (Inputnya adalah OUTPUT dari Train)
    step3 = deploy_comp(model_baru=step2.outputs.model_murid)

    return {"final_model": step2.outputs.model_murid}

# ==============================================================
# 4. EKSEKUSI
# ==============================================================
def main():
    print("🚀 Menghubungkan ke Azure ML...")
    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, SUB_ID, RES_GROUP, WORKSPACE)

    # Input Awal: Model Versi 1 (Atau Latest)
    # Pastikan nama model 'bert-indo-model' sudah ada. Jika belum, ganti namanya.
    seed_model = Input(type=AssetTypes.CUSTOM_MODEL, path="azureml:bert-indo-model:1")

    print("⚙️  Menyusun Pipeline...")
    pipeline_job = youtube_full_pipeline(model_input=seed_model)

    print("📤 Mengirim Pipeline ke Azure...")
    returned_job = ml_client.jobs.create_or_update(
        pipeline_job,
        experiment_name="Youtube-Full-Auto"
    )

    print(f"✅ Pipeline berhasil dikirim!")
    print(f"👉 Buka link ini untuk melihat progres & membuat Jadwal: {returned_job.studio_url}")

if __name__ == "__main__":
    main()
