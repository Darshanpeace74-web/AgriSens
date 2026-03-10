"""Upload all files to HF Space"""
from huggingface_hub import HfApi
import os

TOKEN = os.environ.get("HF_TOKEN", "")
if not TOKEN:
    TOKEN = input("Paste your HF token: ").strip()
if not TOKEN:
    print("No token provided. Set HF_TOKEN env var or paste it when prompted.")
    import sys; sys.exit(1)
SPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf-space")
SPACE_ID = "Darshan881828/AgriGuard-AI"

api = HfApi(token=TOKEN)

files = [
    "README.md",
    "Dockerfile",
    "requirements.txt",
    "main.py",
    "Diseases.png",
    "training_hist.json",
    "trained_plant_disease_model.keras",
]

for f in files:
    filepath = os.path.join(SPACE_DIR, f)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Uploading {f} ({size_mb:.1f} MB)...", end=" ", flush=True)
    api.upload_file(
        path_or_fileobj=filepath,
        path_in_repo=f,
        repo_id=SPACE_ID,
        repo_type="space",
        token=TOKEN,
    )
    print("OK")

print("\nDONE! Visit: https://huggingface.co/spaces/Darshan881828/AgriGuard-AI")
