from huggingface_hub import snapshot_download
import argparse
import os

MODELS_DIR = "./models/"
os.makedirs(MODELS_DIR, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model from Hugging Face Hub")
    parser.add_argument(
        "--model_id", type=str, required=True, help="Model ID on Hugging Face Hub"
    )
    parser.add_argument(
        "--dir_name", type=str, required=True, help="Directory name to save the model"
    )
    args = parser.parse_args()

    local_path = f"{MODELS_DIR}/{args.dir_name}"
    snapshot_download(repo_id=args.model_id, local_dir=local_path)
