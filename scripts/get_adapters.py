"""
Download LoRA adapters for Llama-based models and store them locally.
"""

import os
import argparse
from huggingface_hub import snapshot_download

# Directory to store adapters
LORA_DIR = "./models/lora-adapters"
os.makedirs(LORA_DIR, exist_ok=True)

# List of adapters to fetch
ADAPTERS = {
    "alpaca-lora-7b": "tloen/alpaca-lora-7b",
    "lazylora-7b-chathf": "wuxianchao/lazylora-7b-chathf",
    "llama2-7b-chat-lora-adaptor": "manojpatil/llama-2-7b-chat-lora-adaptor",
    "phi3-mini-4k-alpaca": "johnlam90/phi3-mini-4k-instruct-alpaca-lora",
    "qwen2.5-3b-alpaca": "TheDenk/Qwen2.5-VL-3B-TrackAnyObject-LoRa-v1",
}


def fetch_adapters():
    for name, repo_id in ADAPTERS.items():
        target_dir = os.path.join(LORA_DIR, name)
        print(f"Fetching {repo_id} -> {target_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"Downloaded {name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download lora adapter(s) from Hugging Face Hub"
    )
    parser.add_argument(
        "--lora_id", type=str, required=False, help="Adapter ID on Hugging Face Hub"
    )
    parser.add_argument(
        "--dir_name", type=str, required=False, help="Directory name to save the model"
    )
    args = parser.parse_args()

    if not args.lora_id:
        print("Fetching all defined adapters...")
        fetch_adapters()
    else:
        if not args.dir_name:
            target_dir_name = args.lora_id.split("/")[-1]
        else:
            target_dir_name = args.dir_name
        target_dir = os.path.join(LORA_DIR, target_dir_name)
        snapshot_download(
            repo_id=args.lora_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"Downloaded {target_dir_name}\n")


print("\nAll adapters downloaded successfully!")
print(f"Stored under: {os.path.abspath(LORA_DIR)}")
