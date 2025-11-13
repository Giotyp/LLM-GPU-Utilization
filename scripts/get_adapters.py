"""
Download LoRA adapters for Llama-based models and store them locally.
"""

import os
from huggingface_hub import snapshot_download

# Directory to store adapters
BASE_DIR = "./models/lora-adapters"
os.makedirs(BASE_DIR, exist_ok=True)

# List of adapters to fetch
ADAPTERS = {
    "alpaca-lora-7b": "tloen/alpaca-lora-7b",
    "lazylora-7b-chathf": "wuxianchao/lazylora-7b-chathf",
    "llama2-7b-chat-lora-adaptor": "manojpatil/llama-2-7b-chat-lora-adaptor",
    "llama2-13b-gpt-adaptor": "JadenAI/la-gpt-llama2-13b-lora",
    "phi3-mini-4k-alpaca": "johnlam90/phi3-mini-4k-instruct-alpaca-lora",
    "tinyllama-1.1b-alpaca": "sam2ai/tiny_llama_1b_lora_pt",
    "qwen2.5-3b-alpaca": "TheDenk/Qwen2.5-VL-3B-TrackAnyObject-LoRa-v1",
}

for name, repo_id in ADAPTERS.items():
    target_dir = os.path.join(BASE_DIR, name)
    print(f"Fetching {repo_id} -> {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"Downloaded {name}\n")

print("\nAll adapters downloaded successfully!")
print(f"Stored under: {os.path.abspath(BASE_DIR)}")
