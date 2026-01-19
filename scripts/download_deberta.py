#!/usr/bin/env python3
"""
Download DeBERTa-v3-base model from HuggingFace Hub to local storage.
This script downloads the model files to hf_models/deberta-v3-base/
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    # Define model and local directory
    model_id = "microsoft/deberta-v3-base"
    local_dir = "hf_models/deberta-v3-base"
    
    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    print(f"[download_deberta] Target directory: {local_dir}")
    
    # Check if model already exists
    config_file = os.path.join(local_dir, "config.json")
    model_file = os.path.join(local_dir, "pytorch_model.bin")
    
    if os.path.exists(config_file) and os.path.exists(model_file):
        print(f"[download_deberta] Model already exists in {local_dir}")
        print(f"[download_deberta] Delete the folder to re-download")
        return
    
    print(f"[download_deberta] Downloading {model_id}...")
    print(f"[download_deberta] This may take several minutes (~600MB)")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Don't use symlinks on Windows
            resume_download=True,  # Resume if interrupted
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # Skip unnecessary files
        )
        
        print(f"[download_deberta] ✓ Download complete!")
        print(f"[download_deberta] Model saved to: {os.path.abspath(local_dir)}")
        
        # List downloaded files
        print(f"\n[download_deberta] Downloaded files:")
        for file in sorted(os.listdir(local_dir)):
            file_path = os.path.join(local_dir, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {file} ({size_mb:.2f} MB)")
        
    except Exception as e:
        print(f"[download_deberta] ✗ Error during download: {e}")
        print(f"[download_deberta] You may need to:")
        print(f"  1. Check your internet connection")
        print(f"  2. Login to HuggingFace: huggingface-cli login")
        raise

if __name__ == "__main__":
    main()
