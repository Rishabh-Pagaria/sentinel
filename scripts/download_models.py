#!/usr/bin/env python3
"""
Download all required models for Sentinel phishing detection system.
Models are cached in hf_cache/ directory for reproducibility.

Usage:
    python scripts/download_models.py
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations
MODELS = {
    "deberta_phishing": {
        "repo_id": "rishabhpagaria/deberta_phishing",
        "description": "DeBERTa fine-tuned for phishing classification"
    },
    "gemma_base": {
        "repo_id": "google/gemma-2-2b-it",
        "description": "Google Gemma-2-2B instruction-tuned base model"
    },
    "gemma_phishing_lora": {
        "repo_id": "rishabhpagaria/gemma-2-2b-it_phishing",
        "description": "Gemma-2-2B with LoRA adapter for phishing detection (optional)"
    }
}

def download_models(cache_dir="hf_cache"):
    """
    Download all required models to cache directory.
    
    Args:
        cache_dir: Directory to cache models (default: hf_cache)
    """
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("SENTINEL MODEL DOWNLOADER")
    logger.info("=" * 80)
    logger.info(f"Cache directory: {cache_path.absolute()}\n")
    
    # Set HF cache directory
    os.environ["HF_HOME"] = str(cache_path.absolute())
    
    for model_key, model_info in MODELS.items():
        repo_id = model_info["repo_id"]
        description = model_info["description"]
        
        logger.info(f"\n📦 Downloading {model_key}...")
        logger.info(f"   Repository: {repo_id}")
        logger.info(f"   Description: {description}")
        
        try:
            # Download model
            snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_path,
                local_files_only=False,
                resume_download=True,
                force_download=False
            )
            logger.info(f"   ✓ Successfully downloaded {model_key}")
            
        except Exception as e:
            if "gemma_phishing_lora" in model_key:
                # LoRA model is optional
                logger.warning(f"   ⚠️  Failed to download {model_key} (optional): {e}")
            else:
                logger.error(f"   ✗ Failed to download {model_key}: {e}")
                raise
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ MODEL DOWNLOAD COMPLETE!")
    logger.info("=" * 80)
    logger.info("\nModels are cached in: " + str(cache_path.absolute()))
    logger.info("\nTo use these models in app.py, set:")
    logger.info(f"  export HF_HOME={cache_path.absolute()}")
    logger.info("\nOr update app.py model paths to:")
    logger.info("  DEBERTA_MODEL_PATH = 'rishabhpagaria/deberta_phishing'")
    logger.info("  GEMMA_BASE_MODEL = 'google/gemma-2-2b-it'")
    logger.info("  GEMMA_LORA_ADAPTER = 'rishabhpagaria/gemma-2-2b-it_phishing'")
    logger.info("\n")


if __name__ == "__main__":
    import sys
    
    cache_dir = "hf_cache"
    if len(sys.argv) > 1:
        cache_dir = sys.argv[1]
    
    try:
        download_models(cache_dir=cache_dir)
    except Exception as e:
        logger.error(f"\n❌ Download failed: {e}")
        sys.exit(1)
