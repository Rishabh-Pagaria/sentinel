#!/usr/bin/env python3
"""
Launch Vertex AI fine-tuning job     # Configure worker pool specs with cost-effective compute
    worker_pool_specs = {
        "machine_spec": {
            "machine_type": "e2-standard-4"  # More cost-effective machine type with 4 vCPUs
        },
        "replica_count": 1,  # Single replica to minimize costs
        "python_package_spec": {
            "executor_image_uri": "us-docker.pkg.dev/cloud-ml-pipeline-release/training/base-cpu",  # Pre-built CPU training container
            "package_uris": [f"gs://{bucket_name}/dist/mypackage-0.1.tar.gz"],
            "python_module": "train",
            "args": args
        }
    }detection model.
"""

from google.cloud import aiplatform
from google.cloud.aiplatform import Model
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def launch_finetune_job():
    """Launch Vertex AI fine-tuning job"""
    
    # Get configuration from environment
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION")
    bucket_name = os.getenv("GCP_BUCKET_NAME")
    model_name = os.getenv("FINETUNED_MODEL_NAME")
    
    # Set credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GCP_CREDENTIALS_PATH")
    
    # Initialize Vertex AI
    aiplatform.init(
        project=project_id,
        location=location,
        staging_bucket="sentinel-central1"  # Using regional bucket in us-central1
    )
    
    # Create a Python package for distribution
    import subprocess
    subprocess.check_call([
        "python", "-m", "pip", "install", "--upgrade", "build"
    ])
    subprocess.check_call([
        "python", "-m", "build", 
        "--outdir", "dist/",
        "."
    ])
    
    # Upload the package to GCS
    from google.cloud import storage
    client = storage.Client()
    bucket = client.get_bucket("sentinel-central1")
    blob = bucket.blob("dist/sentinel-0.1.tar.gz")
    blob.upload_from_filename("dist/sentinel-0.1.tar.gz")
    
    # Configure arguments
    args = [
        "--train-file", "gs://sentinel-central1/vertex_ai/train.jsonl",
        "--eval-file", "gs://sentinel-central1/vertex_ai/eval.jsonl",
        "--output-dir", "/opt/ml/model",
        "--batch-size", "16",
        "--num-epochs", "3",
        "--learning-rate", "2e-5"
    ]
    
    # Configure worker pool specs
    worker_pool_specs = {
        "machine_spec": {
            "machine_type": "n1-standard-8"
        },
        "replica_count": 1,
        "python_package_spec": {
            "executor_image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest",  # PyTorch 1.13 training container
            "package_uris": ["gs://sentinel-central1/dist/sentinel-0.1.tar.gz"],
            "python_module": "sentinel.train",
            "args": args
        }
    }
    
    # Configure model training job
    custom_job = aiplatform.CustomJob(
        display_name=model_name,
        worker_pool_specs=[worker_pool_specs]
    )
    
    # Launch training
    job = custom_job.run()
    
    # Get the model
    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=job.output_dir,
                serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest"  # PyTorch 1.13 prediction container
    )
    
    print(f"Training job launched: {model.display_name}")
    print(f"Model resource name: {model.resource_name}")
    return model

if __name__ == "__main__":
    # All configuration is loaded from .env file
    model = launch_finetune_job()
    print("Fine-tuning job completed successfully")