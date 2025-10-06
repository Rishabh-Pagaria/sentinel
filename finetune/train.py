#!/usr/bin/env python3
"""
Training script for fine-tuning LLM on phishing detection task using Vertex AI.
"""

import os
from dotenv import load_dotenv
import json
import argparse
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_dataset
import evaluate
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# Load environment variables
load_dotenv()

# Constants from environment
MODEL_NAME = os.getenv("BASE_MODEL", "google/flan-t5-base")  # Base model to fine-tune
MAX_LENGTH = 512  # Maximum sequence length
NUM_LABELS = 2  # Binary classification: phishing or benign
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION")
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")

# Data paths in GCS
TRAIN_PATH = "gs://{}/vertex_ai/train.jsonl".format(BUCKET_NAME)
EVAL_PATH = "gs://{}/vertex_ai/eval.jsonl".format(BUCKET_NAME)
TEST_PATH = "gs://{}/vertex_ai/test.jsonl".format(BUCKET_NAME)

class PhishingDataset(Dataset):
    """Dataset for phishing detection fine-tuning"""
    
    def __init__(self, jsonl_file: str, tokenizer, max_length: int = MAX_LENGTH):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and process examples from Vertex AI format
        with open(jsonl_file, 'r') as f:
            for line in f:
                example = json.loads(line.strip())
                self.data.append({
                    'text': example['text_snippet']['content'],
                    'label': 1 if example['labels'][0]['label'] == 'phish' else 0
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate standard metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    # Calculate ROC-AUC
    try:
        auc = roc_auc_score(labels, predictions)
    except:
        auc = 0.0
    
    return {
        'accuracy': (predictions == labels).mean(),
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }

def train(
    train_file: str,
    eval_file: str,
    output_dir: str,
    batch_size: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 2e-5
):
    """Main training function"""
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )
    
    # Create datasets
    train_dataset = PhishingDataset(train_file, tokenizer)
    eval_dataset = PhishingDataset(eval_file, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="tensorboard",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model(f"{output_dir}/final-model")
    tokenizer.save_pretrained(f"{output_dir}/final-model")
    
    # Final evaluation
    final_metrics = trainer.evaluate()
    
    # Save metrics
    with open(f"{output_dir}/final_metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    return final_metrics

def main():
    parser = argparse.ArgumentParser(description="Train phishing detection model")
    parser.add_argument("--train-file", required=True, help="Path to training data JSONL")
    parser.add_argument("--eval-file", required=True, help="Path to evaluation data JSONL")
    parser.add_argument("--output-dir", required=True, help="Directory to save model and outputs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    metrics = train(
        train_file=args.train_file,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )
    
    print("\nTraining completed! Final metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()