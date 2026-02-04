#!/usr/bin/env python3
"""
Zero-shot evaluation of DeBERTa model on phishing detection task.
- Loads processed_phishing_data.csv
- Splits data into train (80%), eval (10%), test (10%)
- Performs zero-shot evaluation on test set using the local DeBERTa model
- Reports metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_and_split_data(csv_path: str, test_size: float = 0.1, val_size: float = 0.1):
    """
    Load CSV data and split into train (80%), eval (10%), test (10%).
    
    Args:
        csv_path: Path to processed_phishing_data.csv
        test_size: Proportion for test set
        val_size: Proportion for eval set
        
    Returns:
        Tuple of (train_df, eval_df, test_df)
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # First split: separate test set (10%)
    train_eval_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df['label']
    )
    
    # Second split: separate eval from train (10% of remaining = ~9% of total, close to 10%)
    adjusted_val_size = val_size / (1 - test_size)
    train_df, eval_df = train_test_split(
        train_eval_df, test_size=adjusted_val_size, random_state=42, stratify=train_eval_df['label']
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Eval: {len(eval_df)} samples ({len(eval_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, eval_df, test_df


def setup_model_and_tokenizer(model_dir: str):
    """
    Load the DeBERTa model and tokenizer from local directory.
    
    Args:
        model_dir: Path to local model directory
        
    Returns:
        Tuple of (model, tokenizer, device)
    """
    print(f"\nLoading model from {model_dir}...")
    
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Model config: {model.config}")
    
    return model, tokenizer, device


def get_predictions(model, tokenizer, texts, device, labels=None, batch_size=32):
    """
    Get model predictions for a batch of texts.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        texts: List of input texts
        device: Device to run inference on
        labels: Optional true labels for metrics calculation
        batch_size: Batch size for inference
        
    Returns:
        Predictions and optionally metrics
    """
    predictions = []
    scores = []
    
    print(f"Running inference on {len(texts)} samples...")
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predictions
        logits = outputs.logits
        batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
        batch_scores = torch.softmax(logits, dim=1).cpu().numpy()
        
        predictions.extend(batch_preds)
        scores.extend(batch_scores)
    
    predictions = np.array(predictions)
    scores = np.array(scores)
    
    return predictions, scores


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp),
        'Num Samples': len(y_true),
    }
    
    return metrics


def evaluate_zero_shot(model, tokenizer, dataset_df, device, split_name="Test"):
    """
    Perform zero-shot evaluation on a dataset.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        dataset_df: DataFrame with 'text' and 'label' columns
        device: Device to run inference on
        split_name: Name of the split for logging
        
    Returns:
        Dictionary of metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on {split_name} Set ({len(dataset_df)} samples)")
    print(f"{'='*60}")
    
    texts = dataset_df['text'].tolist()
    y_true = dataset_df['label'].values
    
    # Get predictions
    y_pred, scores = get_predictions(model, tokenizer, texts, device, y_true)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Print metrics
    print(f"\n{split_name} Set Metrics:")
    print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1:        {metrics['F1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics['TN']}, FP: {metrics['FP']}")
    print(f"  FN: {metrics['FN']}, TP: {metrics['TP']}")
    
    return metrics


def main():
    """Main evaluation pipeline."""
    
    # Paths
    data_path = project_root / "data" / "processed_phishing_data.csv"
    model_path = project_root / "hf_models" / "deberta-v3-base"
    output_path = project_root / "artifacts" / "deberta_zero_shot_results.csv"
    
    # Check paths exist
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    # Load and split data
    train_df, eval_df, test_df = load_and_split_data(str(data_path))
    
    # Setup model
    model, tokenizer, device = setup_model_and_tokenizer(str(model_path))
    
    # Evaluate on all splits
    print("\n" + "="*80)
    print("ZERO-SHOT EVALUATION ON DeBERTa MODEL")
    print("="*80)
    
    eval_metrics = evaluate_zero_shot(model, tokenizer, eval_df, device, split_name="Eval")
    test_metrics = evaluate_zero_shot(model, tokenizer, test_df, device, split_name="Test")
    
    # Prepare results for CSV
    results = [
        {
            'Set': 'eval',
            'Accuracy': eval_metrics['Accuracy'],
            'Precision': eval_metrics['Precision'],
            'Recall': eval_metrics['Recall'],
            'F1': eval_metrics['F1'],
            'TN': eval_metrics['TN'],
            'FP': eval_metrics['FP'],
            'FN': eval_metrics['FN'],
            'TP': eval_metrics['TP'],
            'Num Samples': eval_metrics['Num Samples'],
        },
        {
            'Set': 'test',
            'Accuracy': test_metrics['Accuracy'],
            'Precision': test_metrics['Precision'],
            'Recall': test_metrics['Recall'],
            'F1': test_metrics['F1'],
            'TN': test_metrics['TN'],
            'FP': test_metrics['FP'],
            'FN': test_metrics['FN'],
            'TP': test_metrics['TP'],
            'Num Samples': test_metrics['Num Samples'],
        }
    ]
    
    results_df = pd.DataFrame(results)
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n\nResults saved to {output_path}")
    print("\nFinal Results:")
    print(results_df.to_string(index=False))
    
    # Also save as JSON for more detail
    json_path = project_root / "artifacts" / "deberta_zero_shot_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'eval': eval_metrics,
            'test': test_metrics,
        }, f, indent=2)
    print(f"Detailed results saved to {json_path}")


if __name__ == "__main__":
    main()
