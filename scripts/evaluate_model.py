#!/usr/bin/env python3
"""
Evaluate the phishing detection model on:
1. Standard test set (test.jsonl)
2. Adversarial examples
Produces metrics and latency measurements.
"""

import json
import time
import os
import sys
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import asyncio

# Add parent directory to Python path to find app module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our classification endpoint
from src.api.app import classify_email, EmailRequest

def load_test_set(jsonl_path: str) -> List[Dict]:
    """Load test examples from JSONL file"""
    examples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples

def load_adversarial_examples() -> List[Dict]:
    """Load adversarial examples from JSONL file"""
    adv_path = "out_jsonl/eval_adv.jsonl"
    if not os.path.exists(adv_path):
        print(f"Warning: {adv_path} not found. Please run generate_adversarial.py first.")
        return []
    
    examples = []
    with open(adv_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples

def create_adversarial_examples() -> List[Dict]:
    """Create challenging test cases (legacy examples as backup)"""
    return [
        {
            "input": "EMAIL: Dear valued customer, Due to scheduled system maintenance, we need you to verify your account. Please note this is a legitimate request from your bank. Visit https://legitimate-bank.com to verify. Best regards, Your Bank Team",
            "output": {"label": "phish"}  # Looks legitimate but is phishing
        },
        {
            "input": "EMAIL: URGENT: Your order #12345 has been delayed due to weather conditions. Please expect 1-2 days delay in delivery. We apologize for any inconvenience. Track your order at https://amazon.com/orders",
            "output": {"label": "benign"}  # Urgent but legitimate
        },
        {
            "input": "EMAIL: Hey! Just following up on the urgent pr0ject deadline. Need you to l0g1n here: secure-c0rporate.com to update your status ASAP! - Your Manager",
            "output": {"label": "phish"}  # Obfuscated text and URL
        },
        {
            "input": "EMAIL: Your Google Drive storage is almost full (98% used). Upgrade now to avoid service interruption: https://drive.google.com/settings/storage",
            "output": {"label": "benign"}  # Legitimate service notification
        }
    ]

async def evaluate_examples(examples: List[Dict], name: str = "Test Set") -> Tuple[Dict, List[Dict]]:
    """
    Evaluate model on examples and collect metrics
    Returns: (metrics_dict, detailed_results)
    """
    y_true = []
    y_pred = []
    y_scores = []
    latencies = []
    detailed_results = []
    
    print(f"\nEvaluating {name} ({len(examples)} examples)...")
    
    for ex in tqdm(examples):
        # Handle both old and new formats
        if "input" in ex:
            # Original format
            input_text = ex["input"].replace("EMAIL: ", "")
            true_label = 1 if ex["output"]["label"] == "phish" else 0
        else:
            # Adversarial format
            input_text = ex["text"]
            true_label = 1 if ex["true_label"] == "phish" else 0
            
        # Time the prediction
        start_time = time.time()
        try:
            result = await classify_email(EmailRequest(text=input_text))
            latency = time.time() - start_time
            
            pred_label = 1 if result.label == "phish" else 0
            confidence = result.confidence
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            y_scores.append(confidence if pred_label == 1 else 1 - confidence)
            latencies.append(latency)
            
            detailed_results.append({
                "text": input_text[:100] + "...",
                "true_label": ex["output"]["label"],
                "pred_label": result.label,
                "confidence": confidence,
                "tactics": result.tactics,
                "correct": pred_label == true_label,
                "latency": latency
            })
            
        except Exception as e:
            print(f"Error processing example: {str(e)}")
            continue
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_scores)
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "latency_p50": np.percentile(latencies, 50),
        "latency_p95": np.percentile(latencies, 95),
        "total_examples": len(examples),
        "successful_predictions": len(y_pred)
    }
    
    return metrics, detailed_results

def plot_metrics(test_metrics: Dict, adv_metrics: Dict, output_dir: str = "evaluation_results"):
    """Create visualization of metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics comparison plot
    metrics = ['precision', 'recall', 'f1', 'auc']
    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, [test_metrics[m] for m in metrics], width, label='Test Set')
    plt.bar(x + width/2, [adv_metrics[m] for m in metrics], width, label='Adversarial Set')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_comparison.png")
    plt.close()
    
    # Save metrics to CSV
    pd.DataFrame({
        'Metric': metrics + ['latency_p50', 'latency_p95'],
        'Test Set': [test_metrics[m] for m in metrics] + [test_metrics['latency_p50'], test_metrics['latency_p95']],
        'Adversarial': [adv_metrics[m] for m in metrics] + [adv_metrics['latency_p50'], adv_metrics['latency_p95']]
    }).to_csv(f"{output_dir}/metrics.csv", index=False)

async def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"evaluation_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and evaluate test set
    test_examples = load_test_set("out_jsonl/test.jsonl")
    test_metrics, test_details = await evaluate_examples(test_examples, "Standard Test Set")
    
    # Evaluate on adversarial examples
    adv_examples = load_adversarial_examples()
    if not adv_examples:
        print("No adversarial examples found, falling back to legacy examples")
        adv_examples = create_adversarial_examples()
    adv_metrics, adv_details = await evaluate_examples(adv_examples, "Adversarial Set")
    
    # Plot results
    plot_metrics(test_metrics, adv_metrics, output_dir)
    
    # Save detailed results
    with open(f"{output_dir}/test_details.jsonl", 'w') as f:
        for result in test_details:
            f.write(json.dumps(result) + '\n')
    
    with open(f"{output_dir}/adversarial_details.jsonl", 'w') as f:
        for result in adv_details:
            f.write(json.dumps(result) + '\n')
    
    # Print summary
    print("\n=== Evaluation Results ===")
    print("\nTest Set Metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
    
    print("\nAdversarial Set Metrics:")
    for k, v in adv_metrics.items():
        print(f"{k}: {v:.4f}")
    
    print(f"\nDetailed results saved to: {output_dir}/")

if __name__ == "__main__":
    asyncio.run(main())