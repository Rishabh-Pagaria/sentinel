#!/usr/bin/env python3
"""
Create train/eval/test splits from the phishing dataset and save in JSONL format.
Uses preprocessing functions from prep_phish_jsonl.py for consistency.
"""

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import sys

# Import cleaning functions from prep_phish_jsonl
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prep_phish_jsonl import clean_text, find_urls, find_phrases, infer_tactics

def load_processed_data() -> pd.DataFrame:
    """Load and process the dataset from Hugging Face"""
    from datasets import load_dataset
    
    print("Loading dataset from Hugging Face...")
    ds = load_dataset("zefang-liu/phishing-email-dataset")
    train_data = ds['train']
    
    # Convert to DataFrame
    df = pd.DataFrame({
        "text": train_data["Email Text"],
        "label": [1 if x == "Phishing Email" else 0 for x in train_data["Email Type"]]
    })
    
    # Basic cleaning
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"] != ""].reset_index(drop=True)
    
    print(f"Loaded {len(df)} examples ({df['label'].sum()} phishing, {len(df) - df['label'].sum()} safe)")
    return df

def create_splits(df: pd.DataFrame, test_size: float = 0.1, eval_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/eval/test splits"""
    # First split out test set
    train_eval, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df["label"])
    
    # Then split remaining data into train and eval
    train, eval = train_test_split(
        train_eval, 
        test_size=eval_size/(1-test_size),  # Adjust size to get correct proportion
        random_state=42,
        stratify=train_eval["label"]
    )
    
    print(f"\nSplit sizes:")
    print(f"Train: {len(train)} ({len(train[train['label']==1])} phishing)")
    print(f"Eval:  {len(eval)} ({len(eval[eval['label']==1])} phishing)")
    print(f"Test:  {len(test)} ({len(test[test['label']==1])} phishing)")
    
    return train, eval, test

def format_example(text: str, label: int) -> Dict:
    """Format a single example in the required JSON structure"""
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Get tactics and evidence using heuristics
    tactics, evidence = infer_tactics(cleaned_text, "")  # No separate subject in this dataset
    
    # Format as required
    return {
        "input": f"EMAIL: {cleaned_text}",
        "output": {
            "label": "phish" if label == 1 else "benign",
            "confidence": 1.0,  # Using ground truth labels
            "tactics": tactics,
            "evidence": evidence,
            "user_tip": "Be cautious of unsolicited emails requesting urgent action or sensitive information."
        }
    }

def save_jsonl(data: pd.DataFrame, output_path: str):
    """Save data in JSONL format"""
    with open(output_path, 'w') as f:
        for _, row in data.iterrows():
            example = format_example(row['text'], row['label'])
            f.write(json.dumps(example) + '\n')
    print(f"Wrote {len(data)} examples to {output_path}")

def main():
    # Create output directory
    os.makedirs("../out_jsonl", exist_ok=True)
    
    # Load data
    df = load_processed_data()
    
    # Create splits
    train, eval, test = create_splits(df)
    
    # Save splits
    save_jsonl(train, "../out_jsonl/train.jsonl")
    save_jsonl(eval, "../out_jsonl/eval.jsonl")
    save_jsonl(test, "../out_jsonl/test.jsonl")
    
    print("\nDone! Files saved in out_jsonl/")

if __name__ == "__main__":
    main()