#!/usr/bin/env python3
"""
Download, clean, and split phishing email dataset from HuggingFace.
Creates clean training data with consistent output format for Gemma fine-tuning.

Output format:
- instruction: "Analyze this email and determine if it's phishing or legitimate"
- input: "EMAIL: [email text]"
- output: "phishing\n[brief reason]" OR "legitimate\n[brief reason]"
"""

import json
from pathlib import Path
from typing import Dict, List
from datasets import load_dataset
import random

def download_and_prepare_data() -> List[Dict]:
    """Download phishing dataset from HuggingFace and prepare samples"""
    print("Downloading phishing dataset from HuggingFace...")
    
    try:
        ds = load_dataset("zefang-liu/phishing-email-dataset", cache_dir="hf_cache")
        train_data = ds["train"]
        
        print(f"Downloaded {len(train_data)} samples")
        
        samples = []
        for i, item in enumerate(train_data):
            # Handle None values from dataset
            email_text = item.get("Email Text") or ""
            email_type = item.get("Email Type") or ""
            
            email_text = email_text.strip() if isinstance(email_text, str) else ""
            email_type = email_type.strip() if isinstance(email_type, str) else ""
            
            if not email_text:
                continue
            
            # Determine classification
            if "Phishing" in email_type:
                label = "phishing"
                reason = "Contains phishing characteristics"
            else:
                label = "legitimate"
                reason = "Appears to be legitimate communication"
            
            sample = {
                "instruction": "Analyze this email and determine if it's phishing or legitimate.",
                "input": f"EMAIL: {email_text[:1000]}",  # Limit to 1000 chars
                "output": f"{label}\n{reason}",
                "original_type": email_type
            }
            samples.append(sample)
        
        print(f"Prepared {len(samples)} samples")
        return samples
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Falling back to using existing train_gemma.jsonl...")
        return None

def load_existing_data(jsonl_path: str) -> List[Dict]:
    """Load existing training data from JSONL file"""
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError:
                continue
    return samples

def clean_sample_output(output: str) -> str:
    """
    Extract classification from output, handling various formats.
    Returns: "phishing" or "legitimate" (single label only)
    """
    # Remove extra whitespace and normalize
    output_lower = output.lower()
    
    # Determine label - check for keywords
    if "phishing" in output_lower or "phish" in output_lower or "suspicious" in output_lower:
        return "phishing"
    elif "legitimate" in output_lower or "safe" in output_lower or "benign" in output_lower:
        return "legitimate"
    else:
        return None

def create_clean_dataset(samples: List[Dict]) -> List[Dict]:
    """
    Clean dataset to have consistent format.
    Format: "phishing" or "legitimate" (single label)
    """
    clean_samples = []
    skipped = 0
    
    for i, sample in enumerate(samples):
        try:
            output = sample.get("output", "")
            input_text = sample.get("input", "")
            instruction = sample.get("instruction", "")
            
            # Clean the output
            classification = clean_sample_output(output)
            
            if not classification:
                skipped += 1
                continue
            
            # Create clean sample
            clean_sample = {
                "instruction": instruction or "Analyze this email and determine if it's phishing or legitimate.",
                "input": input_text,
                "output": classification  # Just the label, no reason
            }
            clean_samples.append(clean_sample)
            
        except Exception as e:
            skipped += 1
            continue
    
    print(f"Cleaned {len(clean_samples)} samples (skipped {skipped})")
    return clean_samples

def split_dataset(samples: List[Dict]):
    """Split dataset into train, eval, and test (80/10/10)"""
    random.shuffle(samples)  # Randomize to avoid bias
    
    train_idx = int(len(samples) * 0.8)
    eval_idx = int(len(samples) * 0.9)
    
    train_samples = samples[:train_idx]
    eval_samples = samples[train_idx:eval_idx]
    test_samples = samples[eval_idx:]
    
    return train_samples, eval_samples, test_samples

def save_jsonl(samples: List[Dict], output_path: str):
    """Save samples to JSONL file"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Saved {len(samples)} samples to {output_path}")

def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("PHISHING EMAIL DATASET PREPARATION")
    print("=" * 70)
    
    # Try to download fresh data
    samples = download_and_prepare_data()
    
    # If download fails, use existing data
    if samples is None:
        print("Using existing training data...")
        train_file = data_dir / "train_gemma.jsonl"
        if train_file.exists():
            samples = load_existing_data(str(train_file))
            eval_file = data_dir / "eval_gemma.jsonl"
            if eval_file.exists():
                samples.extend(load_existing_data(str(eval_file)))
            test_file = data_dir / "test_gemma.jsonl"
            if test_file.exists():
                samples.extend(load_existing_data(str(test_file)))
        else:
            print("ERROR: No data found!")
            return
    
    print(f"\nTotal samples: {len(samples)}")
    
    # Clean the dataset
    print("\nCleaning dataset to ensure consistent format...")
    clean_samples = create_clean_dataset(samples)
    
    # Split into train, eval, and test
    print("\nSplitting dataset (80% train, 10% eval, 10% test)...")
    train_samples, eval_samples, test_samples = split_dataset(clean_samples)
    
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Eval samples: {len(eval_samples)}")
    print(f"  Test samples: {len(test_samples)}")
    
    # Save
    print("\nSaving cleaned datasets...")
    save_jsonl(train_samples, str(data_dir / "train_gemma_clean.jsonl"))
    save_jsonl(eval_samples, str(data_dir / "eval_gemma_clean.jsonl"))
    save_jsonl(test_samples, str(data_dir / "test_gemma_clean.jsonl"))
    
    # Show sample outputs
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUTS (cleaned format):")
    print("=" * 70)
    
    for i, sample in enumerate(train_samples[:3]):
        print(f"\nSample {i+1}:")
        print(f"  Input preview: {sample['input'][:80]}...")
        print(f"  Output:\n    {sample['output'].replace(chr(10), chr(10) + '    ')}")
    
    print("\n" + "=" * 70)
    print("✓ Dataset preparation complete!")
    print(f"✓ Ready for training with:")
    print(f"  - train_gemma_clean.jsonl ({len(train_samples)} samples)")
    print(f"  - eval_gemma_clean.jsonl ({len(eval_samples)} samples)")
    print(f"  - test_gemma_clean.jsonl ({len(test_samples)} samples)")
    print("=" * 70)

if __name__ == "__main__":
    main()
