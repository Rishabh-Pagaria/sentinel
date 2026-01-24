#!/usr/bin/env python3
"""
Evaluate base Gemma-2-2b-it model (zero-shot) WITHOUT any fine-tuning.
Tests the instruction-tuned model's ability to detect phishing out-of-the-box.
"""

import argparse
import sys
import os
from pathlib import Path

# Disable torch dynamo BEFORE importing torch
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_jsonl(file_path):
    """Load JSONL file into list of dicts"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def parse_prediction(response_text):
    """
    Extract classification from natural language response.
    Looks for "phishing" or "safe" in the model's output.
    """
    response_lower = response_text.lower()
    
    # Check first line for quick classification
    first_line = response_text.split('\n')[0].lower()
    
    if "phishing" in first_line:
        return 1
    elif "safe" in first_line:
        return 0
    
    # Fallback: check entire response
    if "phishing" in response_lower:
        return 1
    elif "safe" in response_lower:
        return 0
    
    # Default to safe if unclear
    return 0


def get_ground_truth_label(output_text):
    """
    Extract ground truth label from output field.
    Handles both old format ("safe"/"phishing") and new format (explanations).
    """
    # Get first sentence/line only
    first_line = output_text.split('\n')[0].strip().lower()
    
    # Check ONLY the classification line
    if "is phishing" in first_line or first_line.startswith("phishing"):
        return 1
    elif "is safe" in first_line or first_line.startswith("safe"):
        return 0
    
    # Fallback for old single-word format
    if output_text.strip().lower() == "phishing":
        return 1
    elif output_text.strip().lower() == "safe":
        return 0
    
    return 0  # Default to safe


def evaluate_zero_shot(model, tokenizer, data, device, max_samples=None, batch_size=4):
    """
    Evaluate base model in zero-shot setting with batch processing for speed.
    """
    model.eval()
    predictions = []
    labels = []
    
    # Limit samples if specified
    eval_data = data[:max_samples] if max_samples else data
    
    print(f"\nEvaluating {len(eval_data)} samples in zero-shot mode...")
    print(f"Using batch size: {batch_size}")
    print("=" * 80)
    
    # Process in batches for speed
    for batch_start in tqdm(range(0, len(eval_data), batch_size), desc="Evaluating batches"):
        batch_end = min(batch_start + batch_size, len(eval_data))
        batch = eval_data[batch_start:batch_end]
        
        # Prepare batch prompts
        batch_prompts = []
        batch_labels = []
        
        for sample in batch:
            instruction = sample.get('instruction', 'Analyze this email and determine if it\'s phishing or safe. Explain your reasoning.')
            email_text = sample['input']
            
            messages = [
                {"role": "user", "content": f"{instruction}\n\n{email_text}"}
            ]
            
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_prompts.append(prompt)
            batch_labels.append(get_ground_truth_label(sample['output']))
        
        # Tokenize batch with padding
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)
        
        # Generate responses for batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,  # Reduced for speed - just need classification
                do_sample=False,    # Greedy for consistency and speed
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode batch outputs
        for i, output in enumerate(outputs):
            input_length = inputs['input_ids'][i].shape[0]
            new_tokens = output[input_length:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            pred_label = parse_prediction(response)
            predictions.append(pred_label)
            labels.append(batch_labels[i])
            
            # Show first few examples
            example_idx = batch_start + i
            if example_idx < 3:
                print(f"\n--- Example {example_idx + 1} ---")
                print(f"Email: {batch[i]['input'][:100]}...")
                print(f"Ground Truth: {'Phishing' if batch_labels[i] == 1 else 'Safe'}")
                print(f"Model Response: {response[:200]}...")
                print(f"Predicted: {'Phishing' if pred_label == 1 else 'Safe'}")
                print(f"Correct: {'✓' if pred_label == batch_labels[i] else '✗'}")
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    cm = confusion_matrix(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "num_samples": len(labels),
    }


def main():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation of base Gemma-2-2b-it model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="hf_models/gemma-2-2b-it",
        help="Path to base Gemma-2-2b-it model",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/test_gemma.jsonl",
        help="Path to test JSONL file",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: None = all samples)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="artifacts/zero_shot_evaluation.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8, increase for speed if you have VRAM)",
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("ZERO-SHOT EVALUATION - Base Gemma-2-2b-it Model")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Test file: {args.test_file}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'All'}")
    print("=" * 80)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model with 4-bit quantization
    print("Loading base model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")
    
    # Load test data
    print(f"\nLoading test data from {args.test_file}...")
    test_data = load_jsonl(args.test_file)
    print(f"Loaded {len(test_data)} test samples")
    
    # Evaluate
    results = evaluate_zero_shot(
        model=model,
        tokenizer=tokenizer,
        data=test_data,
        device=device,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("ZERO-SHOT EVALUATION RESULTS")
    print("=" * 80)
    print(f"Samples evaluated: {results['num_samples']}")
    print(f"Accuracy:  {results['accuracy']:.2%}")
    print(f"Precision: {results['precision']:.2%}")
    print(f"Recall:    {results['recall']:.2%}")
    print(f"F1 Score:  {results['f1']:.2%}")
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Safe  Phishing")
    print(f"Actual Safe   {results['confusion_matrix'][0][0]:4d}  {results['confusion_matrix'][0][1]:4d}")
    print(f"       Phish  {results['confusion_matrix'][1][0]:4d}  {results['confusion_matrix'][1][1]:4d}")
    print("=" * 80)
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output_file}")
    
    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    if results['accuracy'] > 0.8:
        print("✓ Base model shows strong zero-shot performance!")
        print("  Consider whether fine-tuning is necessary.")
    elif results['accuracy'] > 0.6:
        print("○ Base model shows moderate zero-shot performance.")
        print("  Fine-tuning should improve results.")
    else:
        print("✗ Base model shows weak zero-shot performance.")
        print("  Fine-tuning is highly recommended.")
    
    print(f"\nNext step: Fine-tune the model with: python scripts/train_gemma.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
