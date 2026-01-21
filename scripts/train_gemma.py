# scripts/train_gemma.py

import argparse
import sys
import os
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_conversation(sample):
    """
    Convert JSONL sample to conversational format for SFTTrainer.
    SFTTrainer expects 'messages' format with role-based conversation.
    """
    return {
        "messages": [
            {"role": "user", "content": f"{sample['instruction']}\n\n{sample['input']}"},
            {"role": "assistant", "content": sample['output']}
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-2-2b for phishing detection")
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/train_gemma.jsonl",
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default="data/eval_gemma.jsonl",
        help="Path to evaluation JSONL file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="hf_models/gemma-2-2b",
        help="Path to local Gemma model or HF model name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/models/gemma-2-2b-phishing",
        help="Where to save the fine-tuned model",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help="Use QLoRA (4-bit quantization) for memory efficiency",
    )
    args = parser.parse_args()

    print(f"[train_gemma] Starting Gemma fine-tuning...")
    print(f"[train_gemma] Model: {args.model_path}")
    print(f"[train_gemma] Train file: {args.train_file}")
    print(f"[train_gemma] Output: {args.output_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets
    print("[train_gemma] Loading datasets...")
    dataset = load_dataset("json", data_files={"train": args.train_file, "test": args.eval_file})
    
    # Convert to conversational format
    print("[train_gemma] Converting to conversational format...")
    dataset = dataset.map(create_conversation, remove_columns=list(dataset["train"].features), batched=False)

    print(f"[train_gemma] Train samples: {len(dataset['train'])}")
    print(f"[train_gemma] Eval samples: {len(dataset['test'])}")
    
    # Configure quantization for QLoRA if enabled
    if args.use_qlora:
        print("[train_gemma] Using QLoRA (4-bit quantization)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None
        print("[train_gemma] Using standard LoRA (no quantization)...")

    # Load tokenizer from instruction-tuned version (has chat template)
    print("[train_gemma] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2-2b-it",  # Use instruction-tuned tokenizer (has chat template)
        trust_remote_code=True,
    )
    
    # Set padding token (Gemma doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    print("[train_gemma] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # Use eager attention (change to flash_attention_2 if supported)
    )

    # Configure LoRA (as per Google docs)
    print("[train_gemma] Configuring LoRA...")
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",  # Google recommends all-linear for Gemma
        task_type="CAUSAL_LM",
    )

    # Training arguments using SFTConfig
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch_fused",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to=[],
    )

    # Create SFTTrainer - remove incompatible parameters
    print("[train_gemma] Setting up SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Train
    print("[train_gemma] Starting training...")
    trainer.train()

    # Save final model
    print("[train_gemma] Saving final model and tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"[train_gemma] Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
