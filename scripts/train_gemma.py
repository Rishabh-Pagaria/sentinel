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


def tokenize_function(examples, tokenizer):
    """
    Pre-tokenize dataset to avoid on-the-fly tokenization during training.
    This is the BIGGEST speed optimization - tokenizing once vs every batch.
    """
    # Apply chat template
    texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) 
             for msg in examples["messages"]]
    
    # Tokenize with truncation only - NO padding (save memory)
    tokenized = tokenizer(
        texts,
        max_length=512,  # Limit sequence length for speed
        truncation=True,
        padding=False,  # Dynamic padding in data collator saves memory
        return_tensors=None,
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


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
        default="hf_models/gemma-2-2b-it",
        help="Path to local Gemma-2-2b-it model (instruction-tuned)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/models/gemma-2-2b-it-phishing",
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
        default=1,  # Reduced to 1 to avoid OOM errors
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
        args.model_path,  # Use same path as model
        trust_remote_code=True,
    )
    
    # Set padding token (Gemma doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # PRE-TOKENIZE DATASET (CRITICAL FOR SPEED)
    print("[train_gemma] Pre-tokenizing dataset (this is crucial for training speed)...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        batch_size=1000,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    print(f"[train_gemma] Train samples: {len(tokenized_dataset['train'])}")
    print(f"[train_gemma] Eval samples: {len(tokenized_dataset['test'])}")

    # Load model
    print("[train_gemma] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # flash_attention_2 not available on Windows
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
        gradient_accumulation_steps=32,  # Increased to maintain effective batch=32
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch_fused",
        logging_steps=50,  # Reduced logging frequency (was 10)
        save_strategy="steps",  # Save by steps instead of epoch
        save_steps=200,  # Save every 200 steps (more frequent to avoid losing progress)
        eval_strategy="no",  # Disable evaluation to avoid OOM
        learning_rate=args.learning_rate,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        max_seq_length=512,  # Explicit max length for speed
        report_to=[],
        dataloader_num_workers=0,  # Disable parallel loading on Windows (causes issues)
        dataloader_pin_memory=False,  # Disable for stability
    )

    # Create SFTTrainer - use pre-tokenized dataset
    print("[train_gemma] Setting up SFTTrainer...")
    from transformers import DataCollatorForSeq2Seq
    
    # Use DataCollatorForSeq2Seq for dynamic padding (memory efficient)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,  # Pad dynamically to longest in batch
        return_tensors="pt",
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        peft_config=peft_config,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Train (resume from checkpoint if it exists)
    print("[train_gemma] Starting training...")
    checkpoint_dir = None
    if os.path.exists(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            # Get the latest checkpoint
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            checkpoint_dir = os.path.join(args.output_dir, checkpoints[-1])
            print(f"[train_gemma] Resuming from checkpoint: {checkpoint_dir}")
    
    trainer.train(resume_from_checkpoint=checkpoint_dir)

    # Save final model
    print("[train_gemma] Saving final model and tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"[train_gemma] Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
