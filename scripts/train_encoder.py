# scripts/train_encoder.py

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.encoders import DebertaPhishingEncoder, EncoderConfig


def main():
    parser = argparse.ArgumentParser(
        description="Train DeBERTa encoder for phishing detection on GPU"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to processed phishing CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/encoders/deberta-v3-base",
        help="Where to save the trained model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="hf_models/deberta-v3-base",
        help="Path to local DeBERTa model",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512 for GPU)",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Training batch size per GPU (default: 8)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="Evaluation batch size (default: 16)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2, effective batch = 16)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    args = parser.parse_args()

    # GPU-optimized configuration
    config = EncoderConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    print("="*80)
    print("DeBERTa Phishing Detection Training - GPU Optimized")
    print("="*80)
    print(f"Model: {config.model_name}")
    print(f"Max length: {config.max_length}")
    print(f"Batch size (train): {config.train_batch_size}")
    print(f"Batch size (eval): {config.eval_batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.train_batch_size * config.gradient_accumulation_steps}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Output: {config.output_dir}")
    print("="*80)

    encoder = DebertaPhishingEncoder(config)
    encoder.train_from_csv(args.csv_path)


if __name__ == "__main__":
    main()