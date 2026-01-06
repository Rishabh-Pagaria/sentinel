# scripts/train_encoder.py

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.encoders import DebertaPhishingEncoder, EncoderConfig


def main():
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    config = EncoderConfig(
        output_dir=args.output_dir,
    )

    encoder = DebertaPhishingEncoder(config)
    encoder.train_from_csv(args.csv_path)


if __name__ == "__main__":
    main()