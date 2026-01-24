# scripts/download_deberta.py

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    model_id = "microsoft/deberta-v3-base"
    local_dir = Path("hf_models/deberta-v3-base")
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Download] Downloading tokenizer '{model_id}' -> {local_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(local_dir)
    print("[Download] Tokenizer saved.")

    print(f"[Download] Downloading model '{model_id}' -> {local_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,   # we want a 2-class head
    )
    model.save_pretrained(local_dir)
    print("[Download] Model saved.")

if __name__ == "__main__":
    main()
