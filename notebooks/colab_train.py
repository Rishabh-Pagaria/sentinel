# Google Colab Training Script for Phishing Detection
# 1. Mount Google Drive first (run the cell below)
# 2. Upload CSV to Google Drive at: /content/drive/MyDrive/phishing_project/
# 3. Then run this script

!pip install transformers datasets scikit-learn pandas torch

# Mount Google Drive to persist model checkpoints
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

# Check GPU
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# Load and filter data from Google Drive
print("Loading CSV...")
df = pd.read_csv("/content/drive/MyDrive/phishing_project/processed_phishing_data.csv")
df = df[["text", "label"]].dropna()
df["label"] = df["label"].astype(int)

# Filter out extremely long texts (the 17MB corrupted text)
original_len = len(df)
df['text_len'] = df['text'].str.len()
df = df[df['text_len'] < 100000]
df = df.drop(columns=['text_len'])
print(f"Filtered out {original_len - len(df)} samples with >100K characters")

# Split data
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
eval_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)
print(f"Train: {len(train_df)}, Eval: {len(eval_df)}, Test: {len(test_df)}")

# PyTorch Dataset class
class SimpleDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

# Load tokenizer and model from HuggingFace
print("Loading model from HuggingFace...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=2)

# Tokenize datasets with memory-efficient batching
def build_dataset(df, tokenizer, max_length=512):
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    
    print(f"Tokenizing {len(texts)} samples in batches...")
    all_input_ids = []
    all_attention_mask = []
    
    # Smaller batch size to avoid RAM spikes
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        if i % 500 == 0:
            print(f"  Processed {i}/{len(texts)} samples...")
        batch_texts = texts[i:i+batch_size]
        batch_enc = tokenizer(batch_texts, truncation=True, max_length=max_length, padding=False)
        all_input_ids.extend(batch_enc['input_ids'])
        all_attention_mask.extend(batch_enc['attention_mask'])
    
    encodings = {'input_ids': all_input_ids, 'attention_mask': all_attention_mask}
    print(f"✓ Tokenization complete: {len(labels)} samples")
    return SimpleDataset(encodings, labels)

train_ds = build_dataset(train_df, tokenizer)
eval_ds = build_dataset(eval_df, tokenizer)

# Collator for dynamic padding
def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attention_masks = []
    labels = []
    
    for item in batch:
        ids = item["input_ids"]
        attn = item["attention_mask"]
        pad_len = max_len - len(ids)
        
        input_ids.append(ids + [0] * pad_len)
        attention_masks.append(attn + [0] * pad_len)
        labels.append(item["labels"])
    
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_masks),
        "labels": torch.tensor(labels)
    }

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }

# Training arguments - save to Google Drive
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/phishing_project/deberta-phishing",  # Save to Drive
    per_device_train_batch_size=8,  # Reduced to avoid OOM errors
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,  # Effective batch size = 8 * 2 = 16
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
    report_to=[],
    fp16=True,  # Mixed precision for faster training + lower memory
    dataloader_num_workers=2,  # Parallel data loading
    save_total_limit=2,  # Only keep 2 checkpoints to save space
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

# Train
print("Training starting...")
trainer.train()
print("Training complete!")

# Evaluate on test set
print("\n=== Final Test Set Evaluation ===")
test_ds = build_dataset(test_df, tokenizer)
test_results = trainer.evaluate(test_ds)
print(f"Test Accuracy:  {test_results['eval_accuracy']:.4f}")
print(f"Test Precision: {test_results['eval_precision']:.4f}")
print(f"Test F1:        {test_results['eval_f1']:.4f}")
print(f"Test Loss:      {test_results['eval_loss']:.4f}")

# Save final model to Google Drive
save_path = "/content/drive/MyDrive/phishing_project/deberta-phishing-final"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"\n✓ Model saved to Google Drive at: {save_path}")
print("Even if Colab disconnects, your model is safe in Google Drive!")
print("\nTo use locally:")
print("1. Download the 'deberta-phishing-final' folder from Drive")
print("2. Extract to 'artifacts/encoders/deberta-v3-base'")
print("3. Use encoder.load() for predictions")
