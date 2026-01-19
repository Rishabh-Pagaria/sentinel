# models/encoders.py

import os
import glob
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


@dataclass
class EncoderConfig:
    model_name: str = "hf_models/deberta-v3-base"    # <--- local model folder
    max_length: int = 512
    stride: int = 64                                 # chunk overlap: 512 - 64 = 448
    output_dir: str = "artifacts/encoders/deberta-v3-base"
    learning_rate: float = 2e-5
    num_train_epochs: int = 1  # Reduced from 3 for faster CPU training
    train_batch_size: int = 8
    eval_batch_size: int = 16
    weight_decay: float = 0.01
    seed: int = 42
    max_samples: int = None  # Set to e.g. 1000 for quick testing


class DebertaPhishingEncoder:
    def __init__(self, config: Optional[EncoderConfig] = None):
        self.config = config or EncoderConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)

        self.tokenizer = None
        self.model = None

    # ---------------------- Data Helpers ---------------------- #
    def load_csv(self, csv_path, text_col="cleaned_text", label_col="label",
                 eval_size=0.1, test_size=0.1):
        df = pd.read_csv(csv_path)

        if text_col not in df.columns:
            if "text" in df.columns:
                text_col = "text"
            else:
                raise ValueError("No text column found in CSV")

        df = df[[text_col, label_col]].dropna()
        df = df.rename(columns={text_col: "text", label_col: "label"})
        df["label"] = df["label"].astype(int)
        
        # Filter out extremely long texts that cause tokenizer to hang
        original_len = len(df)
        df['text_len'] = df['text'].str.len()
        df = df[df['text_len'] < 100000]  # Max 100K characters
        df = df.drop(columns=['text_len'])
        filtered = original_len - len(df)
        if filtered > 0:
            print(f"[Encoder] Filtered out {filtered} samples with >100K characters", flush=True)

        from sklearn.model_selection import train_test_split
        temp_size = eval_size + test_size

        train_df, temp_df = train_test_split(
            df, test_size=temp_size, stratify=df["label"], random_state=self.config.seed
        )
        relative_eval = eval_size / temp_size

        eval_df, test_df = train_test_split(
            temp_df, test_size=(1 - relative_eval), stratify=temp_df["label"],
            random_state=self.config.seed
        )
        return train_df, eval_df, test_df

    # ---------------------- Chunking -------------------------- #
    def _chunk_text(self, text: str):
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            return_overflowing_tokens=True,
            stride=self.config.max_length - self.config.stride,
            return_tensors=None
        )

        chunks = []
        for ids, attn in zip(enc["input_ids"], enc["attention_mask"]):
            chunks.append({"input_ids": ids, "attention_mask": attn})
        return chunks

    # ---------------------- PyTorch Dataset -------------------- #
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

    def _build_dataset(self, df: pd.DataFrame):
        """
        Use PyTorch Dataset with chunked tokenization.
        """
        print(f"[Encoder] Tokenizing {len(df)} samples in batches...", flush=True)
        
        texts = df["text"].tolist()
        labels = df["label"].tolist()
        
        all_input_ids = []
        all_attention_mask = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            if i % 1000 == 0:
                print(f"[Encoder] Tokenized {i}/{len(texts)}...", flush=True)
            
            batch_texts = texts[i:i+batch_size]
            batch_enc = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.config.max_length,
                padding=False,
                return_tensors=None
            )
            all_input_ids.extend(batch_enc['input_ids'])
            all_attention_mask.extend(batch_enc['attention_mask'])
        
        print(f"[Encoder] Creating PyTorch dataset...", flush=True)
        encodings = {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask
        }
        dataset = self.SimpleDataset(encodings, labels)
        
        print(f"[Encoder] Complete: {len(dataset)} samples", flush=True)
        return dataset

    # ---------------------- Collator --------------------------- #
    @staticmethod
    def _collate_fn(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = 0

        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            ids = item["input_ids"]
            attn = item["attention_mask"]
            pad_len = max_len - len(ids)

            input_ids.append(ids + [pad_id] * pad_len)
            attention_masks.append(attn + [0] * pad_len)
            labels.append(item["labels"])

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_masks),
            "labels": torch.tensor(labels)
        }

    # ---------------------- Training --------------------------- #
    def train_from_csv(self, csv_path):
        print("[Encoder] Loading CSV...", flush=True)
        train_df, eval_df, test_df = self.load_csv(csv_path)
        print(f"[Encoder] Loaded: train={len(train_df)}, eval={len(eval_df)}, test={len(test_df)}", flush=True)

        print("[Encoder] Loading tokenizer & model locally...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=2,
            local_files_only=True
        )

        print("[Encoder] Building datasets (fast tokenization, no sliding window)...", flush=True)
        print("[Encoder] Processing training data...", flush=True)
        train_ds = self._build_dataset(train_df)
        print("[Encoder] Processing evaluation data...", flush=True)
        eval_ds = self._build_dataset(eval_df)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy_score(labels, preds),
                "precision": precision_score(labels, preds, zero_division=0),
                "recall": recall_score(labels, preds, zero_division=0),
                "f1": f1_score(labels, preds, zero_division=0),
            }

        args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            num_train_epochs=self.config.num_train_epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            logging_steps=50,
            report_to=[],
            seed=self.config.seed
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            data_collator=self._collate_fn,
            compute_metrics=compute_metrics
        )

        print("[Encoder] Training starting...", flush=True)
        trainer.train()
        print("[Encoder] Training complete!", flush=True)

        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

        print("[Encoder] Running evaluation on test set...", flush=True)
        self.evaluate_on_dataframe(test_df)

    # ---------------------- Inference -------------------------- #
    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.output_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.output_dir)
        return self

    def predict_proba_email(self, text: str):
        chunks = self._chunk_text(text)
        probs = []

        with torch.no_grad():
            for c in chunks:
                logits = self.model(
                    input_ids=torch.tensor([c["input_ids"]]),
                    attention_mask=torch.tensor([c["attention_mask"]])
                ).logits
                p = torch.softmax(logits, dim=-1)[0, 1].item()
                probs.append(p)

        return max(probs)

    def evaluate_on_dataframe(self, df):
        y_true = df["label"].tolist()
        y_pred = [1 if self.predict_proba_email(t) >= 0.5 else 0 for t in df["text"].tolist()]

        print(classification_report(y_true, y_pred, digits=4))
