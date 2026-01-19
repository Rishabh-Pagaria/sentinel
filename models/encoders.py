# model/encoders.py

import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import traceback
import glob

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


@dataclass
class EncoderConfig:
    model_name: str = "hf_models/deberta-v3-base"  # Use local downloaded model
    max_length: int = 512
    stride: int = 448                 # sliding window => 64 token overlap
    output_dir: str = "artifacts/encoders/deberta-v3-base"
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    train_batch_size: int = 16
    eval_batch_size: int = 32
    weight_decay: float = 0.01
    seed: int = 42


class DebertaPhishingEncoder:
    """
    Wrapper around DeBERTa-v3 for phishing detection.

    - Uses sliding-window chunking for long emails.
    - Trains with HuggingFace Trainer.
    - Provides email-level predict_proba() and evaluation helpers.
    """

    def __init__(self, config: Optional[EncoderConfig] = None):
        self.config = config or EncoderConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None

    # ---------------------- Data helpers ---------------------- #

    def load_csv(
        self,
        csv_path: str = "data/processed_phishing_data.csv",
        text_col: str = "cleaned_text",
        label_col: str = "label",
        eval_size: float = 0.1,
        test_size: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load a single CSV and split into train / eval / test.
        Assumes binary labels 0/1.
        """
        df = pd.read_csv(csv_path)
        # fallback to "text" if "cleaned_text" doesn't exist
        if text_col not in df.columns and "text" in df.columns:
            text_col = "text"

        df = df[[text_col, label_col]].dropna()
        df = df.rename(columns={text_col: "text", label_col: "label"})
        df["label"] = df["label"].astype(int)

        from sklearn.model_selection import train_test_split

        # First split off test
        temp_size = eval_size + test_size
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_size,
            stratify=df["label"],
            random_state=self.config.seed,
        )
        # Then split temp into eval & test
        relative_eval = eval_size / temp_size
        eval_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - relative_eval),
            stratify=temp_df["label"],
            random_state=self.config.seed,
        )

        return train_df.reset_index(drop=True), eval_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def _chunk_text(self, text: str) -> List[Dict[str, List[int]]]:
        """
        Tokenize text with sliding window. Returns a list of chunks,
        each chunk is a dict with 'input_ids' and 'attention_mask'.
        """
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            padding=False,
            return_overflowing_tokens=True,
            stride=self.config.stride,
            return_tensors=None,
        )
        
        chunks = []
        num_chunks = len(enc["input_ids"])
        for i in range(num_chunks):
            chunks.append({
                "input_ids": enc["input_ids"][i],
                "attention_mask": enc["attention_mask"][i]
            })
        return chunks

    def _build_dataset(self, df: pd.DataFrame) -> Dataset:
        """
        Convert a dataframe (text, label) into a HF Dataset with chunk-level rows.
        Each email can produce multiple chunks; all share the same label.
        """
        all_input_ids, all_attn, all_labels = [], [], []
        for _, row in df.iterrows():
            chunks = self._chunk_text(str(row["text"]))
            if not chunks:
                continue
            for ch in chunks:
                all_input_ids.append(ch["input_ids"])
                all_attn.append(ch["attention_mask"])
                all_labels.append(int(row["label"]))

        return Dataset.from_dict(
            {
                "input_ids": all_input_ids,
                "attention_mask": all_attn,
                "labels": all_labels,
            }
        )

    @staticmethod
    def _collate_fn(features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Robust collator that handles various data types.
        """
        processed = []

        for f in features:
            ids = f["input_ids"]
            attn = f["attention_mask"]
            label = f["labels"]

            # Handle scalar inputs
            if isinstance(ids, (int, np.integer)):
                ids = [int(ids)]
            if isinstance(attn, (int, np.integer)):
                attn = [int(attn)]

            # Convert numpy arrays/tensors to lists
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if hasattr(attn, "tolist"):
                attn = attn.tolist()

            processed.append({
                "input_ids": ids,
                "attention_mask": attn,
                "labels": int(label),
            })

        max_len = max(len(p["input_ids"]) for p in processed)
        pad_id = 0

        batch_input_ids = []
        batch_attn = []
        labels = []

        for p in processed:
            ids = p["input_ids"]
            attn = p["attention_mask"]
            pad_len = max_len - len(ids)

            batch_input_ids.append(ids + [pad_id] * pad_len)
            batch_attn.append(attn + [0] * pad_len)
            labels.append(p["labels"])

        return {
        print("[Encoder] Building HF Datasets with sliding-window chunks...", flush=True)
        train_ds = self._build_dataset(train_df)
        eval_ds = self._build_dataset(eval_df)

        # Use sklearn metrics directly to avoid evaluate library bugs
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            
            return {
                "accuracy": float(accuracy_score(labels, preds)),
                "precision": float(precision_score(labels, preds, average='binary', zero_division=0)),
                "recall": float(recall_score(labels, preds, average='binary', zero_division=0)),
                "f1": float(f1_score(labels, preds, average='binary', zero_division=0)),
            }se_fast=False,
            local_files_only=True,  # Use only local files, don't download
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=2,
            local_files_only=True,  # Use only local files, don't download
        )

        print("[Encoder] Building HF Datasets with sliding-window chunks...")
        train_ds = self._build_dataset(train_df)
        eval_ds = self._build_dataset(eval_df)

        f1_metric = evaluate.load("f1")
        acc_metric = evaluate.load("accuracy")
        prec_metric = evaluate.load("precision")
        rec_metric = evaluate.load("recall")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
                "precision": prec_metric.compute(predictions=preds, references=labels)["precision"],
                "recall": rec_metric.compute(predictions=preds, references=labels)["recall"],
                "f1": f1_metric.compute(predictions=preds, references=labels)["f1"],
            }

        args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            num_train_epochs=self.config.num_train_epochs,
        print("[Encoder] Starting training...", flush=True)
        
        # Check if checkpoint exists to resume training
        checkpoints = glob.glob(f"{self.config.output_dir}/checkpoint-*")
        resume_from_checkpoint = None
        if checkpoints:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
            print(f"[Encoder] Found checkpoint: {latest_checkpoint}", flush=True)
            print(f"[Encoder] Resuming training from checkpoint...", flush=True)
            resume_from_checkpoint = latest_checkpoint
        
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        print("[Encoder] Saving model & tokenizer...", flush=True)
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=torch.cuda.is_available(),
            report_to=[],
            seed=self.config.seed,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            data_collator=self._collate_fn,
            compute_metrics=compute_metrics,
        )

        print("[Encoder] Starting training...")
        trainer.train()

        print("[Encoder] Saving model & tokenizer...")
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

        print("[Encoder] Running email-level evaluation on test set...")
        self.evaluate_on_dataframe(test_df)

    # ---------------------- Loading / Inference ---------------------- #

    def load(self):
        """
        Load a previously fine-tuned model from disk.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.output_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.output_dir)
        return self

    def predict_proba_email(self, text: str) -> float:
        """
        Return P(phishing=1) for a single email, using max-pooled
        sliding-window chunks.
        """
        assert self.tokenizer is not None and self.model is not None, "Call load() or train_from_csv() first."

        self.model.eval()
        device = next(self.model.parameters()).device

        chunks = self._chunk_text(str(text))
        if not chunks:
            return 0.0

        probs = []
        with torch.no_grad():
            for ch in chunks:
                inputs = {
                    "input_ids": torch.tensor([ch["input_ids"]], dtype=torch.long, device=device),
                    "attention_mask": torch.tensor([ch["attention_mask"]], dtype=torch.long, device=device),
                }
                logits = self.model(**inputs).logits  # (1,2)
                p = torch.softmax(logits, dim=-1)[0, 1].item()
                probs.append(p)

        return float(np.max(probs))  # max-pooling over chunks

    def evaluate_on_dataframe(self, df: pd.DataFrame, text_col: str = "text", label_col: str = "label"):
        """
        Email-level evaluation on a held-out dataframe.
        Uses 0.5 threshold over predict_proba_email().
        Prints a sklearn classification report.
        """
        assert self.tokenizer is not None and self.model is not None, "Call load() or train_from_csv() first."

        y_true = df[label_col].astype(int).tolist()
        y_pred = []

        print(f"[Encoder] Evaluating on {len(df)} emails...")
        for _, row in df.iterrows():
            p = self.predict_proba_email(str(row[text_col]))
            y_pred.append(1 if p >= 0.5 else 0)

        report = classification_report(y_true, y_pred, digits=4)
        print(report)
        return report
