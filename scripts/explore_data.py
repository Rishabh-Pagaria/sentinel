import os
import re
import json
from typing import Dict

import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split


def load_phishing_data() -> pd.DataFrame:
    """
    Load phishing dataset from Hugging Face and return as DataFrame.
    Uses 'zefang-liu/phishing-email-dataset' with 'Email Text' and 'Email Type'.
    """
    ds = load_dataset("zefang-liu/phishing-email-dataset")
    train_data = ds["train"]

    df = pd.DataFrame({
        "text": train_data["Email Text"],
        "label": [1 if x == "Phishing Email" else 0 for x in train_data["Email Type"]],
    })

    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def analyze_dataset(df: pd.DataFrame) -> Dict:
    return {
        "total_samples": len(df),
        "phishing_emails": int(df["label"].sum()),
        "safe_emails": int(len(df) - df["label"].sum()),
        "avg_text_length": df["text"].str.len().mean(),
        "min_text_length": df["text"].str.len().min(),
        "max_text_length": df["text"].str.len().max(),
    }


def visualize_data(df: pd.DataFrame, output_dir: str = "data/visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    tmp = df.copy()
    tmp["text_length"] = tmp["text"].str.len()

    plt.figure(figsize=(8, 6))
    sns.countplot(data=tmp, x="label")
    plt.title("Distribution of Phishing vs Safe Emails")
    plt.xlabel("Label (0=Safe, 1=Phishing)")
    plt.savefig(f"{output_dir}/label_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(data=tmp, x="text_length", hue="label", bins=50)
    plt.title("Distribution of Email Lengths")
    plt.xlabel("Text Length (characters)")
    plt.savefig(f"{output_dir}/text_length_distribution.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=tmp, x="label", y="text_length")
    plt.title("Email Length by Class")
    plt.xlabel("Label (0=Safe, 1=Phishing)")
    plt.ylabel("Text Length (characters)")
    plt.savefig(f"{output_dir}/text_length_boxplot.png")
    plt.close()


def prepare_data_for_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for classification:
    - Remove HTML and normalize whitespace into 'cleaned_text'
    - Keep labels as 0/1
    """
    def clean_text(text: str) -> str:
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df = df.copy()
    df["cleaned_text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].astype(int)
    return df


def create_jsonl_splits_for_slm(
    df: pd.DataFrame,
    out_dir: str = "data",
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    Create train/eval/test splits and write JSONL files for SLM fine-tuning.
    Only JSONL is written; CSV remains a single processed file.
    """
    assert abs(train_ratio + eval_ratio + test_ratio - 1.0) < 1e-6

    os.makedirs(out_dir, exist_ok=True)
    base_df = df[["cleaned_text", "label"]].dropna().reset_index(drop=True)

    temp_ratio = eval_ratio + test_ratio
    train_df, temp_df = train_test_split(
        base_df,
        test_size=temp_ratio,
        stratify=base_df["label"],
        random_state=42,
    )
    relative_eval = eval_ratio / temp_ratio
    eval_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - relative_eval),
        stratify=temp_df["label"],
        random_state=42,
    )

    def write_jsonl(split_df: pd.DataFrame, path: str):
        label_map = {1: "phishing", 0: "safe"}
        with open(path, "w", encoding="utf-8") as f:
            for _, row in split_df.iterrows():
                record = {
                    "instruction": (
                        "You are a security model. "
                        "Classify the following email as 'phishing' or 'safe'. "
                        "Reply with exactly one word: phishing or safe."
                    ),
                    "input": row["cleaned_text"],
                    "output": label_map[int(row["label"])],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_jsonl(train_df, os.path.join(out_dir, "train_gemma.jsonl"))
    write_jsonl(eval_df, os.path.join(out_dir, "eval_gemma.jsonl"))
    write_jsonl(test_df, os.path.join(out_dir, "test_gemma.jsonl"))


def main():
    os.makedirs("data", exist_ok=True)

    df = load_phishing_data()
    stats = analyze_dataset(df)
    visualize_data(df)

    df_prepared = prepare_data_for_classification(df)
    df_prepared.to_csv("data/processed_phishing_data.csv", index=False)

    df_prepared.sample(n=min(100, len(df_prepared))).to_csv(
        "data/sample_phishing_data.csv", index=False
    )

    create_jsonl_splits_for_slm(df_prepared, out_dir="data")


if __name__ == "__main__":
    main()
