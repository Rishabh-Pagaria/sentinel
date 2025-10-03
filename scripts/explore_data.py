import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os

def load_phishing_data() -> pd.DataFrame:
    """
    Load phishing dataset from Hugging Face and return as DataFrame.
    Using 'zefang-liu/phishing-email-dataset' which contains both subject and body.
    """
    print("Loading dataset from Hugging Face...")
    ds = load_dataset("zefang-liu/phishing-email-dataset")
    print("Available splits:", list(ds.keys()))
    
    # Create DataFrame from train split
    train_data = ds['train']
    print("\nDataset structure:")
    print("Number of examples:", len(train_data))
    print("Features:", train_data.features)
    
    df = pd.DataFrame({
        "text": train_data["Email Text"],
        "label": [1 if x == "Phishing Email" else 0 for x in train_data["Email Type"]]
    })
    
    # Basic cleaning: ensure strings, drop empty rows
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"] != ""].reset_index(drop=True)
    
    print(f"\nProcessed {len(df)} emails ({df['label'].sum()} phishing, {len(df) - df['label'].sum()} safe)")
    return df

def norm_label(x):
    # normalize to 0=benign, 1=phish
    if x is None:
        return 0
    if isinstance(x, (int, float)):
        return 1 if int(x) == 1 else 0
    s = str(x).strip().lower()
    if s in {"1","phish","phishing","malicious","spam"}: return 1
    if s in {"0","benign","ham","legit","legitimate","non-phish","nonphish"}: return 0
    return 0
    df = pd.DataFrame(dataset)
    return df

def analyze_dataset(df: pd.DataFrame) -> Dict:
    """
    Analyze dataset and return key statistics
    """
    stats = {
        "total_samples": len(df),
        "phishing_emails": df["label"].sum(),
        "safe_emails": len(df) - df["label"].sum(),
        "avg_text_length": df["text"].str.len().mean(),
        "min_text_length": df["text"].str.len().min(),
        "max_text_length": df["text"].str.len().max(),
    }
    return stats

def visualize_data(df: pd.DataFrame, output_dir: str = "data/visualizations"):
    """
    Create basic visualizations of the dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Label distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x="label")
    plt.title("Distribution of Phishing vs Safe Emails")
    plt.xlabel("Label (0=Safe, 1=Phishing)")
    plt.savefig(f"{output_dir}/label_distribution.png")
    plt.close()
    
    # Text length distribution
    plt.figure(figsize=(10, 6))
    df["text_length"] = df["text"].str.len()
    sns.histplot(data=df, x="text_length", hue="label", bins=50)
    plt.title("Distribution of Email Lengths")
    plt.xlabel("Text Length (characters)")
    plt.savefig(f"{output_dir}/text_length_distribution.png")
    plt.close()
    
    # Text length boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="label", y="text_length")
    plt.title("Email Length by Class")
    plt.xlabel("Label (0=Safe, 1=Phishing)")
    plt.ylabel("Text Length (characters)")
    plt.savefig(f"{output_dir}/text_length_boxplot.png")
    plt.close()

def prepare_data_for_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for classification:
    1. Clean text
    2. Extract subject/body if available
    3. Format labels
    """
    from bs4 import BeautifulSoup
    import re

    def clean_text(text: str) -> str:
        # Remove HTML
        text = BeautifulSoup(text, "html.parser").get_text()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Clean text
    df["cleaned_text"] = df["text"].apply(clean_text)
    
    # Convert labels to binary
    df["label"] = df["label"].map({"phishing": 1, "ham": 0})
    
    return df

def main():
    # Create output directories
    os.makedirs("data", exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    df = load_phishing_data()
    
    # Analyze
    print("\nAnalyzing dataset...")
    stats = analyze_dataset(df)
    print("\nDataset Statistics:")
    for k, v in stats.items():
        print(f"{k}: {v}")
    
    # Visualize
    print("\nCreating visualizations...")
    visualize_data(df)
    
    # Prepare data
    print("\nPreparing data for classification...")
    df_prepared = prepare_data_for_classification(df)
    
    # Save processed data
    print("\nSaving processed data...")
    df_prepared.to_csv("data/processed_phishing_data.csv", index=False)
    
    # Save sample for quick testing
    df_prepared.sample(n=min(100, len(df_prepared))).to_csv(
        "data/sample_phishing_data.csv", index=False
    )

if __name__ == "__main__":
    main()