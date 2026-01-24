import os
import re
import json
from typing import Dict, List
import warnings

import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from bs4 import MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

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


def extract_html_features(html_text: str) -> Dict:
    """Extract HTML-based phishing indicators"""
    soup = BeautifulSoup(html_text, "html.parser")
    
    features = {
        "link_mismatches": [],
        "has_forms": False,
        "has_scripts": False,
        "suspicious_links": [],
    }
    
    # Extract and analyze links
    for a_tag in soup.find_all("a", href=True):
        link_text = a_tag.get_text().strip().lower()
        href = a_tag["href"].lower()
        
        # Check for link text vs URL mismatch (e.g., text says "paypal.com" but goes elsewhere)
        common_brands = ["paypal", "bank", "amazon", "microsoft", "apple", "google", "ebay"]
        for brand in common_brands:
            if brand in link_text and brand not in href:
                features["link_mismatches"].append({
                    "text": link_text,
                    "url": href,
                    "brand": brand
                })
        
        # Flag suspicious TLDs or obfuscated URLs
        suspicious_tlds = [".ru", ".tk", ".ml", ".ga", ".cf", ".cn"]
        if any(tld in href for tld in suspicious_tlds):
            features["suspicious_links"].append(href)
    
    # Check for forms (credential harvesting)
    features["has_forms"] = bool(soup.find_all("form"))
    
    # Check for scripts (potentially malicious)
    features["has_scripts"] = bool(soup.find_all("script"))
    
    return features


def detect_text_tactics(text: str) -> List[str]:
    """Detect text-based phishing tactics using keywords"""
    text_lower = text.lower()
    tactics = []
    
    # Urgency tactics
    urgency_words = ["urgent", "immediate", "suspended", "expire", "act now", "limited time"]
    if any(word in text_lower for word in urgency_words):
        tactics.append("urgency framing")
    
    # Credential requests
    cred_words = ["verify", "confirm", "password", "account", "login", "credentials"]
    if any(word in text_lower for word in cred_words):
        tactics.append("credential harvesting")
    
    # Financial lures
    financial_words = ["prize", "winner", "refund", "lottery", "claim", "$"]
    if any(word in text_lower for word in financial_words):
        tactics.append("financial lure")
    
    # Authority impersonation
    authority_words = ["irs", "tax", "government", "federal", "court", "legal"]
    if any(word in text_lower for word in authority_words):
        tactics.append("authority impersonation")
    
    return list(set(tactics))  # Remove duplicates


def prepare_data_for_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for classification:
    - Keep raw HTML in 'text' column (NO cleaning)
    - Normalize whitespace only
    - Keep labels as 0/1
    """
    df = df.copy()
    # Only normalize excessive whitespace, keep HTML tags
    df["text"] = df["text"].apply(lambda x: re.sub(r"\s+", " ", str(x)).strip())
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
    base_df = df[["text", "label"]].dropna().reset_index(drop=True)

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
        with open(path, "w", encoding="utf-8") as f:
            for _, row in split_df.iterrows():
                # Get plain text for analysis
                plain_text = BeautifulSoup(row["text"], "html.parser").get_text()
                plain_text = re.sub(r"\s+", " ", plain_text).strip()
                
                # Extract HTML features
                html_features = extract_html_features(row["text"])
                
                # Detect text-based tactics
                text_tactics = detect_text_tactics(plain_text)
                
                # Build natural language explanation
                label = "phishing" if row["label"] == 1 else "safe"
                
                if label == "phishing":
                    # Build reason and tactics for phishing
                    reasons = []
                    all_tactics = []
                    
                    # HTML-based reasons
                    if html_features["link_mismatches"]:
                        mismatch = html_features["link_mismatches"][0]
                        reasons.append(f"Link text mentions '{mismatch['brand']}' but redirects to '{mismatch['url']}'")
                        all_tactics.append("link spoofing")
                    
                    if html_features["has_forms"]:
                        reasons.append("Contains HTML forms requesting information")
                        all_tactics.append("credential harvesting")
                    
                    if html_features["suspicious_links"]:
                        reasons.append(f"Contains suspicious links to untrusted domains")
                        all_tactics.append("malicious links")
                    
                    # Text-based tactics
                    all_tactics.extend(text_tactics)
                    
                    # Generic reason if no specific features detected
                    if not reasons:
                        reasons.append("Contains typical phishing indicators")
                    
                    reason_text = ". ".join(reasons[:2]) + "."  # Limit to 2 main reasons
                    
                    # Format tactics list
                    tactics_list = list(set(all_tactics))[:4]  # Max 4 unique tactics
                    tactics_text = "\n".join([f"- {t.title()}" for t in tactics_list]) if tactics_list else "- Social engineering"
                    
                    output = f"This email is phishing.\n\nReason: {reason_text}\n\nTactics detected:\n{tactics_text}\n\nRecommendation: Do not click any links or provide personal information. Verify requests by contacting the organization directly through official channels."
                else:
                    # Safe email
                    output = "This email is safe.\n\nReason: No suspicious indicators detected. Professional tone, legitimate content, no credential requests or urgency tactics.\n\nNo phishing tactics detected.\n\nRecommendation: This appears to be legitimate communication."
                
                record = {
                    "instruction": "Analyze this email and determine if it's phishing or safe. Explain your reasoning.",
                    "input": f"EMAIL: {row['text']}",  # Keep raw HTML in input
                    "output": output,
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
