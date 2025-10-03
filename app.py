#!/usr/bin/env python3
# Step 1: load phishing dataset(s) and save a single CSV

import pandas as pd
from datasets import load_dataset

# ---- pick a dataset (good starter) ----
HF_DATASET = "zefang-liu/phishing-email-dataset"

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

print(f"[INFO] Loading Hugging Face dataset: {HF_DATASET}")
ds = load_dataset(HF_DATASET)  # will fetch train/test splits

# merge all available splits into one dataframe
frames = []
for split in ds.keys():
    S = ds[split]
    frames.append(pd.DataFrame({
        "subject": [ex.get("subject","") for ex in S],
        "body":    [ex.get("body","")    for ex in S],
        "label":   [norm_label(ex.get("label",0)) for ex in S],
        "source_split": split
    }))
df = pd.concat(frames, ignore_index=True)

# basic cleaning: ensure strings, drop empty rows (both subject and body empty)
df["subject"] = df["subject"].fillna("").astype(str)
df["body"]    = df["body"].fillna("").astype(str)
df = df[ (df["subject"] != "") | (df["body"] != "") ].reset_index(drop=True)

# quick sanity prints
print("[INFO] Rows:", len(df))
print("[INFO] Label counts (0=benign, 1=phish):")
print(df["label"].value_counts(dropna=False))
print("[INFO] Sample rows:")
print(df.sample(5, random_state=42)[["subject","label"]])

# save CSV
out_csv = "phish_all.csv"
df[["subject","body","label"]].to_csv(out_csv, index=False)
print(f"[OK] Wrote {out_csv}")
