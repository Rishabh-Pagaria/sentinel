#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prep phishing dataset for Gemini fine-tuning:
- Read CSV with columns: subject, body, label
- Clean text (strip HTML/script/style, normalize whitespace)
- Map labels to {"phish","benign","uncertain"}
- (Optional) light heuristics to populate tactics/evidence
- (Optional) adversarial augmentations (paraphrase-ish swaps, URL obfuscation)
- Split into train/eval/adv and write JSONL with:
  {
    "input": "SUBJECT: ...\nEMAIL: ...",
    "output": { "label": "...", "confidence": 0.0-1.0, "tactics": [...], "evidence": [...], "user_tip": "..." }
  }
"""

import argparse, csv, html, json, os, random, re, sys
from typing import Dict, List, Tuple

random.seed(42)

# ---------------------- cleaning ----------------------
SCRIPT_RE = re.compile(r"(?is)<script.*?>.*?</script>")
STYLE_RE  = re.compile(r"(?is)<style.*?>.*?</style>")
TAG_RE    = re.compile(r"(?s)<[^>]+>")
WS_RE     = re.compile(r"\s+")

def clean_text(t: str) -> str:
    if not t: return ""
    t = html.unescape(t)
    t = SCRIPT_RE.sub(" ", t)
    t = STYLE_RE.sub(" ", t)
    t = TAG_RE.sub(" ", t)
    t = WS_RE.sub(" ", t).strip()
    return t

# ---------------------- heuristics ----------------------
URL_RE = re.compile(r"https?://[^\s)>\]]+", re.I)
SUSP_PHRASES = [
    r"\bverify (your )?account\b", r"\bverify now\b",
    r"\baccount (suspension|suspended|locked)\b",
    r"\bupdate (your )?payment\b",
    r"\burgent\b", r"\bimmediate action\b",
    r"\blogin (here|now)\b", r"\bconfirm (identity|password)\b",
]

def find_urls(t: str) -> List[str]:
    return URL_RE.findall(t or "")

def find_phrases(t: str) -> List[str]:
    low = (t or "").lower()
    out = []
    for pat in SUSP_PHRASES:
        m = re.search(pat, low)
        if m: out.append(m.group(0))
    # dedup
    seen=set(); uniq=[]
    for x in out:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq[:6]

def infer_tactics(body: str, subject: str) -> Tuple[List[str], List[Dict[str,str]]]:
    text = f"{subject}\n{body}".lower()
    tactics, evidence = [], []
    # urgency
    if any(x in text for x in ["urgent", "immediate action", "verify now", "suspend", "suspension"]):
        tactics.append("urgency_framing")
        evidence.append({"span":"urgent/verify now/suspend", "reason":"urgency_framing"})
    # credential harvest
    if any(x in text for x in ["login", "password", "verify your account", "confirm identity"]):
        tactics.append("credential_harvest")
        evidence.append({"span":"login/password/verify/confirm", "reason":"credential_harvest"})
    # domain mismatch (best-effort; we don’t have headers—use URL presence)
    if find_urls(body):
        tactics.append("domain_mismatch")
        u = find_urls(body)[0]
        evidence.append({"span":u, "reason":"domain_mismatch"})
    # dedup
    tactics = list(dict.fromkeys(tactics))
    # cap
    evidence = evidence[:3]
    return tactics, evidence

# ---------------------- adversarial augmentation ----------------------
REPL = [
    ("verify now", "confirm promptly"),
    ("account", "profile"),
    ("password", "passcode"),
    ("login", "sign in"),
    ("urgent", "time-sensitive"),
    ("click", "tap"),
]

def obfuscate_urls(text: str) -> str:
    def repl(m):
        u = m.group(0)
        return u.replace("http", "hxxp").replace(".", "[.]")
    return URL_RE.sub(repl, text)

def light_paraphrase(text: str) -> str:
    out = text
    for a,b in REPL:
        out = re.sub(rf"\b{re.escape(a)}\b", b, out, flags=re.I)
    return out

def adversarial_variants(subject: str, body: str, n: int = 1) -> List[Tuple[str,str]]:
    outs=[]
    for _ in range(n):
        b = body
        # randomly choose one or both
        if random.random() < 0.7:
            b = light_paraphrase(b)
        if random.random() < 0.7:
            b = obfuscate_urls(b)
        s = light_paraphrase(subject) if random.random()<0.5 else subject
        outs.append((s,b))
    return outs

# ---------------------- label mapping ----------------------
def map_label(raw: str, phish_values: List[str]) -> str:
    if raw is None: return "uncertain"
    r = str(raw).strip().lower()
    if r in phish_values: return "phish"
    if r in {"0","benign","ham","legit","legitimate","non-phish","nonphish"}: return "benign"
    # numeric?
    if r.isdigit():
        return "phish" if int(r)==1 else "benign"
    return "uncertain"

# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV path")
    ap.add_argument("--subject-col", default="subject", help="CSV column for subject")
    ap.add_argument("--body-col",    default="body",    help="CSV column for body")
    ap.add_argument("--label-col",   default="label",   help="CSV column for label")
    ap.add_argument("--phish-values", nargs="*", default=["1","phish","spam","phishing"], help="Values to consider as phishing")
    ap.add_argument("--out-dir", required=True, help="Output dir for JSONL files")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--eval-ratio",  type=float, default=0.2)
    ap.add_argument("--adv-count",   type=int, default=0, help="Adversarial samples per phishing email to generate into adv split")
    ap.add_argument("--balance", action="store_true", help="Balance classes in train (downsample majority)")
    ap.add_argument("--no-heuristics", action="store_true", help="If set, do not auto-fill tactics/evidence")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows=[]
    with open(args.csv, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            subj = clean_text(r.get(args.subject_col,""))
            body = clean_text(r.get(args.body_col,""))
            lab  = map_label(r.get(args.label_col,""), args.phish_values)
            if not (subj or body): 
                continue
            rows.append({"subject": subj, "body": body, "label": lab})

    # shuffle
    random.shuffle(rows)

    # split
    n = len(rows)
    n_train = int(n * args.train_ratio)
    n_eval  = int(n * args.eval_ratio)
    train = rows[:n_train]
    eval_ = rows[n_train:n_train+n_eval]
    # remainder (if any) ignored for simplicity

    # balance train (optional)
    if args.balance:
        ph = [r for r in train if r["label"]=="phish"]
        be = [r for r in train if r["label"]=="benign"]
        k = min(len(ph), len(be))
        random.shuffle(ph); random.shuffle(be)
        train = ph[:k] + be[:k]
        random.shuffle(train)

    # build JSONL
    def mk_record(subj, body, label):
        tactics, evidence = ([],[])
        if not args.no-heuristics and label=="phish":
            tactics, evidence = infer_tactics(body, subj)
        out = {
            "input": f"SUBJECT: {subj}\nEMAIL: {body}",
            "output": {
                "label": label,
                "confidence": 0.9 if label=="phish" else 0.7 if label=="benign" else 0.5,
                "tactics": tactics,
                "evidence": evidence,
                "user_tip": "Do not click links; use the official site."
            }
        }
        return out

    # write train/eval
    with open(os.path.join(args.out_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(mk_record(r["subject"], r["body"], r["label"])) + "\n")

    with open(os.path.join(args.out_dir, "eval.jsonl"), "w", encoding="utf-8") as f:
        for r in eval_:
            f.write(json.dumps(mk_record(r["subject"], r["body"], r["label"])) + "\n")

    # adversarial split (optional)
    if args.adv_count > 0:
        adv = []
        for r in eval_:
            if r["label"] != "phish": 
                continue
            for s,b in adversarial_variants(r["subject"], r["body"], n=args.adv_count):
                adv.append({"subject": s, "body": b, "label": "phish"})
        with open(os.path.join(args.out_dir, "eval_adv.jsonl"), "w", encoding="utf-8") as f:
            for r in adv:
                f.write(json.dumps(mk_record(r["subject"], r["body"], r["label"])) + "\n")

    print("Done. Wrote:", os.listdir(args.out_dir))

if __name__ == "__main__":
    main()