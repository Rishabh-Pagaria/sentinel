#!/usr/bin/env python3
"""
generate_adversarial.py

Generate adversarial email examples from a seed file using multiple techniques:
 - synonym replacement (WordNet)
 - optional back-translation (GoogleTranslator via deep_translator)
 - character-level homoglyphs
 - structure/hardening transforms

This is intended to create a held-out adversarial evaluation set for zero-shot evaluation.
"""
import argparse
import asyncio
import json
import os
import random
import re
from datetime import datetime
from typing import List, Dict

import nltk
from nltk.corpus import wordnet
from tqdm import tqdm

# Optional: deep_translator used for back-translation. You can disable with --no-backtranslate
try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except Exception:
    DEEP_TRANSLATOR_AVAILABLE = False

# Ensure nltk resources
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Homoglyph mappings for character-level attacks (sample)
HOMOGLYPHS = {
    'a': ['а', 'α', '@'],
    'e': ['е', 'э', 'ε'],
    'i': ['і', 'ı'],
    'o': ['о', 'ο', '0'],
    's': ['ѕ', '$'],
    'l': ['ӏ', '1'],
    'r': ['г', 'ř'],
    'n': ['п', 'ń'],
}

# back-translation candidate languages (ISO codes for deep_translator)
DEFAULT_LANGS = ['fr', 'de', 'es', 'ru']

# -------------------------
# Utilities
# -------------------------
def normalize_label(label):
    """Normalize label to 'phish' or 'benign' if possible."""
    if label is None:
        return None
    if isinstance(label, (int, float)):
        return 'phish' if int(label) == 1 else 'benign'
    s = str(label).strip().lower()
    if s in ('phish', 'phishing', '1', 'true', 't'):
        return 'phish'
    if s in ('benign', 'ham', '0', 'false', 'f'):
        return 'benign'
    return s  # fallback (may be 'phish'/'benign' already)

def safe_get_text(example: Dict):
    """Return the textual content from example, handling common key names."""
    # Handle your specific format
    if 'input' in example:
        text = example['input']
        # Remove 'EMAIL: ' prefix if present
        if text.startswith('EMAIL: '):
            text = text[7:]
        return text
        
    # Fallback for other formats
    for k in ('text', 'body', 'body_text', 'email'):
        if k in example and example[k]:
            return example[k]
    # fallback: try to combine subject + other fields
    subj = example.get('subject', '')
    maybe = example.get('content', '')
    return (subj + "\n" + maybe).strip()

# -------------------------
# Synonym replacement
# -------------------------
def get_synonyms(word: str) -> List[str]:
    # basic guard
    w = word.strip().lower()
    if not w.isalpha() or len(w) < 3:
        return []
    # POS tagging (nltk)
    try:
        pos_tag = nltk.pos_tag([w])[0][1]
        wn_pos = {
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV,
            'J': wordnet.ADJ
        }.get(pos_tag[0], wordnet.NOUN)
    except Exception:
        wn_pos = wordnet.NOUN

    synonyms = set()
    try:
        for syn in wordnet.synsets(w, pos=wn_pos):
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ')
                if name and name != w:
                    # avoid unnaturally long lemmas
                    if len(name.split()) <= 2:
                        synonyms.add(name)
    except Exception:
        pass
    return list(synonyms)

WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)

def replace_with_synonyms(text: str, replace_prob: float = 0.25) -> str:
    # Tokenize with word boundaries to preserve punctuation
    def repl(match):
        token = match.group(0)
        if random.random() < replace_prob:
            syns = get_synonyms(token)
            if syns:
                return random.choice(syns)
        return token
    return WORD_RE.sub(repl, text)

# -------------------------
# Back-translation (optional)
# -------------------------
def back_translate(text: str, langs: List[str] = DEFAULT_LANGS) -> str:
    if not DEEP_TRANSLATOR_AVAILABLE:
        return text
    current = text
    try:
        path = random.sample(langs, min(len(langs), random.randint(1, 2)))
        for lang in path:
            # translate to lang then back
            t1 = GoogleTranslator(source='auto', target=lang)
            mid = t1.translate(current)
            t2 = GoogleTranslator(source=lang, target='en')
            current = t2.translate(mid)
        return current
    except Exception:
        return text

# -------------------------
# Homoglyph substitutions
# -------------------------
def apply_homoglyphs(text: str, replace_prob: float = 0.2) -> str:
    out_chars = []
    for ch in text:
        low = ch.lower()
        if low in HOMOGLYPHS and random.random() < replace_prob:
            out_chars.append(random.choice(HOMOGLYPHS[low]))
        else:
            out_chars.append(ch)
    return "".join(out_chars)

# -------------------------
# Structure modifications
# -------------------------
def modify_email_structure(email: str) -> str:
    lines = email.splitlines()
    new_lines = []
    for line in lines:
        if not line.strip():
            new_lines.append(line)
            continue
        r = random.random()
        if r < 0.25:
            new_lines.append("  " + line + "  ")
        elif r < 0.45:
            new_lines.append(line + "\u200b")  # zero width space
        elif r < 0.65:
            new_lines.append(f"<div>{line}</div>")
        elif r < 0.80:
            new_lines.append(f"X-Custom-Header: {line}")
        else:
            new_lines.append(line)
    return "\n".join(new_lines)

# -------------------------
# Generators and pipeline
# -------------------------
def generate_adversarial_example(seed: Dict, enable_backtranslate: bool = True) -> Dict:
    raw_text = safe_get_text(seed)
    base_label = normalize_label(seed.get('true_label', seed.get('label')))

    # choose 2 distinct techniques
    techniques = ['synonym', 'backtranslate', 'homoglyph', 'structure']
    chosen = random.sample(techniques, k=2)

    text = raw_text
    tactics = []
    for t in chosen:
        if t == 'synonym':
            text = replace_with_synonyms(text, replace_prob=0.25)
            tactics.append('language_manipulation')
        elif t == 'backtranslate' and enable_backtranslate:
            text = back_translate(text)
            tactics.append('content_manipulation')
        elif t == 'homoglyph':
            text = apply_homoglyphs(text, replace_prob=0.18)
            tactics.append('obfuscation')
        elif t == 'structure':
            text = modify_email_structure(text)
            tactics.append('obfuscation')

    # preserve original tactic tags (if provided)
    orig_tactics = seed.get('tactics', []) or []
    for ot in orig_tactics:
        if ot not in tactics:
            tactics.append(ot)

    return {
        'text': text,
        'true_label': base_label,
        'pred_label': None,
        'confidence': 0.0,
        'tactics': tactics,
        'created_from': seed.get('seed_id', None),
        'created_at': datetime.utcnow().isoformat()
    }

# -------------------------
# Dedupe helper
# -------------------------
def is_duplicate(text, seen_norms, threshold=0.95):
    # simple normalized exact-check (lower+strip). For fuzzy use rapidfuzz.
    norm = re.sub(r'\s+', ' ', text.strip().lower())
    return norm in seen_norms, norm

# -------------------------
# Main orchestrator
# -------------------------
async def classify_and_filter(seeds, min_confidence=0.8, classifier_call=None):
    """
    classifier_call: async function taking (text) and returning {'label':..., 'confidence':...}
    If None, we just return high-confidence seeds (no classifier)
    """
    selected = []
    if classifier_call is None:
        # keep all seeds if no classifier
        return seeds

    for s in tqdm(seeds, desc="Scoring seeds"):
        try:
            text = safe_get_text(s)
            res = await classifier_call(text)
            conf = float(res.get('confidence', 0.0))
            lab = normalize_label(res.get('label', None))
            s['confidence'] = conf
            s['pred_label'] = lab
            s['correct'] = (lab == normalize_label(s.get('true_label', s.get('label'))))
            if conf >= min_confidence and s['correct']:
                selected.append(s)
        except Exception:
            continue
    return selected

def build_request_for_classifier(text: str, subject: str = ""):
    """
    Build a minimal payload for EmailRequest; adapt in your integration if needed.
    """
    return {"text": text, "subject": subject}

async def main_async(args):
    # reproducibility
    random.seed(args.seed)

    # load seed file
    if not os.path.exists(args.source_file):
        raise FileNotFoundError(args.source_file)
    seeds = []
    with open(args.source_file, 'r', encoding='utf8') as f:
        for ln in f:
            if not ln.strip(): continue
            seeds.append(json.loads(ln))

    # optional classifier evaluation step (to pick high-confidence seeds)
    candidate_seeds = seeds
    if args.use_classifier:
        # import classifier
        from app import classify_email, EmailRequest  # adapt to your app API
        async def _call_classifier(text):
            # adapt EmailRequest signature if needed; here we call with text only
            try:
                res = await classify_email(EmailRequest(text=text))
            except TypeError:
                # try alternate signature
                res = await classify_email(EmailRequest(body_text=text))
            # return normalised dict
            return {"label": getattr(res, 'label', None), "confidence": getattr(res, 'confidence', 0.0)}
        candidate_seeds = await classify_and_filter(seeds, min_confidence=args.min_confidence, classifier_call=_call_classifier)
        print(f"Selected {len(candidate_seeds)} seeds after classifier filtering (min_conf={args.min_confidence})")
    else:
        print(f"Using all {len(candidate_seeds)} seeds (no classifier filtering)")

    if not candidate_seeds:
        raise SystemExit("No seeds available after filtering. Lower min-confidence or disable classifier filtering.")

    # generate adversarial examples
    adversarials = []
    seen_norms = set()
    tries = 0
    while len(adversarials) < args.num_examples and tries < args.num_examples * 10:
        tries += 1
        seed = random.choice(candidate_seeds)
        adv = generate_adversarial_example(seed, enable_backtranslate=(not args.no_backtranslate))
        is_dup, norm = is_duplicate(adv['text'], seen_norms)
        if is_dup:
            continue
        seen_norms.add(norm)
        adversarials.append(adv)

    # save to file
    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, args.output_filename)
    with open(outpath, 'w', encoding='utf8') as fo:
        for a in adversarials:
            fo.write(json.dumps(a, ensure_ascii=False) + "\n")

    print(f"Saved {len(adversarials)} adversarial examples to {outpath}")

def main():
    parser = argparse.ArgumentParser(description="Generate adversarial email examples")
    parser.add_argument("--source-file", default="out_jsonl/train.jsonl", help="Seed JSONL file with examples")
    parser.add_argument("--num-examples", type=int, default=200, help="Number of adversarial examples to generate")
    parser.add_argument("--min-confidence", type=float, default=0.8, help="If --use-classifier, require seeds with conf >= this")
    parser.add_argument("--output-dir", default="out_jsonl", help="Where to save adversarial output")
    parser.add_argument("--output-filename", default="eval_adv.jsonl", help="Output filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sample-size", type=int, default=1000, help="Max seeds to consider (randomly sampled from source file)")
    parser.add_argument("--use-classifier", action='store_true', help="Use existing classifier to filter high-confidence seeds")
    parser.add_argument("--no-backtranslate", action='store_true', help="Disable backtranslation (faster, more deterministic)")
    args = parser.parse_args()

    # run async main
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
