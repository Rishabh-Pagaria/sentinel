# Sentinel: Explainable Phishing Detection System

AI-powered phishing email detection with natural language explanations using DeBERTa (NLP) and Gemma-2-2B-IT (SLM) fusion.

---

## Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (optional but recommended)
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/Rishabh-Pagaria/sentinel.git
cd sentinel
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv my_env
.\my_env\Scripts\activate

# Linux/macOS
python3 -m venv my_env
source my_env/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Setup HuggingFace Authentication
```bash
# Create account at https://huggingface.co/join
# Generate token at https://huggingface.co/settings/tokens
huggingface-cli login
# Paste your token when prompted
```

### Step 5: Download Models
```bash
python scripts/download_models.py
```

This downloads:
- `rishabhpagaria/deberta_phishing` - Fine-tuned DeBERTa classifier
- `google/gemma-2-2b-it` - Base Gemma instruction-tuned model
- `rishabhpagaria/gemma-2-2b-it_phishing` - LoRA adapter (optional)

### Step 6: Run Backend Server
```bash
uvicorn app:app --reload
```

Server available at `http://localhost:8000`

### Step 7: (Optional) Gmail Add-on Integration
```bash
# Terminal 2: Create public tunnel
ngrok http 8000
# Copy ngrok URL and paste in gmail_addon/Code.gs line 11:
# const CONFIG = { BACKEND_URL: 'https://your-ngrok-url/classify' };

# Deploy to Gmail
cd gmail_addon
clasp push
```

---

## API Example

**Request**:
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Verify your account now or risk suspension",
    "subject": "URGENT: Account Verification Required"
  }'
```

**Response**:
```json
{
  "label": "phish",
  "confidence": 0.92,
  "tactics": ["urgency_framing", "credential_harvest"],
  "explanation": "This email is phishing. Creates artificial urgency with threat of account suspension...",
  "deberta_score": 0.95,
  "gemma_score": 0.85
}
```

---

## Models

### DeBERTa (Classification)
- **Model**: `rishabhpagaria/deberta_phishing`
- **Type**: Encoder-based sequence classification
- **Parameters**: 184M
- **Accuracy**: 97.91%
- **Speed**: ~50ms per email

### Gemma-2-2B-IT (Explanation)
- **Model**: `google/gemma-2-2b-it` (base, no fine-tuning)
- **Type**: Decoder-based language model
- **Parameters**: 2B
- **Speed**: 2-5 seconds per email

### Optional: Gemma LoRA Adapter
- **Model**: `rishabhpagaria/gemma-2-2b-it_phishing`
- **Type**: LoRA fine-tuned adapter
- **Size**: ~50MB

---

## Architecture

**Two-Model Fusion**:
1. **DeBERTa**: Fast, accurate classification (97.91% accuracy)
2. **Gemma-2-2B-IT**: Natural language explanations

**Adaptive Fusion**:
- Combines predictions using confidence-based weighting
- Detects disagreement between models
- Preserves explanations for transparency

---

## Testing

### Test Backend API
```bash
python test_inference.py
```

### Evaluate on Test Set
```bash
python scripts/evaluate_fusion.py --data data/test_gemma.jsonl
```

---

## Performance

| Model | Task | Accuracy | Precision | Recall | Speed |
|-------|------|----------|-----------|--------|-------|
| DeBERTa | Classification | 97.91% | 98.5% | 97.2% | ~50ms |
| Gemma (base) | Classification + Explanation | 85-92% | 80-90% | 75-88% | 2-5s |
| **Fusion** | **Combined Decision** | **88-94%** | **85-92%** | **82-90%** | **2-5s** |

---

## Citation

```bibtex
@misc{pagaria2026sentinel,
  title={Sentinel: Explainable Phishing Detection via NLP+SLM Fusion},
  author={Pagaria, Rishabh},
  year={2026},
  howpublished={\url{https://github.com/Rishabh-Pagaria/sentinel}},
}
```

---

## License

See LICENSE file for details.
