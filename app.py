#!/usr/bin/env python3
"""
FastAPI server for phishing email detection using NLP + SLM Fusion Framework.
Combines Deberta (NLP) for classification with Gemma (SLM) for explainability.
Provides a /classify endpoint that accepts email text and returns structured analysis with explanations.
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import sys
from pathlib import Path
import torch
import re
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)
from peft import PeftModel
import json
import logging
from prep_phish_jsonl import clean_text, find_urls, find_phrases, infer_tactics
from dotenv import load_dotenv

# Import fusion module
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from fusion import AdaptiveFusion

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch._dynamo.config.suppress_errors = True

# Initialize FastAPI app
app = FastAPI(
    title="Phishing Email Detector - PhishGuard Lite",
    description="API for detecting phishing emails using NLP + SLM Fusion Framework (Deberta + Gemma)",
    version="2.0.0"
)

# Add CORS middleware for Gmail Add-on integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event to pre-warm models
@app.on_event("startup")
async def startup_event():
    """Pre-load models on startup to avoid timeout on first request"""
    logger.info("⏳ Pre-warming models on startup...")
    try:
        # Load Deberta
        load_deberta()
        logger.info("✓ Deberta loaded")
        
        # Load Gemma (this is the slow one)
        load_gemma()
        logger.info("✓ Gemma loaded")
        
        # Initialize fusion
        global fusion
        fusion = AdaptiveFusion(deberta_weight=0.7, gemma_weight=0.3)
        logger.info("✓ Fusion layer initialized")
        
        logger.info("🚀 All models ready! Server will respond quickly to requests.")
    except Exception as e:
        logger.error(f"⚠️ Failed to pre-warm models: {e}")
        logger.warning("Models will load on first request (may cause timeout)")

# Models for request/response
class EmailRequest(BaseModel):
    text: str = Field(..., description="The email text to analyze")
    subject: Optional[str] = Field(None, description="Optional email subject")

class Evidence(BaseModel):
    span: str = Field(..., description="The text span that provides evidence")
    reason: str = Field(..., description="The reason this span is considered evidence")

class ClassificationResponse(BaseModel):
    label: str = Field(..., description="Classification result: 'phish', 'benign', or 'uncertain'")
    confidence: float = Field(..., description="Confidence score between 0 and 1", ge=0, le=1)
    tactics: List[str] = Field(default_factory=list, description="List of detected phishing tactics")
    evidence: List[str] = Field(default_factory=list, description="Evidence supporting the classification")
    user_tip: str = Field(..., description="User-friendly tip based on the analysis")
    explanation: str = Field(..., description="Detailed explanation from the model")
    deberta_score: float = Field(..., description="Deberta model confidence score")
    gemma_score: float = Field(..., description="Gemma model confidence score")
    fusion_info: Optional[Dict] = Field(None, description="Transparency info about fusion (debugging)")

# Model paths
DEBERTA_MODEL_PATH = "rishabhpagaria/deberta_phishing"
GEMMA_BASE_MODEL = "google/gemma-2-2b-it"
GEMMA_LORA_ADAPTER = "rishabhpagaria/gemma-2-2b-it_phishing"

# Global model instances (lazy loaded)
deberta_model = None
deberta_tokenizer = None
gemma_model = None
gemma_tokenizer = None
fusion_layer = None

def load_deberta():
    """Lazy load Deberta NLP model for classification"""
    global deberta_model, deberta_tokenizer
    if deberta_model is None:
        try:
            logger.info(f"Loading Deberta fine-tuned model: {DEBERTA_MODEL_PATH}")
            deberta_tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_PATH, trust_remote_code=True)
            deberta_model = AutoModelForSequenceClassification.from_pretrained(
                DEBERTA_MODEL_PATH,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            deberta_model.eval()
            
            if torch.cuda.is_available():
                deberta_model = deberta_model.to("cuda")
                logger.info("✓ DeBERTa moved to GPU")
            else:
                logger.info("✓ DeBERTa on CPU")
            
            logger.info("✓ DeBERTa fine-tuned model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load DeBERTa: {str(e)}")
            raise
    return deberta_model, deberta_tokenizer


def load_gemma():
    """Lazy load Gemma BASE model (instruction-tuned, NO LoRA fine-tuning)"""
    global gemma_model, gemma_tokenizer
    if gemma_model is None:
        try:
            logger.info(f"Loading Gemma BASE model (no fine-tuning): {GEMMA_BASE_MODEL}")
            
            # Load tokenizer from base model
            gemma_tokenizer = AutoTokenizer.from_pretrained(
                GEMMA_BASE_MODEL, 
                trust_remote_code=True
            )
            if gemma_tokenizer.pad_token is None:
                gemma_tokenizer.pad_token = gemma_tokenizer.eos_token
                gemma_tokenizer.pad_token_id = gemma_tokenizer.eos_token_id
            
            # Load base model WITHOUT LoRA adapter
            gemma_model = AutoModelForCausalLM.from_pretrained(
                GEMMA_BASE_MODEL,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # NO LoRA adapter - using base instruction-tuned model only
            gemma_model.eval()
            
            if torch.cuda.is_available():
                logger.info("✓ Gemma moved to GPU")
            else:
                logger.info("✓ Gemma on CPU")
            
            logger.info("✓ Gemma 2-2B-IT BASE model loaded successfully (no fine-tuning)")
        except Exception as e:
            logger.error(f"Failed to load Gemma: {str(e)}")
            raise
    return gemma_model, gemma_tokenizer


def load_fusion():
    """Initialize fusion layer"""
    global fusion_layer
    if fusion_layer is None:
        logger.info("Initializing adaptive fusion layer")
        fusion_layer = AdaptiveFusion(
            deberta_base_weight=0.7,  # Deberta has 97.91% accuracy
            gemma_base_weight=0.3
        )
        logger.info("✓ Fusion layer initialized")
    return fusion_layer

# Make EmailRequest available for import
__all__ = ['classify_email', 'EmailRequest']


def get_deberta_prediction(text: str) -> float:
    """Get phishing probability from Deberta NLP model"""
    model, tokenizer = load_deberta()
    
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt"
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        phishing_prob = probs[0][1].item()  # Index 1 = phishing class
    
    return phishing_prob


def get_gemma_analysis(text: str, subject: str) -> Dict:
    """Get concise explanation from Gemma base model"""
    model, tokenizer = load_gemma()
    
    # Concise prompt asking for specific red flags
    messages = [
        {
            "role": "user",
            "content": f"""Analyze this email for phishing. Give me 2-3 key findings only.

EMAIL:
Subject: {subject}
{text[:1000]}

Respond with:
1. Classification (phishing/legitimate)
2. 2-3 specific red flags (URLs, phrases, or tactics) - be brief and concrete

Keep it under 3 sentences total."""
        }
    ]
    
    max_retries = 2
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Calling Gemma for analysis (attempt {attempt + 1}/{max_retries})...")
            
            # Use chat template to properly format the prompt
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                logger.info(f"Using chat template for prompt formatting")
            except:
                # Fallback if chat template not available
                prompt = messages[0]["content"]
                logger.info(f"Chat template not available, using raw prompt")
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            # Move to same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            logger.info(f"Input tokens: {inputs['input_ids'].shape[1]}")
            
            # Generate with parameters tuned for concise, specific output
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=200,  # Enough for specific analysis
                    temperature=0.7,  # Balanced for specific details
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.15,  # Prevent repetitive output
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode only the new tokens
            input_len = inputs["input_ids"].shape[1]
            response_ids = output_ids[0][input_len:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            
            logger.info(f"Raw response length: {len(response_text)} chars")
            logger.info(f"Full raw response:\n{response_text}\n")
            
            # Validate response
            if not response_text or len(response_text) < 3:
                logger.warning(f"Response too short")
                last_error = "Response too short"
                continue
            
            # Check for training data artifacts (model, gary, signatures, etc)
            if any(word in response_text.lower() for word in ['model', 'gary', '--', 'murphy', 'signature']):
                logger.warning(f"Response contains training data artifacts")
                last_error = "Training data echo"
                continue
            
            # Parse the response - expect 2 lines: classification + reason
            lines = [line.strip() for line in response_text.split('\n') if line.strip()]
            
            logger.info(f"Parsed lines: {lines}")
            
            if len(lines) < 1:
                logger.warning("Response has no valid lines")
                last_error = "No valid lines"
                continue
            
            # First line should be classification
            first_line = lines[0].lower()
            
            if "phishing" in first_line or "phish" in first_line:
                label = "phish"
                confidence = 0.80
            elif "legitimate" in first_line or "safe" in first_line or "benign" in first_line:
                label = "benign"
                confidence = 0.80
            else:
                logger.warning(f"Cannot parse classification: {first_line}")
                last_error = "Invalid classification"
                continue
            
            logger.info(f"✓ Gemma result: label={label}, conf={confidence:.3f}")
            
            return {
                'probability': confidence if label == 'phish' else (1.0 - confidence),
                'tactics': [],
                'evidence': [],
                'user_tip': f'Email assessment: {label.upper()}',
                'explanation': response_text[:200],
                'label': label,
                'confidence': confidence,
                'gemma_response': response_text
            }
            
        except Exception as e:
            last_error = f"Error on attempt {attempt + 1}: {type(e).__name__}: {str(e)}"
            logger.error(last_error, exc_info=True)
            continue
    
    # If we got here, Gemma failed multiple times
    logger.error(f"Gemma failed after {max_retries} attempts. Last error: {last_error}")
    raise Exception(f"Gemma analysis failed: {last_error}")


def get_user_tip_for_tactic(tactic: str) -> str:
    """Get a user-friendly tip for a specific tactic"""
    tips = {
        'urgency_framing': '⚠️ This email uses urgency tactics. Legitimate organizations rarely demand immediate action. Verify by contacting them directly.',
        'credential_harvest': '🔒 This email requests account verification. Never share credentials via email links. Contact support directly.',
        'domain_mismatch': '🔗 The email contains suspicious links. Hover over links to verify they match the claimed sender.',
        'financial_lure': '💰 This email offers financial incentives. These "too good to be true" offers are classic phishing bait. Never click links in such emails.',
        'authority_impersonation': '🏢 This email impersonates a trusted authority. Always verify by contacting the organization directly through official channels.',
    }
    return tips.get(tactic, '✓ Review email carefully for any unusual requests or offers.')

def build_educational_explanation(tactics: List[str], text: str, subject: str) -> str:
    """Build an educational explanation for detected phishing tactics"""
    explanations = {
        'urgency_framing': 'The email uses urgency language like "urgent", "immediate action", or "account suspended" to pressure you into responding quickly without thinking. Phishers rely on panic to bypass careful reasoning.',
        'credential_harvest': 'This email requests you to verify, confirm, or update account credentials. Legitimate companies never ask for passwords or sensitive info via email. This is a common phishing tactic.',
        'domain_mismatch': 'The email contains links that may not match the claimed sender. Phishers use lookalike domains or shortened URLs to hide their true destination. Always verify URLs before clicking.',
        'financial_lure': 'This email offers financial incentives like prizes, refunds, or money. These "too good to be true" offers are classic phishing bait to trick you into clicking malicious links.',
        'authority_impersonation': 'The email impersonates a trusted authority like a bank, government agency, or popular service. Phishers use brand names to gain your trust and bypass skepticism.',
    }
    
    main_explanations = [explanations.get(t, f'Detected: {t}') for t in tactics[:2]]
    combined = ' '.join(main_explanations)
    
    if len(combined) > 250:
        combined = combined[:250] + '...'
    
    return combined

def build_safe_email_explanation(text: str, subject: str) -> str:
    """Build explanation for emails that appear safe"""
    # Analyze why it's safe
    reasons = []
    
    text_lower = text.lower()
    subject_lower = subject.lower() if subject else ""
    
    # Check for professional/business tone
    if any(word in text_lower for word in ['regards', 'sincerely', 'best', 'thanks', 'professional']):
        reasons.append('professional tone and format')
    
    # Check for lack of urgency
    if not any(word in text_lower for word in ['urgent', 'immediate', 'now', 'asap', 'suspended']):
        reasons.append('no artificial urgency')
    
    # Check for legitimate domain/organization references
    if any(word in text_lower for word in ['department', 'office', 'team', 'meeting', 'project']):
        reasons.append('business content referencing internal operations')
    
    # Check for no credential requests
    if not any(word in text_lower for word in ['password', 'verify account', 'confirm identity', 'login']):
        reasons.append('no requests for sensitive credentials')
    
    # Default explanation
    if not reasons:
        return "This email does not contain typical phishing indicators. The content appears to be legitimate business communication with no requests for sensitive information or suspicious links."
    
    reason_text = ', '.join(reasons)
    return f"This email appears safe based on: {reason_text}. It contains legitimate business content without suspicious requests or urgency tactics typical of phishing emails."

@app.post("/classify", response_model=ClassificationResponse)
async def classify_email(request: EmailRequest):
    """
    Analyze an email for phishing using NLP + SLM Fusion Framework.
    
    Process:
    1. Get prediction from Deberta (NLP) - fast, accurate classification
    2. Get analysis from Gemma (SLM) - contextual reasoning and explanation
    3. Fuse predictions using adaptive confidence-based weighting
    4. Return combined result with transparency
    """
    
    try:
        # Clean and preprocess
        text = request.text
        subject = request.subject or ""
        cleaned_text = clean_text(text)
        
        logger.info(f"Analyzing email: {subject[:50]}...")
        
        # Step 1: Get Deberta prediction (NLP - linguistic transparency)
        logger.info("Getting Deberta classification...")
        deberta_prob = get_deberta_prediction(cleaned_text)
        logger.info(f"Deberta probability: {deberta_prob:.3f}")
        
        # Step 2: Get Gemma analysis (SLM - contextual reasoning)
        logger.info("Getting Gemma explanation...")
        gemma_analysis = get_gemma_analysis(cleaned_text, subject)
        logger.info(f"Gemma probability: {gemma_analysis['probability']:.3f}")
        
        # Step 3: Fuse predictions using adaptive weighting
        fusion = load_fusion()
        fused_result = fusion.fuse_with_explanation(
            deberta_prob=deberta_prob,
            gemma_prob=gemma_analysis['probability'],
            gemma_explanation=gemma_analysis
        )
        
        logger.info(f"Fused result: {fused_result['label']} "
                   f"(probability: {fused_result['probability']:.3f}, "
                   f"confidence: {fused_result['confidence']:.3f})")
        
        # Step 4: Format response
        response = ClassificationResponse(
            label="phish" if fused_result['label'] == 'phishing' else ("uncertain" if fused_result['label'] == 'uncertain' else "benign"),
            confidence=fused_result['confidence'],  # Use actual confidence, not probability!
            tactics=fused_result.get('tactics', []),
            evidence=fused_result.get('evidence', []),
            user_tip=fused_result.get('user_tip', 'Review email carefully.'),
            explanation=fused_result.get('explanation', ''),
            deberta_score=deberta_prob,  # Individual model score
            gemma_score=gemma_analysis['probability'],  # Individual model score
            fusion_info=fused_result.get('transparency', {})
        )
        
        logger.info(f"Response label: {response.label}, human_intervention: {fused_result.get('human_intervention_needed', False)}")
        return response
    except Exception as e:
        logger.error(f"Classification error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")




@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "ok",
        "service": "PhishGuard Lite",
        "model": "Deberta + Gemma Fusion",
        "version": "2.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 80)
    logger.info("STARTING PHISHGUARD LITE - NLP + SLM FUSION FRAMEWORK")
    logger.info("=" * 80)
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        timeout_keep_alive=120,  # Keep connections alive for 2 minutes
        timeout_graceful_shutdown=30
    )

