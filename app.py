#!/usr/bin/env python3
"""
FastAPI server for phishing email detection using locally trained Gemma 2-2B-IT model with LoRA adapter.
Provides a /classify endpoint that accepts email text and returns structured analysis with explanations.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import json
import logging
from prep_phish_jsonl import clean_text, find_urls, find_phrases, infer_tactics
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Phishing Email Detector - PhishGuard Lite",
    description="API for detecting phishing emails using locally trained Gemma 2-2B-IT model",
    version="1.0.0"
)

# Add CORS middleware for Gmail Add-on integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for request/response
class EmailRequest(BaseModel):
    text: str = Field(..., description="The email text to analyze")
    subject: Optional[str] = Field(None, description="Optional email subject")

class Evidence(BaseModel):
    span: str = Field(..., description="The text span that provides evidence")
    reason: str = Field(..., description="The reason this span is considered evidence")

class ClassificationResponse(BaseModel):
    label: str = Field(..., description="Classification result: 'phish' or 'benign'")
    confidence: float = Field(..., description="Confidence score between 0 and 1", ge=0, le=1)
    tactics: List[str] = Field(default_factory=list, description="List of detected phishing tactics")
    evidence: List[Evidence] = Field(default_factory=list, description="Evidence supporting the classification")
    user_tip: str = Field(..., description="User-friendly tip based on the analysis")
    explanation: str = Field(..., description="Detailed explanation from the model")

# Initialize Gemma 2-2B-IT model with LoRA adapter
BASE_MODEL = "hf_models/gemma-2-2b-it"
LORA_ADAPTER_PATH = "artifacts/models/gemma-2-2b-phishing/new-checkpoint/checkpoint-1395"
llm_pipe = None

def load_model():
    """Lazy load the model with LoRA adapter on first use"""
    global llm_pipe
    if llm_pipe is None:
        try:
            logger.info(f"Loading base model: {BASE_MODEL}")
            # Load base model WITHOUT device_map (IMPORTANT: apply LoRA first, then move to device)
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            
            logger.info(f"Loading LoRA adapter from: {LORA_ADAPTER_PATH}")
            # Load and merge LoRA adapter BEFORE moving to device
            model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
            
            # Now move to appropriate device
            if torch.cuda.is_available():
                model = model.to("cuda")
                logger.info("✓ Model moved to GPU")
            else:
                logger.info("✓ Model on CPU")
            
            logger.info("✓ Gemma 2-2B-IT model with LoRA adapter loaded successfully")
            
            # Create pipeline with the loaded model
            llm_pipe = pipeline(
                task="text-generation",
                model=model,
                tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL)
            )
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    return llm_pipe

# Response schema for Gemma
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["phish", "benign"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "tactics": {"type": "array", "items": {"type": "string"}},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "span": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["span", "reason"]
            }
        },
        "user_tip": {"type": "string"},
        "explanation": {"type": "string"}
    },
    "required": ["label", "confidence", "tactics", "evidence", "user_tip", "explanation"]
}

SYSTEM_PROMPT = """You are a phishing detection expert. Analyze emails and output structured JSON only.

Format:
{
  "label": "phish" or "benign",
  "confidence": 0.0-1.0,
  "tactics": ["urgency", "credentials", "links"],
  "evidence": [{"span": "text", "reason": "why suspicious"}],
  "user_tip": "security advice",
  "explanation": "Detailed explanation of the classification decision"
}

Focus on:
- Urgency/threats
- Credential requests
- Suspicious links
- Social engineering
- Poor grammar

Be concise. Return only valid JSON."""

# Make EmailRequest available for import
__all__ = ['classify_email', 'EmailRequest']

@app.post("/classify", response_model=ClassificationResponse)
async def classify_email(request: EmailRequest):
    """
    Analyze an email for phishing attempts using Gemma 2-2B-IT model
    """
    try:
        # Load model on first use
        model = load_model()
        
        # Clean and preprocess the text
        text = request.text
        subject = request.subject or ""
        cleaned_text = clean_text(text)
        
        # Get initial heuristic analysis
        tactics, evidence = infer_tactics(cleaned_text, subject)
        urls = find_urls(cleaned_text)
        phrases = find_phrases(cleaned_text)
        
        # Prepare prompt for Gemma
        email_prompt = f"""You are a phishing detection expert. Analyze the following email and provide a JSON response.

Email Subject: {subject}
Email Body: {cleaned_text}

Detected URLs: {', '.join(urls) if urls else 'none'}
Suspicious phrases: {', '.join(phrases) if phrases else 'none'}

Respond with ONLY a valid JSON object in this format:
{{
  "label": "phish" or "benign",
  "confidence": 0.0-1.0,
  "tactics": ["tactic1", "tactic2"],
  "evidence": [{{"span": "text snippet", "reason": "why suspicious"}}],
  "user_tip": "security advice",
  "explanation": "detailed explanation"
}}"""

        # Generate analysis using Gemma
        logger.info(f"Analyzing email: {subject[:50]}...")
        response = model(
            email_prompt,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            return_full_text=False,
        )
        
        response_text = response[0]["generated_text"].strip()
        logger.info(f"Model response: {response_text[:200]}...")
        
        # Extract JSON from response
        if not response_text.startswith("{"):
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
        
        # Clean up JSON
        response_text = response_text.replace('\n', ' ')
        response_text = re.sub(r',\s*}', '}', response_text)
        response_text = re.sub(r',\s*]', ']', response_text)
        
        try:
            result = json.loads(response_text)
            
            # Ensure all required fields
            required_fields = ["label", "confidence", "tactics", "evidence", "user_tip", "explanation"]
            for field in required_fields:
                if field not in result:
                    if field == "label":
                        result["label"] = "benign"
                    elif field == "confidence":
                        result["confidence"] = 0.5
                    elif field in ["tactics", "evidence"]:
                        result[field] = [] if field == "tactics" else []
                    elif field == "user_tip":
                        result["user_tip"] = "Review email carefully for suspicious content."
                    elif field == "explanation":
                        result["explanation"] = "Analysis completed."
            
            # Ensure label is lowercase
            result["label"] = result["label"].lower().strip()
            
            return ClassificationResponse(**result)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            return ClassificationResponse(
                label="benign",
                confidence=0.5,
                tactics=[],
                evidence=[],
                user_tip="Unable to complete analysis. Please review manually.",
                explanation="JSON parsing error in model response."
            )
        
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "ok",
        "service": "PhishGuard Lite",
        "model": "Gemma 2-2B-IT"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

