#!/usr/bin/env python3
"""
FastAPI server for phishing email detection using Gemini 2.5 Flash.
Provides a /classify endpoint that accepts email text and returns structured analysis.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import google.generativeai as genai
import os
import json
from prep_phish_jsonl import clean_text, find_urls, find_phrases, infer_tactics
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
# Initialize FastAPI app
app = FastAPI(
    title="Phishing Email Detector",
    description="API for detecting phishing emails using local models (DeBERTa & Gemma 2-2B)",
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

# TODO: Initialize local models (DeBERTa & Gemma 2-2B)
# This section will be replaced with local model loading
# Currently using Gemini as placeholder until local models are integrated

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Using Gemini Flash 2.5 Lite as temporary placeholder
    model = genai.GenerativeModel(model_name='models/gemini-2.5-flash-lite',
                                generation_config={
                                    'temperature': 0.1,
                                    'top_p': 0.9,
                                    'top_k': 32,
                                    'max_output_tokens': 1024
                                })
else:
    model = None  # Will use local models when integrated

# Response schema for Gemini
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
        "user_tip": {"type": "string"}
    },
    "required": ["label", "confidence", "tactics", "evidence", "user_tip"]
}

SYSTEM_PROMPT = """You are a phishing detection expert. Analyze emails and output structured JSON only.

Format:
{
  "label": "phish" or "benign",
  "confidence": 0.0-1.0,
  "tactics": ["urgency", "credentials", "links"],
  "evidence": [{"span": "text", "reason": "why suspicious"}],
  "user_tip": "security advice"
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
    Analyze an email for phishing attempts and return structured analysis
    """
    try:
        # Clean and preprocess the text
        text = request.text
        subject = request.subject or ""
        cleaned_text = clean_text(text)
        
        # Get initial heuristic analysis
        tactics, evidence = infer_tactics(cleaned_text, subject)
        urls = find_urls(cleaned_text)
        phrases = find_phrases(cleaned_text)
        
        # Prepare prompt for Gemini
        email_prompt = f"""Email to analyze:
Subject: {subject}
Text: {cleaned_text}
URLs found: {', '.join(urls) if urls else 'none'}
Suspicious phrases: {', '.join(phrases) if phrases else 'none'}"""

        # Get Gemini's analysis
        response = model.generate_content(
            contents=[SYSTEM_PROMPT, email_prompt],
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                }
            ]
        )
        
        # Parse the response
        response_text = response.text.strip()
        
        # 1. Try to find JSON block if response contains other text
        if not response_text.startswith("{"):
            import re
            # Look for JSON between curly braces, including nested structures
            json_match = re.search(r'\{(?:[^{}]|(?R))*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
        
        # 2. Clean up common JSON formatting issues
        response_text = response_text.replace('\n', ' ')  # Remove newlines
        response_text = re.sub(r',\s*}', '}', response_text)  # Remove trailing commas
        response_text = re.sub(r',\s*]', ']', response_text)  # Remove trailing commas in arrays
        
        try:
            # 3. Try to parse the JSON
            result = json.loads(response_text)
            
            # 4. Ensure all required fields are present
            required_fields = ["label", "confidence", "tactics", "evidence", "user_tip"]
            missing_fields = [field for field in required_fields if field not in result]
            
            if missing_fields:
                # Provide default values for missing fields
                if "label" not in result:
                    result["label"] = "benign"
                if "confidence" not in result:
                    result["confidence"] = 0.5
                if "tactics" not in result:
                    result["tactics"] = []
                if "evidence" not in result:
                    result["evidence"] = []
                if "user_tip" not in result:
                    result["user_tip"] = "Analysis incomplete. Please review carefully."
            
            return ClassificationResponse(**result)
            
        except json.JSONDecodeError as e:
            # If JSON parsing fails, return a safe fallback response
            return ClassificationResponse(
                label="benign",
                confidence=0.5,
                tactics=[],
                evidence=[],
                user_tip="Unable to complete analysis. Please review manually."
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

