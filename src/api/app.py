#!/usr/bin/env python3
"""
FastAPI server for phishing email detection using Gemini 2.5 Flash.
Provides a /classify endpoint that accepts email text and returns structured analysis.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import google.generativeai as genai
import os
import sys
from pathlib import Path
import json
import re
import html
from dotenv import load_dotenv

# Regular expressions for cleaning (from prep_phish_jsonl)
SCRIPT_RE = re.compile(r"(?is)<script.*?>.*?</script>")
STYLE_RE = re.compile(r"(?is)<style.*?>.*?</style>")
TAG_RE = re.compile(r"(?s)<[^>]+>")
WS_RE = re.compile(r"\s+")

# Add project root to PYTHONPATH
project_root = str(Path(__file__).parents[2])  # Go up 2 levels from src/api/
sys.path.append(project_root)
from src.data.prep_phish_jsonl import clean_text, find_urls, find_phrases, infer_tactics


def sanitize_prompt_input(text: str) -> str:
    """
    Sanitize input text to prevent prompt injection attacks.
    
    Techniques used:
    1. Clean HTML/scripts using techniques from prep_phish_jsonl
    2. Remove common prompt injection markers
    3. Strip system commands or role-playing attempts
    4. Remove attempts to override previous instructions
    5. Escape special characters that might interfere with prompt structure
    """
    if not text:
        return ""
        
    # First apply cleaning from prep_phish_jsonl
    # Unescape HTML entities
    text = html.unescape(str(text))
    
    # Remove scripts, styles, and HTML tags
    text = SCRIPT_RE.sub(" ", text)
    text = STYLE_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    
    # Remove potential system command or role override attempts
    blacklist = [
        # LLM control attempts
        "system:", "assistant:", "user:", "human:", 
        "instructions:", "prompt:", "context:", 
        "<system>", "</system>", 
        "<instructions>", "</instructions>",
        "<prompt>", "</prompt>",
        
        # Role override attempts
        "you are now", "ignore previous", "disregard",
        "act as", "you must", "you should",
        "forget", "do not consider", "instead of",
        
        # JSON injection attempts
        '"label":', '"confidence":', '"tactics":',
        '"evidence":', '"user_tip":', '"output":',
        
        # Command injection
        "```python", "```bash", "```shell",
        "import ", "print(", "exec(", "eval(",
    ]
    
    # Case-insensitive replacement of blacklisted terms
    text_lower = text.lower()
    for term in blacklist:
        index = text_lower.find(term)
        while index != -1:
            # Replace in original text while preserving case
            text = text[:index] + " " + text[index + len(term):]
            text_lower = text.lower()
            index = text_lower.find(term)
    
    # Remove markdown code blocks that might contain instructions
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`.*?`', '', text)
    
    # Remove excessive whitespace and normalize using prep_phish_jsonl's approach
    text = WS_RE.sub(" ", text).strip()
    
    # Additional safety: ensure no double-encoding attempts
    text = re.sub(r'\\[nrt"]', ' ', text)  # Remove common escape sequences
    
    # Escape special characters that could interfere with JSON
    text = json.dumps(text)[1:-1]  # Use json.dumps but remove outer quotes
    
    return text


load_dotenv()  # Load environment variables from .env file
# Initialize FastAPI app
app = FastAPI(
    title="Phishing Email Detector",
    description="API for detecting phishing emails using Gemini 2.5 Flash",
    version="1.0.0"
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

# Initialize Gemini

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=GEMINI_API_KEY)
# Using Gemini Flash 2.5 Lite for faster inference
model = genai.GenerativeModel(model_name='models/gemini-2.5-flash-lite',
                            generation_config={
                                'temperature': 0.1,  # Low temperature for more focused outputs
                                'top_p': 0.9,
                                'top_k': 32,
                                'max_output_tokens': 1024  # Limit output size for faster responses
                            })

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
        text = sanitize_prompt_input(request.text)
        subject = sanitize_prompt_input(request.subject) if request.subject else ""
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
            # Look for JSON between the first { and last }
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start != -1 and end != -1:
                response_text = response_text[start:end+1]
        
        # 2. Clean up common JSON formatting issues
        response_text = response_text.replace('\n', ' ')  # Remove newlines
        response_text = re.sub(r',(\s*[}\]])', r'\1', response_text)  # Remove trailing commas
        response_text = re.sub(r'\\([^"])', r'\1', response_text)  # Remove unnecessary escapes
        
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

