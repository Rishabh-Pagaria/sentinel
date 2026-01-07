"""
Gemma 2-9b model initialization and helper functions for text generation.
This module provides a centralized model instance that can be imported by other modules.
"""

import torch
from transformers import pipeline
from typing import Dict

# Initialize the model pipeline
pipe = pipeline(
    task="text-generation",
    model="google/gemma-2-9b",
    dtype=torch.bfloat16,
    device_map="auto",
)

def generate_phishing_analysis(
    email_text: str,
    max_length: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> Dict:
    """
    Generate structured phishing analysis using Gemma 2-9b model.
    
    Args:
        email_text (str): The email text to analyze
        max_length (int): Maximum length of generated response
        temperature (float): Sampling temperature (0.0-1.0)
        top_p (float): Nucleus sampling parameter
        
    Returns:
        Dict: Structured analysis containing label, confidence, tactics, and evidence
    """
    # Format the prompt for phishing analysis
    prompt = f"""Analyze this email for phishing indicators:

{email_text}

Provide a structured analysis in this format:
{{"label": "phish" or "benign",
  "confidence": 0.0-1.0,
  "tactics": ["list", "of", "tactics"],
  "evidence": [
    {{"span": "suspicious text",
      "reason": "why it's suspicious"}}
  ]
}}"""

    # Generate response
    response = pipe(
        prompt,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        return_full_text=False,
    )[0]["generated_text"]
    
    # TODO: Add response parsing and validation
    # For now, return raw response
    return response

# Singleton instance that can be imported by other modules
model = pipe
