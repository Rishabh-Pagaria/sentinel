"""
Gemma 2-2b model initialization and helper functions for text generation.
This module provides a centralized model instance that can be imported by other modules.
"""

import torch
from transformers import pipeline
from typing import Dict

# Initialize the model pipeline (using 2b version for faster inference)
pipe = pipeline(
    task="text-generation",
    model="google/gemma-2-2b",
    dtype=torch.bfloat16,
    device_map="auto",
)

def generate_phishing_analysis(
    email_text: str,
    max_length: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:
    """
    Generate phishing classification using Gemma 2-2b model.
    
    Args:
        email_text (str): The email text to analyze
        max_length (int): Maximum length of generated response
        temperature (float): Sampling temperature (0.0-1.0)
        top_p (float): Nucleus sampling parameter
        
    Returns:
        str: Classification result ("phishing" or "safe")
    """
    # Simple prompt format matching training data
    prompt = f"""You are a security model. Classify the following email as 'phishing' or 'safe'. Reply with exactly one word: phishing or safe.

{email_text}

Classification:"""

    # Generate response
    response = pipe(
        prompt,
        max_new_tokens=10,  # Only need 1 word output
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        return_full_text=False,
    )[0]["generated_text"]
    
    # Extract classification (first word)
    result = response.strip().lower().split()[0] if response.strip() else "unknown"
    return result

# Singleton instance that can be imported by other modules
model = pipe
