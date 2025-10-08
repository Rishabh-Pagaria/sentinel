#!/usr/bin/env python3
"""
Cloud Run service for Gmail polling with webhook endpoint
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from gmail_poller import GmailPoller
from datetime import datetime
import json

# Initialize FastAPI app
app = FastAPI(
    title="Gmail Poller Service",
    description="Service to poll Gmail for new emails and process them",
    version="1.0.0"
)

# Initialize Gmail Poller
poller = GmailPoller(os.getenv('GMAIL_CREDENTIALS_PATH', 'credentials.json'))

class EmailResponse(BaseModel):
    message_id: str
    thread_id: str
    subject: str
    sender: str
    date: str
    timestamp: str
    num_new_messages: int

@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/poll", response_model=List[EmailResponse])
async def poll_emails():
    """Poll for new emails and process them"""
    try:
        messages = poller.get_new_messages()
        if not messages:
            return []
            
        processed_messages = []
        for msg in messages:
            email = poller.extract_email_content(msg)
            processed_messages.append(EmailResponse(
                message_id=email['message_id'],
                thread_id=email['thread_id'],
                subject=email['subject'],
                sender=email['sender'],
                date=email['date'],
                timestamp=email['timestamp'],
                num_new_messages=len(messages)
            ))
        
        # Here you can add your email processing logic
        # For example, send to your phishing detection model
        
        return processed_messages
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))