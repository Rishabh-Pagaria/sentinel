#!/usr/bin/env python3
"""
Real-time email monitoring dashboard.
"""

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
import asyncio
import os
import sys
from typing import Set
from contextlib import asynccontextmanager
import socket
from pathlib import Path

from gmail_poller import GmailPoller, process_messages
from app import classify_email, EmailRequest

# Global poller instance
gmail_poller = None

def initialize_gmail_poller():
    """Initialize Gmail poller and ensure it works"""
    global gmail_poller
    
    if not os.environ.get('LOCAL_TEST'):
        print("Please set LOCAL_TEST=1 environment variable")
        return False
    
    # Set service account credentials for Google Cloud services
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vertex-key.json'
        
    try:
        print("\nInitializing Gmail Poller...")
        poller = GmailPoller('credentials.json')  # This is still needed for Gmail OAuth
        poller.authenticate()
        gmail_poller = poller
        print("Gmail Poller initialized successfully!")
        return True
    except Exception as e:
        print(f"Failed to initialize Gmail Poller: {str(e)}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup/shutdown"""
    polling_task = asyncio.create_task(poll_emails())
    yield
    polling_task.cancel()
    try:
        await polling_task
    except asyncio.CancelledError:
        pass

# Initialize FastAPI only if Gmail Poller is ready
if not initialize_gmail_poller():
    print("Server startup aborted due to Gmail Poller initialization failure")
    sys.exit(1)

app = FastAPI(
    title="Email Security Monitor",
    lifespan=lifespan
)

# Store active WebSocket connections
active_connections: Set[WebSocket] = set()

# Dashboard HTML template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Email Security Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; }
        .header { background: #fff; padding: 20px; border-radius: 8px; 
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .email-list { background: #fff; padding: 20px; border-radius: 8px;
                     box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .email-item { border-left: 4px solid #ddd; margin-bottom: 15px;
                     padding: 15px; background: #fafafa; }
        .email-item.phish { border-left-color: #ff4444; }
        .email-item.benign { border-left-color: #00C851; }
        .subject { font-weight: bold; margin-bottom: 5px; }
        .meta { font-size: 0.9em; color: #666; margin-bottom: 10px; }
        .tactics { margin-top: 10px; }
        .tactic-tag { display: inline-block; background: #e0e0e0;
                     padding: 2px 8px; border-radius: 12px;
                     font-size: 0.8em; margin-right: 5px; }
        .tip { margin-top: 10px; padding: 10px; background: #e8f5e9;
               border-radius: 4px; font-size: 0.9em; }
        .confidence { float: right; padding: 2px 8px; border-radius: 12px;
                     font-size: 0.8em; background: #e0e0e0; }
        .status { position: fixed; bottom: 20px; right: 20px;
                 padding: 10px 20px; background: #333; color: white;
                 border-radius: 20px; opacity: 0; transition: opacity 0.3s; }
        .status.active { opacity: 1; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Email Security Monitor</h1>
            <p>Real-time phishing detection powered by Gemini</p>
        </div>
        <div class="email-list" id="emailList"></div>
    </div>
    <div class="status" id="status">Connected to server</div>

    <script>
        const emailList = document.getElementById('emailList');
        const status = document.getElementById('status');
        let ws;

        function showStatus(message) {
            status.textContent = message;
            status.classList.add('active');
            setTimeout(() => status.classList.remove('active'), 3000);
        }

        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = () => {
                showStatus('Connected to server');
            };
            
            ws.onclose = () => {
                showStatus('Disconnected - reconnecting...');
                setTimeout(connectWebSocket, 1000);
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                // Create email item
                const emailDiv = document.createElement('div');
                emailDiv.className = `email-item ${data.label}`;
                
                // Add confidence score
                const confidence = Math.round(data.confidence * 100);
                const confidenceDiv = document.createElement('div');
                confidenceDiv.className = 'confidence';
                confidenceDiv.textContent = `${confidence}% ${data.label}`;
                emailDiv.appendChild(confidenceDiv);

                // Add subject and metadata
                const subject = document.createElement('div');
                subject.className = 'subject';
                subject.textContent = data.subject || '(no subject)';
                emailDiv.appendChild(subject);

                const meta = document.createElement('div');
                meta.className = 'meta';
                meta.textContent = `Date: ${data.date}`;
                emailDiv.appendChild(meta);

                // Add tactics if present
                if (data.tactics && data.tactics.length > 0) {
                    const tactics = document.createElement('div');
                    tactics.className = 'tactics';
                    data.tactics.forEach(tactic => {
                        const tag = document.createElement('span');
                        tag.className = 'tactic-tag';
                        tag.textContent = tactic;
                        tactics.appendChild(tag);
                    });
                    emailDiv.appendChild(tactics);
                }

                // Add user tip if present
                if (data.user_tip) {
                    const tip = document.createElement('div');
                    tip.className = 'tip';
                    tip.textContent = data.user_tip;
                    emailDiv.appendChild(tip);
                }

                // Insert at the top
                emailList.insertBefore(emailDiv, emailList.firstChild);
                showStatus('New email analyzed');
            };
        }

        connectWebSocket();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the monitoring dashboard"""
    return DASHBOARD_HTML

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections"""
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        active_connections.remove(websocket)

async def broadcast_email(email_data: dict):
    """Send email analysis to all connected clients"""
    disconnected = set()
    for connection in active_connections:
        try:
            await connection.send_json(email_data)
        except Exception:
            disconnected.add(connection)
    active_connections.difference_update(disconnected)

async def poll_emails():
    """Poll for new emails and analyze them"""
    while True:
        try:
            print("\nPolling for new messages...")
            messages = gmail_poller.get_new_messages()
            
            if messages:
                print(f"Found {len(messages)} new messages")
                processed = process_messages(messages, gmail_poller)
                
                for email in processed:
                    try:
                        print(f"\nAnalyzing email: {email['subject']}")
                        email_text = f"SUBJECT: {email['subject']}\n\n{email['body']}"
                        result = await classify_email(EmailRequest(text=email_text))
                        
                        broadcast_data = {
                            "subject": email["subject"],
                            "sender": email["sender"],
                            "date": email["date"],
                            "label": result.label,
                            "confidence": result.confidence,
                            "tactics": result.tactics,
                            "evidence": [e.dict() for e in result.evidence],
                            "user_tip": result.user_tip
                        }
                        print(f"Analysis complete: {result.label} (confidence: {result.confidence})")
                        
                        await broadcast_email(broadcast_data)
                    except Exception as e:
                        print(f"Error analyzing email: {str(e)}")
                        continue
            else:
                print("No new messages found")
                
        except Exception as e:
            print(f"Error during polling: {str(e)}")
        
        await asyncio.sleep(30)

def find_available_port(start_port, end_port):
    """Find first available port in range"""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return port
            except OSError:
                continue
    return None

if __name__ == "__main__":
    import uvicorn
    
    # Find available port
    port = find_available_port(8888, 8899)
    if port:
        print(f"\nStarting server on http://localhost:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        print("No available ports found in range 8888-8899")