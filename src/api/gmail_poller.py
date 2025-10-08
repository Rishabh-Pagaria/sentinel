#!/usr/bin/env python3
"""
Gmail Poller service that continuously fetches new emails using Gmail API
with history ID tracking for efficiency.
"""

import os
from typing import List, Dict, Optional
from datetime import datetime
import pickle
import base64
from email import utils
import email.header
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from google.cloud import storage

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

class GmailPoller:
    """Gmail API handler that tracks history ID for efficient polling"""
    
    def __init__(self, credentials_path: str):
        """Initialize the Gmail API client"""
        self.credentials_path = credentials_path
        self.credentials = None
        self.service = None
        self.history_id = None
        self.bucket_name = os.getenv('HISTORY_BUCKET_NAME')
        
    def authenticate(self) -> None:
        """Authenticate with Gmail API using OAuth 2.0"""
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                self.credentials = pickle.load(token)
        
        # If there are no (valid) credentials available, let the user log in
        if not self.credentials or not self.credentials.valid:
            if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                self.credentials.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES)
                # Try different ports if 8080 is in use
                for port in [8080, 8081, 8082, 8090]:
                    try:
                        self.credentials = flow.run_local_server(
                            port=port,
                            success_message='Authentication successful! You can close this window.')
                        print(f"\nUsing port {port} for authentication")
                        break
                    except OSError as e:
                        print(f"Port {port} is in use, trying next port...")
            
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(self.credentials, token)
        
        self.service = build('gmail', 'v1', credentials=self.credentials)
    
    def _load_history_id(self) -> Optional[str]:
        """Load last processed history ID from Cloud Storage"""
        if not self.bucket_name:
            return None
            
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob('history_id.txt')
        
        try:
            return blob.download_as_text().strip()
        except Exception:
            return None
    
    def _save_history_id(self, history_id: str) -> None:
        """Save last processed history ID to Cloud Storage"""
        if not self.bucket_name:
            return
            
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob('history_id.txt')
        blob.upload_from_string(history_id)
    
    def get_new_messages(self) -> List[Dict]:
        """Fetch only new messages since last history ID"""
        if not self.service:
            self.authenticate()
        
        try:
            # Initialize history ID if not exists
            if not self.history_id:
                print("\nNo history ID found. Initializing...")
                try:
                    profile = self.service.users().getProfile(userId='me').execute()
                    self.history_id = profile['historyId']
                    self._save_history_id(self.history_id)
                    print(f"Initialized history ID: {self.history_id}")
                    return []  # Skip processing on first run
                except Exception as e:
                    print(f"Error initializing history ID: {str(e)}")
                    return []
            
            print(f"\nChecking for new messages since history ID: {self.history_id}")
            new_messages = []
            
            try:
                # Get history of changes
                history_results = self.service.users().history().list(
                    userId='me',
                    startHistoryId=self.history_id,
                    labelId='INBOX'
                ).execute()
                
                if 'history' in history_results:
                    history_entries = history_results['history']
                    print(f"Found {len(history_entries)} history entries")
                    
                    for entry in history_entries:
                        if 'messagesAdded' in entry:
                            for msg_added in entry['messagesAdded']:
                                try:
                                    msg_id = msg_added['message']['id']
                                    # Get full message content
                                    msg = self.service.users().messages().get(
                                        userId='me',
                                        id=msg_id,
                                        format='full'
                                    ).execute()
                                    
                                    # Only include messages currently in INBOX
                                    if 'INBOX' in msg.get('labelIds', []):
                                        new_messages.append(msg)
                                        print(f"Found new message with ID: {msg_id}")
                                
                                except Exception as e:
                                    print(f"Error fetching message: {str(e)}")
                                    continue
                
                # Update history ID
                if 'historyId' in history_results:
                    new_history_id = history_results['historyId']
                    if new_history_id != self.history_id:
                        print(f"Updating history ID: {self.history_id} -> {new_history_id}")
                        self.history_id = new_history_id
                        self._save_history_id(self.history_id)
                
                return new_messages
                
            except Exception as e:
                print(f"Error in history request: {str(e)}")
                return []
                
        except Exception as e:
            print(f"Error in get_new_messages: {str(e)}")
            return []
    
    def extract_email_content(self, message: Dict) -> Dict:
        """Extract relevant information from email message"""
        headers = message['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '(no subject)')
        sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'unknown')
        date = next((h['value'] for h in headers if h['name'].lower() == 'date'), '')
        
        # Convert email date to datetime
        parsed_date = email.utils.parsedate_to_datetime(date) if date else datetime.now()
        
        # Extract body with better MIME part handling
        body = ''
        parts = [message['payload']]
        
        while parts:
            part = parts.pop(0)
            
            if 'parts' in part:
                parts.extend(part['parts'])
            elif 'body' in part and 'data' in part['body']:
                try:
                    decoded = base64.urlsafe_b64decode(
                        part['body']['data'].encode('UTF-8')).decode('utf-8')
                    body += decoded + '\n'
                except Exception as e:
                    print(f"Error decoding message part: {str(e)}")
        
        # Clean up body
        body = body.strip()
        
        return {
            'message_id': message['id'],
            'thread_id': message['threadId'],
            'subject': subject,
            'sender': sender,
            'date': parsed_date.strftime('%Y-%m-%d %H:%M:%S'),
            'body': body,
            'received_at': datetime.fromtimestamp(
                int(message['internalDate'])/1000).strftime('%Y-%m-%d %H:%M:%S'),
            'size': message['sizeEstimate']
        }

def process_messages(messages: List[Dict], poller: GmailPoller) -> List[Dict]:
    """Process new messages and return their content"""
    processed = []
    if messages:
        for msg in messages:
            try:
                content = poller.extract_email_content(msg)
                processed.append(content)
            except Exception as e:
                print(f"Error processing message: {str(e)}")
    return processed

if __name__ == '__main__':
    import time
    
    # For local testing
    if os.environ.get('LOCAL_TEST'):
        print("\nStarting Gmail Poller in continuous mode")
        print("Press Ctrl+C to stop")
        
        # Initialize poller once
        print("\nInitializing Gmail Poller...")
        poller = GmailPoller('credentials.json')
        
        try:
            while True:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{current_time}] Checking for new emails...")
                
                try:
                    messages = poller.get_new_messages()
                    processed = process_messages(messages, poller)
                    
                    if processed:
                        print(f"\nFound {len(processed)} new messages:")
                        for content in processed:
                            print("\n--------------------")
                            print(f"Subject: {content['subject']}")
                            print(f"From: {content['sender']}")
                            print(f"Date: {content['date']}")
                            if content.get('body'):
                                print(f"Preview: {content['body'][:150]}...")
                            print("--------------------")
                    else:
                        print("No new messages")
                    
                except Exception as e:
                    print(f"Error during polling: {str(e)}")
                    
                print("\nWaiting 60 seconds before next check...")
                time.sleep(60)
                
        except KeyboardInterrupt:
            print("\n\nStopping Gmail Poller. Goodbye!")
    
    # For Cloud Run
    else:
        print("\nStarting single Gmail poll")
        poller = GmailPoller('credentials.json')
        
        try:
            messages = poller.get_new_messages()
            processed = process_messages(messages, poller)
            
            if processed:
                print(f"\nProcessed {len(processed)} new messages")
                for content in processed:
                    print(f"- {content['subject']} from {content['sender']}")
            else:
                print("No new messages to process")
                
        except Exception as e:
            print(f"Error during polling: {str(e)}")
            raise