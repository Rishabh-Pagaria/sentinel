# Sentinel: Real-time Email Phishing Detection

A real-time phishing detection system powered by Google's Gemini 1.5 Flash model with adversarial defense mechanisms and prompt injection protection.

## Features

- Real-time email monitoring via Gmail API
- Advanced phishing detection using Gemini 1.5 Flash
- WebSocket-based live monitoring dashboard
- Adversarial defense mechanisms
- Prompt injection protection
- Cloud-native deployment ready

## Prerequisites

- Python 3.8 or higher
- A Google Cloud Project with Gmail API enabled
- Gemini API access
- Enable Cloud Storage API

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/sentinel.git
cd sentinel
```

### 2. Create Python Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Up Google Cloud Credentials

#### a. Gmail API Credentials:
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select existing one
3. Enable Gmail API
4. Create OAuth 2.0 credentials
   - Application type: Web
   - Download as `credentials.json`
   - Place in project root directory

#### b. Service Account for Cloud Storage:
1. Create service account in Google Cloud Console
2. Grant Storage Object Viewer/Creator roles, if there is error then grant Admin Roles
3. Download key as `vertex-key.json`
4. Place in project root directory

### 4. Environment Configuration
Create `.env` file in project root:
```env
GEMINI_API_KEY=your_gemini_api_key
HISTORY_BUCKET_NAME=your_bucket_name
LOCAL_TEST=1  # For local testing
GCP_SERVICE_ACCOUNT="service_aacount"
GCP_CREDENTIALS_PATH=vertex-key.json
GMAIL_CREDENTIALS_PATH="credentials.json"
GCP_PROJECT_ID="project_id"
```

## Running the System

### 1. Local Development
```bash
# Start the real-time monitor
python src/api/realtime_monitor.py
```
Visit `http://localhost:8000` in your browser to see the monitoring dashboard.

## Security Features

- Prompt injection mitigation
- Adversarial perturbation resistance
- Input sanitization
- Rate limiting
- Secure credential handling

## Evaluation

Before running the evaluation, prepare the datasets:

1. Generate train/test/eval splits:
```bash
# Explore and preprocess the data
python scripts/explore_data.py

# Create train/test/eval splits
python scripts/create_splits.py
```

2. Generate adversarial examples:
```bash
# Create adversarial test examples 
python scripts/generate_adversarial.py
```

3. Run model evaluation:
```bash
python scripts/evaluate_model.py
```

This will:
- Test against standard dataset
- Run adversarial examples
- Generate performance metrics

## Demo
Watch our system in action: [Demo Video](https://www.canva.com/design/DAG1SBAbgp4/IBaorwZM_icZSeuJ378Jxg/watch?utm_content=DAG1SBAbgp4&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h41f20d003b)

The demo showcases:
- Real-time email monitoring
- Phishing detection with explanations
- Live dashboard updates