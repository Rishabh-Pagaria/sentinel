# Quick Start Guide - Sentinel Gmail Add-on

## ⚡ Fast Setup (5 minutes)

### 1. Install clasp (Google Apps Script CLI)
```powershell
npm install -g @google/clasp
```

### 2. Enable Apps Script API
Visit: https://script.google.com/home/usersettings
Toggle ON: "Google Apps Script API"

### 3. Login to clasp
```powershell
cd gmail_addon
clasp login
```
Follow browser authentication.

### 4. Create & Deploy
```powershell
# Create new Apps Script project
clasp create --type standalone --title "Sentinel Email Analyzer"

# Upload your code
clasp push

# Open in browser
clasp open
```

### 5. Install Add-on in Gmail
In the Apps Script editor that opens:
1. Click **Deploy** → **Test deployments**
2. Click **Install**
3. Grant permissions
4. Open Gmail and click any email
5. Look for "Sentinel Email Analyzer" in right sidebar

### 6. Start Backend
```powershell
cd ..
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## 🎯 What You'll See

```
Gmail Interface:
┌─────────────────────────┬──────────────────────────┐
│ Inbox                   │ Sentinel Email Analyzer   │
│                         ├──────────────────────────┤
│ □ Email 1               │ Subject: Order Confirm    │
│ □ Email 2               │ From: noreply@amazon.com  │
│ ✓ Email 3 (selected)    │                          │
│                         │ Email Body:               │
│                         │ Dear customer, your...    │
│                         │                          │
│                         │ [Analyze for Phishing]    │
└─────────────────────────┴──────────────────────────┘
```

After clicking "Analyze for Phishing":
```
┌──────────────────────────┐
│ ✅ Email Appears Safe     │
│ Confidence: 87%          │
├──────────────────────────┤
│ Subject: Order Confirm   │
│ From: noreply@amazon.com │
├──────────────────────────┤
│ Detected Tactics: none   │
│                          │
│ 💡 Security Tip:         │
│ This email appears       │
│ legitimate. Always       │
│ verify sender addresses. │
└──────────────────────────┘
```