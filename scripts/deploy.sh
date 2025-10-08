#!/bin/bash
# Deploy Gmail Poller to Cloud Run and set up Cloud Scheduler

# Load environment variables
set -a
source .env
set +a

# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/gmail-poller

gcloud run deploy gmail-poller \
    --image gcr.io/$GCP_PROJECT_ID/gmail-poller \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 512Mi \
    --timeout 300 \
    --update-env-vars HISTORY_BUCKET_NAME=$HISTORY_BUCKET_NAME

# Get the deployed service URL
SERVICE_URL=$(gcloud run services describe gmail-poller \
    --platform managed \
    --region us-central1 \
    --format 'value(status.url)')

# Create Cloud Scheduler job
gcloud scheduler jobs create http gmail-poller-job \
    --schedule "*/5 * * * *" \
    --uri "${SERVICE_URL}/poll" \
    --http-method POST \
    --attempt-deadline 5m \
    --location us-central1