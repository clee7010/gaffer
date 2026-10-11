#!/usr/bin/env bash
# Deploy Gaffer to Google Cloud Run.
#
# One-time setup:
#   gcloud auth login && gcloud config set project YOUR_PROJECT
#   gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com
#   printf '%s' "$ANTHROPIC_API_KEY" | gcloud secrets create anthropic-api-key --data-file=-
#   # let the Cloud Run runtime service account read the secret:
#   gcloud secrets add-iam-policy-binding anthropic-api-key \
#     --member="serviceAccount:$(gcloud projects describe "$(gcloud config get-value project)" \
#       --format='value(projectNumber)')-compute@developer.gserviceaccount.com" \
#     --role=roles/secretmanager.secretAccessor
set -euo pipefail

SERVICE="${SERVICE:-gaffer}"
REGION="${REGION:-australia-southeast1}"

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300 \
  --concurrency 4 \
  --max-instances 3 \
  --set-secrets ANTHROPIC_API_KEY=anthropic-api-key:latest \
  --no-allow-unauthenticated

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')
echo "Deployed: $URL"
echo "Try: curl -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" $URL/health"
