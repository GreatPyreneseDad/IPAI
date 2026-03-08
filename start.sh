#!/usr/bin/env bash
set -e

# If GCP_SERVICE_ACCOUNT_JSON is set, write it to disk so BigQuery client finds it
if [ -n "$GCP_SERVICE_ACCOUNT_JSON" ]; then
  echo "$GCP_SERVICE_ACCOUNT_JSON" > /tmp/gcp-key.json
  export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-key.json
  echo "[start.sh] GCP credentials written to /tmp/gcp-key.json"
fi

exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
