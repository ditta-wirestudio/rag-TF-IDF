#!/usr/bin/env bash
# Zero-setup demo: offline embeddings + in-memory store. No Postgres, no API key.
set -e
pip install fastapi uvicorn >/dev/null 2>&1 || true
cd "$(dirname "$0")/backend"
echo "RAG eval service (offline demo) → http://127.0.0.1:8000/"
STORE=memory exec uvicorn main:app --host 127.0.0.1 --port 8000
