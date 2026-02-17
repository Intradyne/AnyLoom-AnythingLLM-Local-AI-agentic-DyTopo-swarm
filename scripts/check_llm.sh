#!/bin/bash
# check_llm.sh — Verify LLM server is reachable and serving the expected model.
#
# Usage:  bash scripts/check_llm.sh [endpoint]
# Default endpoint: http://localhost:8008/v1
#
# Exit codes:
#   0 — LLM server is healthy and serving the model
#   1 — LLM server is not reachable or model not found

set -euo pipefail

ENDPOINT="${1:-http://localhost:8008/v1}"
EXPECTED_MODEL="qwen3-30b-a3b"  # partial match

echo "Checking LLM server at $ENDPOINT..."

# Check /v1/models
RESPONSE=$(curl -s --max-time 10 "$ENDPOINT/models" 2>/dev/null) || {
    echo "FAIL: Cannot reach $ENDPOINT/models"
    echo "  Is the LLM server running? Start it with: docker compose up -d llm"
    exit 1
}

# Check model name
if echo "$RESPONSE" | grep -qi "$EXPECTED_MODEL"; then
    MODEL_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
    echo "OK: LLM server is healthy"
    echo "  Model: $MODEL_ID"
    echo "  Endpoint: $ENDPOINT"
    exit 0
else
    echo "WARN: LLM server is reachable but model '$EXPECTED_MODEL' not found"
    echo "  Response: $RESPONSE"
    exit 1
fi
