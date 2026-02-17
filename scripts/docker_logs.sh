#!/bin/bash
# View AnyLoom Docker logs

SERVICE=${1:-}

if [ -z "$SERVICE" ]; then
    echo "Usage: $0 [service]"
    echo ""
    echo "Available services:"
    echo "  • qdrant"
    echo "  • llm"
    echo "  • anythingllm"
    echo "  • all (default)"
    echo ""
    echo "Example: $0 llm"
    echo ""
    docker compose logs --tail=50
else
    docker compose logs -f "$SERVICE"
fi
