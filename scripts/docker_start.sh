#!/bin/bash
# Start AnyLoom Docker Stack

set -e

echo "============================================"
echo "  Starting AnyLoom Docker Stack"
echo "============================================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Quick GPU check (non-blocking — avoids pulling a CUDA image)
if docker info 2>/dev/null | grep -qi nvidia; then
    echo "GPU support detected."
else
    echo "WARNING: NVIDIA runtime not detected in Docker."
    echo "LLM server requires GPU. Make sure Docker Desktop has WSL2 integration enabled."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "✓ Created .env file"
fi

# Source .env for MODEL_DIR (strip \r from Windows CRLF line endings)
LLM_MODEL_DIR="${LLM_MODEL_DIR:-./models}"
if [ -f .env ]; then
    # shellcheck disable=SC1091
    source <(sed 's/\r$//' .env) 2>/dev/null || true
fi

# Check models directory and GGUF files
GGUF_FILE="Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
if [ ! -d "$LLM_MODEL_DIR" ]; then
    echo "Creating models directory: $LLM_MODEL_DIR"
    mkdir -p "$LLM_MODEL_DIR"
fi

if [ ! -f "$LLM_MODEL_DIR/$GGUF_FILE" ]; then
    echo ""
    echo "WARNING: Model file not found: $LLM_MODEL_DIR/$GGUF_FILE"
    echo "The LLM server needs a GGUF model file to start."
    echo ""
    echo "Download it with:"
    echo "  huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507-GGUF \\"
    echo "    $GGUF_FILE --local-dir $LLM_MODEL_DIR"
    echo ""
    read -p "Continue without model? LLM container will fail. (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

EMBED_FILE="bge-m3-q8_0.gguf"
if [ ! -f "$LLM_MODEL_DIR/$EMBED_FILE" ]; then
    echo ""
    echo "WARNING: Embedding model not found: $LLM_MODEL_DIR/$EMBED_FILE"
    echo "The embedding server needs this file for RAG."
    echo ""
    echo "Download it with:"
    echo "  huggingface-cli download ggml-org/bge-m3-Q8_0-GGUF \\"
    echo "    $EMBED_FILE --local-dir $LLM_MODEL_DIR"
    echo ""
    read -p "Continue without embedding model? RAG will not work. (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if the configured LLM image exists
LLM_IMAGE="${LLM_IMAGE:-local/llama.cpp:server-cuda-blackwell}"
if ! docker image inspect "$LLM_IMAGE" > /dev/null 2>&1; then
    echo ""
    echo "WARNING: Docker image not found: $LLM_IMAGE"
    if [[ "$LLM_IMAGE" == *"blackwell"* ]]; then
        echo ""
        echo "Build the Blackwell image (recommended for RTX 5090):"
        echo "  bash scripts/build_llm_image.sh"
        echo ""
        echo "Or use the official image (slower on RTX 5090, works on all GPUs):"
        echo "  Set LLM_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda in .env"
        echo ""
        read -p "Use official image for now? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            export LLM_IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda"
            echo "Using fallback: $LLM_IMAGE"
        else
            echo "Run 'bash scripts/build_llm_image.sh' first, then try again."
            exit 1
        fi
    fi
fi

# Create external Docker volumes if they don't exist
for vol in anyloom_qdrant_storage anyloom_anythingllm_storage anyloom_anythingllm_hotdir; do
    if ! docker volume inspect "$vol" > /dev/null 2>&1; then
        echo "Creating Docker volume: $vol"
        docker volume create "$vol"
    fi
done

# Start services
echo ""
echo "Starting services..."
docker compose up -d

echo ""
echo "============================================"
echo "  Waiting for services to be healthy..."
echo "============================================"
echo ""

# Wait for Qdrant (uses /healthz, not /health)
echo -n "Waiting for Qdrant..."
until curl -sf http://localhost:6333/healthz > /dev/null 2>&1; do
    echo -n "."
    sleep 2
done
echo " ✓"

# Wait for Embedding server
echo -n "Waiting for Embedding server..."
until curl -sf http://localhost:8009/health > /dev/null 2>&1; do
    echo -n "."
    sleep 2
done
echo " ✓"

# Wait for LLM server (this can take a while on first start)
echo -n "Waiting for LLM server (this may take 2-5 minutes on first start)..."
LLM_WAIT=0
until curl -sf http://localhost:8008/health > /dev/null 2>&1; do
    echo -n "."
    sleep 5
    LLM_WAIT=$((LLM_WAIT + 5))
    if [ $LLM_WAIT -ge 300 ]; then
        echo ""
        echo "WARNING: LLM server is taking longer than expected."
        echo "Check logs with: docker compose logs llm"
        echo "Continuing anyway..."
        break
    fi
done
echo " ✓"

# Wait for AnythingLLM (fresh install returns 403 before setup wizard;
# curl without -f succeeds on any HTTP response including 403)
echo -n "Waiting for AnythingLLM..."
until curl -s -o /dev/null http://localhost:3001/ 2>/dev/null; do
    echo -n "."
    sleep 2
done
echo " ✓"

echo ""
echo "============================================"
echo "  AnyLoom Stack Ready!"
echo "============================================"
echo ""
echo "Services:"
echo "  • Qdrant:       http://localhost:6333"
echo "  • Embedding:    http://localhost:8009/v1"
echo "  • LLM API:      http://localhost:8008/v1"
echo "  • AnythingLLM:  http://localhost:3001"
echo ""
echo "Check status:    docker compose ps"
echo "View logs:       docker compose logs -f [service]"
echo "Stop services:   docker compose down"
echo ""
echo "Next steps:"
echo "  1. Configure AnythingLLM: python scripts/configure_anythingllm.py"
echo "  2. Run benchmarks: python scripts/benchmarks/bench_run_all.py"
echo ""
