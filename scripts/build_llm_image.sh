#!/bin/bash
# Build llama.cpp Docker image with Blackwell (sm_120) CUDA support.
#
# The official ghcr.io/ggml-org/llama.cpp:server-cuda image ships CUDA 12.4,
# which predates Blackwell. RTX 5090 falls back to PTX JIT -> ~3-4 tok/s
# instead of ~50-60 tok/s. This script builds a local image with CUDA 12.8
# and native sm_120 kernels.
#
# Requirements:
#   - Docker Desktop with WSL2 integration
#   - ~10 GB disk for build context + image layers
#   - ~10-20 minutes build time (first time; cached rebuilds are faster)
#
# Usage:
#   bash scripts/build_llm_image.sh

set -e

IMAGE_NAME="local/llama.cpp:server-cuda-blackwell"
CUDA_VERSION="12.8.0"
CUDA_ARCH="120"
LLAMA_CPP_DIR="/tmp/llama.cpp-build"

echo "============================================"
echo "  Building llama.cpp with Blackwell support"
echo "============================================"
echo ""
echo "  Image:  $IMAGE_NAME"
echo "  CUDA:   $CUDA_VERSION"
echo "  Arch:   sm_$CUDA_ARCH (Blackwell)"
echo ""

# Clone or update llama.cpp source
if [ -d "$LLAMA_CPP_DIR" ]; then
    echo "Updating existing llama.cpp source..."
    cd "$LLAMA_CPP_DIR"
    git fetch --depth 1 origin master
    git checkout FETCH_HEAD
else
    echo "Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
    cd "$LLAMA_CPP_DIR"
fi

echo ""
echo "Building Docker image (this takes 10-20 minutes)..."
echo ""

# Build with CUDA 12.8 + sm_120 (Blackwell native).
# GGML_CUDA_NO_PINNED=1 avoids GDDR7 pinned-memory issues on RTX 5090.
# GGML_CUDA_FORCE_CUBLAS=ON fixes prompt processing bug on Blackwell (0.27 t/s without).
# GGML_CUDA_FA_ALL_QUANTS=ON enables sub-f16 KV cache with flash attention.
# GGML_CUDA_GRAPHS=ON batches kernel launches (up to 1.2x speedup).
# GGML_FLASH_ATTN=ON enables flash attention kernels (~27% speedup on Blackwell).
# Explicit sm_120 avoids pulling in compute_120a (requires CUDA 13.x).
docker build -t "$IMAGE_NAME" \
    --build-arg CUDA_VERSION="$CUDA_VERSION" \
    --build-arg CUDA_DOCKER_ARCH="$CUDA_ARCH" \
    --build-arg CMAKE_ARGS="-DGGML_CUDA_NO_PINNED=1 -DGGML_CUDA_FORCE_CUBLAS=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_GRAPHS=ON -DGGML_FLASH_ATTN=ON" \
    --target server \
    -f .devops/cuda.Dockerfile .

echo ""
echo "============================================"
echo "  Build complete!"
echo "============================================"
echo ""
echo "  Image: $IMAGE_NAME"
echo ""
echo "  Verify with:"
echo "    docker run --rm --gpus all $IMAGE_NAME --help 2>&1 | head -5"
echo ""
echo "  docker-compose.yml already references this image."
echo "  Restart the stack:"
echo "    bash scripts/docker_stop.sh && bash scripts/docker_start.sh"
echo ""
