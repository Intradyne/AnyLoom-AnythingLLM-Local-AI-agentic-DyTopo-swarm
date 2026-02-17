# LLM Engine Docker Setup Guide

This guide covers running llama.cpp server in Docker for the AnyLoom system.

---

## Overview

llama.cpp runs as a Docker container managed by `docker-compose.yml`. The container exposes the OpenAI-compatible API on **host port 8008** (container port 8080).

---

## Quick Start

### 1. Download the Model

```bash
# Install huggingface-cli if needed
pip install huggingface-hub

# Download Q4_K_M GGUF (~18.6 GB)
huggingface-cli download unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF \
  Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
  --local-dir ./models
```

### 2. Start the Container

```bash
docker compose up -d
```

This starts the llama.cpp container in detached mode with:
- Model: Qwen3-30B-A3B-Instruct-2507 (Q4_K_M GGUF)
- Host port: 8008
- Container port: 8080
- GPU access enabled
- 131K context length

### 3. Check Container Status

```bash
docker compose ps
```

Expected output should show the container as "running" with port mapping `0.0.0.0:8008->8080/tcp`.

### 4. Monitor Logs

```bash
docker compose logs -f anyloom-llm
```

Wait for the health check to pass. You can verify with:
```bash
curl http://localhost:8008/health
```

This means llama.cpp is ready. Press `Ctrl+C` to stop following logs (container continues running).

### 5. Verify API Access

```bash
curl http://localhost:8008/v1/models
```

Test inference:

```bash
curl http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-30B-A3B-Instruct-2507",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'
```

### 6. Configure AnythingLLM

AnythingLLM connects to llama.cpp via the Docker network:

- **LLM Provider:** Generic OpenAI
- **Base URL:** `http://anyloom-llm:8080/v1` (Docker internal)
- **Model:** `Qwen3-30B-A3B-Instruct-2507`
- **API Key:** (leave empty)

Or from the host: `http://localhost:8008/v1`

---

## Stopping the Container

```bash
docker compose down
```

This stops and removes the container. To stop without removing:

```bash
docker compose stop
```

---

## Configuration

llama.cpp settings are defined in `docker-compose.yml`. Key parameters:

- **Context length:** Set via `--ctx-size` flag (default: 131072)
- **GPU layers:** Set via `--n-gpu-layers` flag (99 = full offload)
- **KV cache quantization:** Set via `--cache-type-k` and `--cache-type-v` flags
- **Flash attention:** Enabled via `--flash-attn on`
- **Tool calling:** Enabled via `--jinja`

See `docker-compose.yml` for current configuration.

---

## Troubleshooting

### Container won't start

Check Docker logs:
```bash
docker compose logs anyloom-llm
```

Common issues:
- Out of GPU memory: Use a smaller GGUF quantization or reduce `--ctx-size`
- Port conflict: Ensure port 8008 is not in use
- Model not found: Verify the GGUF file is in the mounted volume

### Can't access API

Verify container is running:
```bash
docker compose ps
```

Check health endpoint:
```bash
curl http://localhost:8008/health
```

---

## See Also

- **[INSTALL.md](../INSTALL.md)** - Full installation guide
- **[docs/llm-engine.md](llm-engine.md)** - Detailed llama.cpp configuration
- **[docs/qwen3-model.md](qwen3-model.md)** - Model details and sampling parameters
