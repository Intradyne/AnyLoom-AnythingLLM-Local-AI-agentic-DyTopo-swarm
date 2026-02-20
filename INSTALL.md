# AnyLoom Installation Guide (Docker Edition)

Complete Docker-based installation for AnyLoom - the fully local AI agentic stack.

> **TL;DR:** Install Docker Desktop with GPU support → Download GGUF model to `./models/` → Run `bash scripts/docker_start.sh` → Done!

---

## System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | RTX 4090/5090 (32GB VRAM) | RTX 5090 (32GB VRAM) |
| **RAM** | 32GB | 64GB+ |
| **Storage** | 100GB free | 200GB+ SSD |
| **CPU** | 8-core | 12+ core |

### Laptop Profile (8GB VRAM)

This branch (`feature/laptop-constraint-2070`) is optimized for laptops:

- **GPU:** RTX 2070/2080 (8GB VRAM)
- **LLM:** Qwen2.5-Coder-7B-Instruct Q4_K_M (~4.5 GB)
- **Context:** 8K tokens (vs 131K on desktop)
- **Embedding:** BGE-M3 on CPU (no GPU VRAM used)
- **Concurrency:** Single slot, sequential execution
- **DyTopo rounds:** 3 max (vs 5 on desktop)

Download the laptop model:
```bash
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
  qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --local-dir models
```

### Software

| Component | Version | Purpose |
|-----------|---------|---------|
| **Windows** | 10/11 Pro/Home | Host OS |
| **WSL2** | Latest | Docker backend |
| **Docker Desktop** | 24.0+ | Container runtime |
| **NVIDIA Driver** | 570+ | GPU support (570.65+ for RTX 5090 Blackwell) |
| **Python** | 3.10+ | Benchmarks and DyTopo scripts |

---

## Architecture Overview

AnyLoom runs as a **Docker Compose stack**:

```
┌──────────────────────────────────────────────────────────────┐
│  Docker Network: anyloom                                     │
│                                                              │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │   Qdrant     │  │   llama.cpp LLM  │  │  llama.cpp   │  │
│  │   :6333      │  │   Qwen3-30B GPU  │  │  Embedding   │  │
│  │   Vector DB  │  │   :8008          │  │  BGE-M3 GPU  │  │
│  │              │  │                  │  │  :8009        │  │
│  └──────────────┘  └──────────────────┘  └──────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              AnythingLLM                              │   │
│  │   Chat UI + Document Management                       │   │
│  │   :3001                                               │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │   Windows Host          │
              │   Python Scripts        │
              │   DyTopo, Benchmarks    │
              │   Health Monitor        │
              └─────────────────────────┘
```

### Endpoints

```bash
LLM_ENDPOINT="http://localhost:8008/v1"         # llama.cpp LLM (GPU)
EMBEDDING_ENDPOINT="http://localhost:8009/v1"    # llama.cpp Embedding (BGE-M3, for AnythingLLM)
QDRANT_URL="http://localhost:6333"               # Qdrant vector DB
ANYTHINGLLM_URL="http://localhost:3001"          # AnythingLLM UI
```

---

## Installation Steps

### 1. Install Prerequisites

#### A. Install WSL2 (if not already installed)

```powershell
# Run in PowerShell as Administrator
wsl --install
# Restart your computer when prompted
```

Verify:
```powershell
wsl --list --verbose
# Should show a default Linux distribution
```

#### B. Update NVIDIA Driver (Windows)

1. Download latest driver from [nvidia.com/drivers](https://www.nvidia.com/drivers)
2. Install (requires 570+ for Blackwell/RTX 5090)
3. Verify in PowerShell:

```powershell
nvidia-smi
# Should show your GPU and driver version 570+
```

#### C. Install Docker Desktop

1. Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
2. Install and restart Windows
3. Open Docker Desktop and verify it starts successfully

**Enable WSL2 integration:**
- Docker Desktop → Settings → Resources → WSL Integration
- Enable integration with your default WSL distribution
- Click "Apply & Restart"

**Enable GPU support:**
- Docker Desktop should automatically detect NVIDIA GPU
- Verify GPU access:

```bash
# In WSL or PowerShell
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If this shows your GPU, you're ready!

---

### 2. Clone Repository

```bash
git clone <repo-url>
cd AnyLoom
```

---

### 3. Download Models

llama.cpp loads GGUF model files from the local `./models/` directory. You need two models: the LLM (GPU) and the embedding model (GPU for AnythingLLM, ONNX INT8 CPU for MCP RAG).

```bash
# Create the models directory
mkdir -p models
pip install huggingface_hub

# LLM model — Qwen3-30B-A3B Q4_K_M (~18.6 GB, runs on GPU)
huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507-GGUF \
  Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
  --local-dir models

# Embedding model — BGE-M3 Q8_0 (~605 MB, runs on GPU)
huggingface-cli download ggml-org/bge-m3-Q8_0-GGUF \
  bge-m3-q8_0.gguf \
  --local-dir models
```

Verify both files are in place:

```bash
ls -lh models/*.gguf
# Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf  ~18.6 GB
# bge-m3-q8_0.gguf                          ~605 MB
```

> **Note:** Model filenames must match `docker-compose.yml`. If you use a different LLM quantization (e.g., Q5_K_M), update the `--model` path in the compose file.

> **Already have the LLM GGUF?** If you downloaded it via LM Studio, symlink instead:
> ```bash
> ln -s ~/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf models/
> ```

---

### 4. Build Blackwell Docker Image (RTX 5090)

The official llama.cpp Docker image ships CUDA 12.4 which lacks Blackwell (sm_120) support. RTX 5090 falls back to PTX JIT — ~3-4 tok/s instead of ~50-60 tok/s. Build a local image with native Blackwell kernels:

```bash
bash scripts/build_llm_image.sh
```

This clones llama.cpp, builds with CUDA 12.8 + sm_120, and tags as `local/llama.cpp:server-cuda-blackwell`. Takes ~10-20 minutes on first build. The docker-compose.yml uses this image by default.

The build script passes several Blackwell-critical CMAKE flags:
- **`GGML_CUDA_FORCE_CUBLAS=ON`** — Fixes a Blackwell matmul bug that tanks prompt processing to ~0.27 t/s without it.
- **`GGML_CUDA_FA_ALL_QUANTS=ON`** — Enables flash-attention kernels for quantized KV cache types (q8_0, q4_0, etc.).
- **`GGML_CUDA_GRAPHS=ON`** — Enables CUDA graph capture for reduced kernel launch overhead.
- **`GGML_FLASH_ATTN=ON`** — Builds flash-attention support.
- **`GGML_CUDA_NO_PINNED=1`** — Avoids pinned memory allocation issues on some systems.

> **Non-Blackwell GPUs:** Skip this step. Set `LLM_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda` in `.env` to use the official image.

---

### 5. Configure Environment (Optional)

The defaults are tuned for an **RTX 5090 (32GB VRAM)**. They work out-of-the-box on that card, but you can customize:

```bash
# Create environment file
cp .env.example .env

# Edit if needed (optional)
notepad .env
```

**Key settings in `.env`:**

```bash
# llama.cpp settings (defaults tuned for RTX 5090 / 32GB VRAM)
LLM_MODEL_DIR=./models
LLM_CONTEXT_SIZE=131072

# Ports (defaults are fine)
LLM_PORT=8008           # llama.cpp LLM (GPU)
EMBEDDING_PORT=8009     # llama.cpp Embedding (BGE-M3, for AnythingLLM)
QDRANT_PORT=6333        # Qdrant vector DB
ANYTHINGLLM_PORT=3001   # AnythingLLM UI
```


---

### 6. Start the Stack

```bash
# Automated startup with health checks
bash scripts/docker_start.sh
```

Or manually (must create volumes first):

```bash
docker volume create anyloom_qdrant_storage
docker volume create anyloom_anythingllm_storage
docker volume create anyloom_anythingllm_hotdir
docker compose up -d
```

**What happens on startup:**

1. **Qdrant** starts immediately (~5 seconds)
2. **llama.cpp LLM** reads the Qwen3 GGUF from `./models/` and loads it into GPU VRAM (~1-2 minutes)
3. **llama.cpp Embedding** reads the BGE-M3 GGUF from `./models/` and loads into GPU VRAM (~10 seconds)
4. **AnythingLLM** starts after Qdrant and Embedding are healthy (doesn't block on LLM)

**First startup: ~1-2 minutes** (model VRAM loading). AnythingLLM UI is available while the LLM finishes loading — chat will work as soon as the LLM health check passes.

---

### 7. Verify Services

Check all services are healthy:

```bash
docker compose ps
```

Expected output:

```
NAME                     STATUS
anyloom-qdrant         Up (healthy)
anyloom-llm            Up (healthy)
anyloom-embedding      Up (healthy)
anyloom-anythingllm    Up (healthy)
```

**Test endpoints:**

```bash
# Qdrant
curl http://localhost:6333/health

# llama.cpp LLM
curl http://localhost:8008/v1/models

# llama.cpp Embedding
curl http://localhost:8009/v1/models

# AnythingLLM
curl http://localhost:3001/api/v1/system
```

All should return successful responses.

---

### 8. Configure AnythingLLM

AnythingLLM requires a one-time browser setup before the API is accessible:

1. **Open http://localhost:3001** and complete the initial setup wizard (set a password, basic preferences)
2. Once the UI loads, the API is unlocked. Then run the automated configuration:

```bash
# Install Python dependencies first (if not already done)
pip install requests

# Configure AnythingLLM workspace
python scripts/configure_anythingllm.py
```

This script:
- Checks all 4 services are reachable (LLM, Embedding, Qdrant, AnythingLLM)
- Sets system-wide defaults: LLM provider (Generic OpenAI → llama.cpp), embedding engine (BGE-M3 via llama.cpp GPU container), vector DB (Qdrant), max generation tokens (4096), chunk size (2500 chars, ~625 tokens via EmbeddingModelMaxChunkLength -- the only effective chunk control), and the default system prompt
- Creates an `AnyLoom` workspace
- Uploads and embeds RAG reference documents from `rag-docs/anythingllm/*.md` into the workspace's Qdrant vector store (idempotent — skips files already uploaded/embedded)
- Pushes workspace settings (temp=0.3, topN=8, history=30, system prompt)
- Verifies all settings took effect
- Runs a smoke-test query to verify end-to-end connectivity (now with RAG context!)

**Or configure manually (after browser setup):**

1. Settings → LLM Provider:
   - Provider: `Generic OpenAI`
   - Base URL: `http://anyloom-llm:8080/v1` (Docker internal)
   - Model: `qwen3-30b-a3b-instruct-2507`
   - Token Context Window: `131072`
   - Max Tokens: `4096`
   - API Key: `not-needed`
2. Settings → Vector Database:
   - Provider: `Qdrant`
   - URL: `http://anyloom-qdrant:6333`
3. Settings → Embedding:
   - Provider: `Generic OpenAI`
   - Base URL: `http://anyloom-embedding:8080/v1` (Docker internal, GPU)
   - Model: `bge-m3-q8_0`
   - API Key: `not-needed`
   - Max Chunk Length: `2500` (this is the only effective chunk size control; `TextSplitterChunkSize` and `TextSplitterChunkOverlap` are not functional via the API)
5. Create workspace → set system prompt from `prompts/anythingllm-system-prompt.md`
6. Upload RAG documents: In the workspace, click the upload icon and drag in all `.md` files from `rag-docs/anythingllm/`. Click "Move to Workspace" then "Save and Embed".

---

### 9. Custom Agent Skills

Custom agent skills extend AnythingLLM's `@agent` with additional tools that can be invoked during chat. They run as JavaScript modules inside the AnythingLLM container and are called on demand when the agent determines a skill is relevant to the user's query.

**Installation:**

```bash
# Requires the Docker stack to be running
bash scripts/install_skills.sh
```

This installs the following skills:

| Skill | Description |
|-------|-------------|
| **asset-price** | Fetches market prices for stocks, crypto, commodities, indices, and ETFs via Yahoo Finance |
| **smart-web-reader** | Extracts clean, readable content from web pages using Defuddle/Readability |

**Post-install configuration:**

1. In the AnythingLLM UI, go to **Workspace Settings > Agent Configuration > Custom Skills**
2. Enable the skills you want to use

**Manual installation alternative:**

If you prefer to install skills individually:

```bash
docker cp skills/asset-price anyloom-anythingllm:/app/server/storage/plugins/agent-skills/
docker cp skills/smart-web-reader anyloom-anythingllm:/app/server/storage/plugins/agent-skills/
```

---

### 10. MCP Servers

Model Context Protocol (MCP) provides additional tools to the AnythingLLM agent beyond what custom skills offer. AnyLoom uses **8 MCP servers** split across two layers.

**AnythingLLM MCP servers (6)** — run inside the AnythingLLM container as stdio child processes, configured in `config/anythingllm_mcp_servers.json` and deployed by `scripts/configure_anythingllm.py` (step 8 above):

| Server | Transport | Description |
|--------|-----------|-------------|
| **fetch** | `uvx` (Python) | Simple page fetching for basic URL retrieval |
| **memory** | `npx` (Node) | Persistent knowledge graph across sessions |
| **tavily** | `npx` (Node) | Web search via Tavily API |
| **context7** | `npx` (Node) | Up-to-date library documentation lookup |
| **filesystem** | `npx` (Node) | File read/write within AnythingLLM storage |
| **sequential-thinking** | `npx` (Node) | Structured multi-step reasoning |

**llama.cpp agent MCP servers (2)** — available to the llama.cpp inference backend via stdio:

| Server | Tools | Description |
|--------|-------|-------------|
| **qdrant-rag** | 8 tools | RAG search, reindex, sources, file info + DyTopo swarm start/status/result |
| **system-status** | 6 tools | Service health, Qdrant collections, GPU status, LLM slots, Docker status, stack config |

**Verification:**

In the AnythingLLM UI, go to **Settings > Agent Configuration > MCP** to confirm all 6 AnythingLLM servers are listed and active.

> **Note:** MCP servers auto-start when `@agent` is invoked — there is no separate process to manage.

---

### 11. Install Python Dependencies (for Benchmarks & DyTopo)

```bash
# Windows PowerShell or Command Prompt
cd AnyLoom
pip install -r requirements-dytopo.txt
```

---

### 12. Start Health Monitor (Optional)

The health monitor is a standalone Python sidecar that runs alongside the Docker stack. It performs deterministic health checks every 30 seconds, auto-restarts failed containers, and logs structured JSONL to `~/anyloom-logs/health.jsonl`.

```bash
# Start the health monitor (runs in foreground, Ctrl+C to stop)
python scripts/health_monitor.py

# Or override the check interval via environment variable
CHECK_INTERVAL=5 python scripts/health_monitor.py
```

**Features:**
- Probes LLM, Qdrant, AnythingLLM, Embedding, and GPU every 30s (configurable)
- Auto-restarts failed Docker containers via `docker restart`
- Crash window protection: stops restarting after 3 failures within 15 minutes
- Alert cooldown: suppresses duplicate alerts for 30 minutes
- Config from `dytopo_config.yaml` `health_monitor` section, with env var overrides

> **Tip:** Run in the background or as a system service for always-on monitoring. Logs are append-only JSONL for easy parsing.

---

### 13. Run Benchmarks (Optional)

Test the full stack with benchmarks:

```bash
ANYTHINGLLM_API_KEY=your-key python scripts/benchmarks/bench_run_all.py
```

This runs 6 benchmark phases testing:
- Phase 1: Explanation tier (depth calibration, ≤150w)
- Phase 2: Adversarial fabrication (price hallucination resistance)
- Phase 3: Cross-workspace parity (RAG retrieval)
- Phase 4: Depth stability (determinism across 3 runs)
- Phase 5: Direct llama.cpp API validation
- Phase 6: Showcase gallery (curated responses)

Results are saved to `scripts/benchmarks/results/`. Current score: **15/20 (75%)**. See [benchmark results](scripts/benchmarks/docs/benchmark-results-showcase.md) for details.

---

## Daily Usage

### Starting the Stack

```bash
# Automatic startup with health checks
bash scripts/docker_start.sh

# Or manually
docker compose up -d
```

### Stopping the Stack

```bash
# Graceful shutdown
bash scripts/docker_stop.sh

# Or manually
docker compose down
```

### Viewing Logs

```bash
# All services
docker compose logs -f

# Specific service
bash scripts/docker_logs.sh llm
bash scripts/docker_logs.sh embedding
bash scripts/docker_logs.sh anythingllm
bash scripts/docker_logs.sh qdrant
```

### Restarting a Service

```bash
# Restart llama.cpp (e.g., after changing model settings)
docker compose restart llm

# Restart all services
docker compose restart
```

---

## Troubleshooting

### LLM Won't Start - GPU Not Detected

**Symptom:** llama.cpp container fails with "No GPU detected" or CUDA errors

**Solution:**

1. Verify GPU access in Docker:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

2. If this fails:
   - Update NVIDIA driver to 570+ (Windows host)
   - Restart Docker Desktop
   - Enable WSL2 integration in Docker Desktop settings
   - Restart Windows

### LLM Out of Memory

**Symptom:** `CUDA out of memory` in llama.cpp logs

**Solution:**

Q4_K_M weights are ~18.6 GiB + ~7 GiB KV cache (q8_0/q8_0 at 131K) = ~25.6 GiB total. 32GB VRAM recommended for full 131K context. 24GB GPUs can run with reduced context.

**If OOM on 32GB:**

1. Reduce context length in `.env`:
   ```bash
   LLM_CONTEXT_SIZE=16384   # or 8192 for minimal context
   ```

2. Ensure `--n-gpu-layers 99` and `--flash-attn on` are set in `docker-compose.yml` (they are by default).

3. Restart:
   ```bash
   docker compose down
   docker compose up -d
   ```

### AnythingLLM Can't Connect to llama.cpp

**Symptom:** AnythingLLM UI shows "LLM Error" or timeouts

**Solution:**

1. Check llama.cpp is healthy:
   ```bash
   docker compose ps anyloom-llm
   docker logs anyloom-llm
   ```

2. Verify internal Docker network connectivity:
   ```bash
   docker compose exec anythingllm curl http://anyloom-llm:8080/v1/models
   ```

3. Re-run configuration script:
   ```bash
   python scripts/configure_anythingllm.py
   ```

### LLM Won't Start - Model Not Found

**Symptom:** llama.cpp container exits with "model file not found" or similar error

**Solution:**

1. Verify the GGUF file exists in `./models/`:
   ```bash
   ls -lh models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
   ```

2. If missing, download the model (see Step 3 above)

3. Verify the filename matches `docker-compose.yml`:
   ```bash
   grep -- "--model" docker-compose.yml
   ```

4. Restart llama.cpp:
   ```bash
   docker compose restart llm
   ```

### LLM Won't Start - `--flash-attn` Error

**Symptom:** llama.cpp container exits with `unknown value for --flash-attn`

**Solution:**

The latest llama.cpp Docker image changed `--flash-attn` from a boolean flag to requiring an explicit value (`on`, `off`, or `auto`).

In `docker-compose.yml`, ensure the command uses:
```
--flash-attn on
```

Not just `--flash-attn` (without a value). Then restart:
```bash
docker compose down && docker compose up -d
```

### AnythingLLM API Returns 403

**Symptom:** `python scripts/configure_anythingllm.py` fails with `403 Forbidden`

**Solution:**

AnythingLLM requires a **one-time browser setup** before its API is accessible:

1. Open **http://localhost:3001** in your browser
2. Complete the setup wizard (set a password, choose preferences)
3. Once you see the main UI, the API is unlocked
4. Re-run `python scripts/configure_anythingllm.py`

### LLM Queries Time Out

**Symptom:** First query to llama.cpp times out (>120s)

**Solution:**

With 131K context and a large system prompt (~21K chars), the first request can take 2-3 minutes as llama.cpp processes the prompt cache. Subsequent requests are much faster. The benchmark helper uses a 300-second timeout to accommodate this.

If timeouts persist, check GPU utilization:
```bash
docker logs anyloom-llm --tail 20
```

### Services Won't Start - Port Conflicts

**Symptom:** `Error: port already in use`

**Solution:**

Check what's using the ports:

```bash
# Windows PowerShell
netstat -ano | findstr :8008
netstat -ano | findstr :8009
netstat -ano | findstr :6333
netstat -ano | findstr :3001
```

Either:
- Stop the conflicting service
- Or change ports in `.env`:

```bash
LLM_PORT=8010           # Change from 8008
EMBEDDING_PORT=8011     # Change from 8009
QDRANT_PORT=6334        # Change from 6333
ANYTHINGLLM_PORT=3002   # Change from 3001
```

Then restart: `docker compose down && docker compose up -d`

---

## Maintenance

### Updating Models

To use a different model:

1. Place the new GGUF file in `./models/`
2. Update the `--model` path in `docker-compose.yml`
3. Restart llama.cpp:

```bash
docker compose restart llm
```

### Backing Up Data

```bash
# Backup all Docker volumes
docker run --rm -v anyloom_qdrant_storage:/data -v $(pwd):/backup ubuntu \
  tar czf /backup/qdrant_backup.tar.gz /data

docker run --rm -v anyloom_anythingllm_storage:/data -v $(pwd):/backup ubuntu \
  tar czf /backup/anythingllm_backup.tar.gz /data
```

### Cleaning Up

```bash
# Stop and remove containers (keeps data)
docker compose down

# Remove everything including Docker volume data (⚠️ DESTRUCTIVE)
docker compose down -v

# Remove model files to free disk space (~19.2 GB)
rm -rf models/
```

---

## Performance Tuning

Defaults are tuned for an **RTX 5090 (32GB VRAM)** — 131K context. Q4_K_M weights are ~18.6 GiB + ~7 GiB KV cache (q8_0/q8_0 at 131K) = ~25.6 GiB total. Fits comfortably in 32GB.

### Tunable Parameters

The default `docker-compose.yml` command is already tuned for RTX 5090. Key flags you can adjust:

```yaml
# In docker-compose.yml under the llm service command:
--ctx-size 131072      # Context window — biggest VRAM lever
--batch-size 8192      # Prompt processing batch ceiling
--ubatch-size 4096     # GPU micro-batch per prefill pass (↑ = faster prompt processing, ↑ VRAM)
--threads 8            # CPU threads for inter-batch prep
--cache-type-k q8_0    # KV cache key quantization
--cache-type-v q8_0    # KV cache value quantization
```

After changes: `docker compose down && docker compose up -d llm`

### VRAM Requirements

32GB VRAM recommended for full 131K context. VRAM breakdown: ~18.6 GiB weights + ~7 GiB KV cache (q8_0/q8_0 at 131K) = ~25.6 GiB total. 24GB GPUs can run with reduced context.

> **Tip:** `--ctx-size` is the biggest VRAM lever. Halving it roughly halves the KV-cache memory. The default 131K uses ~25.6 GiB and fits comfortably on 32GB. Reduce to 32K or 64K if you need headroom for concurrent requests.

---

## Advanced: Running Without Docker Compose

If you prefer manual container management:

```bash
# Create the shared network first
docker network create anyloom

# Qdrant
docker run -d --name anyloom-qdrant \
  --network anyloom \
  -p 6333:6333 \
  -v anyloom_qdrant_storage:/qdrant/storage \
  --restart unless-stopped \
  qdrant/qdrant:latest

# llama.cpp LLM (GPU)
docker run -d --name anyloom-llm \
  --network anyloom \
  -p 8008:8080 \
  -v ./models:/models:ro \
  --gpus all \
  --ipc host \
  -e GGML_CUDA_GRAPH_OPT=1 \
  --restart unless-stopped \
  local/llama.cpp:server-cuda-blackwell \
  --model /models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
  --alias gpt-4 \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 131072 \
  --n-gpu-layers 99 \
  --flash-attn on \
  --fit on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --batch-size 8192 \
  --ubatch-size 4096 \
  --threads 8 \
  --threads-batch 16 \
  --jinja

# llama.cpp Embedding (GPU)
docker run -d --name anyloom-embedding \
  --network anyloom \
  -p 8009:8080 \
  -v ./models:/models:ro \
  --gpus all \
  --restart unless-stopped \
  local/llama.cpp:server-cuda-blackwell \
  --model /models/bge-m3-q8_0.gguf \
  --embeddings \
  --ctx-size 16384 \
  --batch-size 8192 \
  --ubatch-size 8192 \
  --n-gpu-layers 99 \
  --flash-attn on \
  --parallel 2 \
  --threads 4 \
  --host 0.0.0.0 \
  --port 8080

# AnythingLLM
docker run -d --name anyloom-anythingllm \
  --network anyloom \
  -p 3001:3001 \
  --shm-size 1g \
  --init \
  -v anyloom_anythingllm_storage:/app/server/storage \
  -e NODE_OPTIONS=--dns-result-order=ipv4first \
  -e ANYTHINGLLM_CHROMIUM_ARGS=--no-sandbox,--disable-setuid-sandbox,--disable-dev-shm-usage,--disable-gpu,--no-zygote,--disable-software-rasterizer \
  --restart unless-stopped \
  mintplexlabs/anythingllm:latest
```

But `docker compose` is recommended for easier management!

---

## Next Steps

- **Configure AnythingLLM workspaces** — Upload documents, create chat sessions
- **Run benchmarks** — Test the stack with `python scripts/benchmarks/bench_run_all.py`
- **Start the health monitor** — `python scripts/health_monitor.py` for always-on monitoring and auto-recovery
- **Explore DyTopo** — Multi-agent swarm orchestration with stigmergic routing (see `docs/dytopo-swarm.md`)
- **Review MCP tools** — 8 MCP servers for RAG, swarm, memory, web, files, and diagnostics (see `docs/qdrant-servers.md`)
- **Review policy.json** — The `policy.json` file at the project root defines deny-first tool-call policies for the DyTopo PolicyEnforcer (PCAS-Lite). Edit this file to customize which file paths, shell commands, and network hosts are allowed or denied during swarm execution. See `docs/dytopo-swarm.md` for details.

---

## Additional Resources

- [llm-engine.md](docs/llm-engine.md) — llama.cpp Docker configuration details
- [qwen3-model.md](docs/qwen3-model.md) — Model architecture and settings
- [architecture.md](docs/architecture.md) — System topology and design
- [anythingllm-settings.md](docs/anythingllm-settings.md) — AnythingLLM configuration guide

---

**You're now running a fully local, production-grade AI agentic stack!** 🎉

Everything runs automatically on startup, restarts on crashes, and persists data across reboots. Enjoy your privacy-first AI assistant!
