# llama.cpp Server: GPU-Accelerated Inference Backend

**GitHub:** https://github.com/ggerganov/llama.cpp
**Docker Image:** `local/llama.cpp:server-cuda-blackwell` (built locally with CUDA 12.8 + sm_120)
**Fallback Image:** `ghcr.io/ggml-org/llama.cpp:server-cuda` (CUDA 12.4, no Blackwell — PTX JIT, ~10× slower on RTX 5090)
**Container Name:** `anyloom-llm`
**Port:** 8008 (host) / 8080 (container)

---

## Overview

llama.cpp server is AnyLoom's inference backend for DyTopo swarm agent execution and AnythingLLM chat. It provides GPU-accelerated LLM serving with GGUF model support, quantized KV cache, and 131K+ context on a single RTX 5090.

### Why llama.cpp?

**GGUF Quantization:** Q4_K_M reduces model weights to ~18.6 GiB (vs ~29 GiB for FP8), freeing ~12.3 GiB for KV cache. This enables 131K context on 32GB VRAM — previously impossible with vLLM's FP8 approach.

**Quantized KV Cache:** Supports per-component KV quantization (`--cache-type-k q8_0 --cache-type-v q8_0`) at ~52 KiB/token, dramatically extending context capacity with negligible quality loss (+0.004 perplexity for Q8_0 keys).

**Flash Attention:** Native flash attention support (`--flash-attn`) for efficient long-context inference.

**Tool Calling:** Jinja template support (`--jinja`) enables structured tool calling compatible with OpenAI API format.

**KV Cache Prefix Sharing:** Multi-slot configuration (`-np 2 -sps 0.5`) enables KV cache reuse across concurrent requests that share system prompt prefixes, eliminating redundant prefill computation.

**Performance:** ~234 tok/s generation, ~110 tok/s at 32K context on RTX 5090 with Q4_K_M weights.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Windows Host                                               │
│                                                             │
│  ┌─────────────────┐                                       │
│  │  DyTopo Agent 1 │────┐                                  │
│  └─────────────────┘    │                                  │
│  ┌─────────────────┐    │                                  │
│  │  DyTopo Agent 2 │────┤                                  │
│  └─────────────────┘    │                                  │
│  ┌─────────────────┐    │                                  │
│  │  DyTopo Agent 3 │────┤                                  │
│  └─────────────────┘    │                                  │
│         ...             │                                  │
│  ┌─────────────────┐    │                                  │
│  │  DyTopo Agent 8 │────┘                                  │
│  └─────────────────┘    │                                  │
│           ↓             │                                  │
│     Parallel HTTP/1.1   │                                  │
│           ↓             │                                  │
│    localhost:8008       │                                  │
│           ↓             │                                  │
└───────────┼─────────────────────────────────────────────────┘
            │ Docker Port Mapping (8008:8080)
            ↓
┌─────────────────────────────────────────────────────────────┐
│  Docker Network: anyloom                                   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  llama.cpp Container (server-cuda-blackwell)         │   │
│  │  CUDA 12.8 + sm_120 (Blackwell native)              │   │
│  │  Internal Port: 8080                                │   │
│  │  Host Port: 8008                                    │   │
│  │                                                     │   │
│  │  ┌──────────────────────────────────────────────┐  │   │
│  │  │  GGUF Model Loader                           │  │   │
│  │  │  • Q4_K_M quantized weights (~18.6 GiB)      │  │   │
│  │  │  • Full GPU offload (48 layers)               │  │   │
│  │  └──────────────────────────────────────────────┘  │   │
│  │                                                     │   │
│  │  ┌──────────────────────────────────────────────┐  │   │
│  │  │  Quantized KV Cache                          │  │   │
│  │  │  • K: Q8_0, V: Q8_0 (~52 KiB/token)          │  │   │
│  │  │  • ~12.3 GiB budget → 131K+ context          │  │   │
│  │  └──────────────────────────────────────────────┘  │   │
│  │                                                     │   │
│  │  ┌──────────────────────────────────────────────┐  │   │
│  │  │  Qwen3-30B-A3B (Q4_K_M GGUF, ~18.6 GiB)     │  │   │
│  │  │  • GPU: RTX 5090 (32GB VRAM)                  │  │   │
│  │  │  • Context: 131K tokens                       │  │   │
│  │  └──────────────────────────────────────────────┘  │   │
│  │                                                     │   │
│  │  OpenAI-Compatible API: /v1/chat/completions       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Docker Desktop with GPU support (Docker 25.0+ with NVIDIA Container Toolkit)
- NVIDIA GPU with CUDA capability 7.0+ (RTX 2000 series or newer)
- NVIDIA Driver 570+ on Windows host (570.65+ required for Blackwell/RTX 5090)
- 32GB+ system RAM recommended
- ~20GB free disk space for GGUF model file

### Blackwell (RTX 5090) — Build Local Image

The official Docker image ships CUDA 12.4 which lacks Blackwell (sm_120) support. RTX 5090 falls back to PTX JIT compilation (~3-4 tok/s instead of ~50-60 tok/s). Build a local image:

```bash
bash scripts/build_llm_image.sh
```

This clones llama.cpp, builds with CUDA 12.8 + sm_120, and tags as `local/llama.cpp:server-cuda-blackwell`. The docker-compose.yml uses this image by default (configurable via `LLM_IMAGE` in `.env`).

Key build flags:
- `CUDA_VERSION=12.8.0` — first toolkit with sm_120 support
- `CUDA_DOCKER_ARCH=120` — native Blackwell kernels (not 120a, which needs CUDA 13.x)
- `GGML_CUDA_NO_PINNED=1` — avoids GDDR7 pinned-memory issues on RTX 5090
- `GGML_CUDA_FORCE_CUBLAS=ON` — fixes prompt processing bug on Blackwell (0.27 t/s without it)
- `GGML_CUDA_FA_ALL_QUANTS=ON` — enables sub-f16 KV cache with flash attention
- `GGML_CUDA_GRAPHS=ON` — batches kernel launches (up to 1.2x speedup)
- `GGML_FLASH_ATTN=ON` — enables flash attention kernels (~27% speedup on Blackwell)

### GGUF Model Download

Download the model before starting the container:

```bash
# Install huggingface-cli if needed
pip install huggingface-hub

# Download Q4_K_M GGUF (~18.6 GB)
huggingface-cli download unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF \
  Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
  --local-dir ./models
```

### Automated Setup

```bash
# From project root
docker compose up -d
```

This command:
1. Uses `local/llama.cpp:server-cuda-blackwell` image (or fallback via `LLM_IMAGE` env var)
2. Configures GPU passthrough via NVIDIA runtime
3. Sets `ipc: host` for CUDA graph optimization (shared memory access)
4. Sets `GGML_CUDA_GRAPH_OPT=1` environment variable for kernel launch batching
5. Loads `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` from the mounted models volume
6. Starts llama.cpp server on port 8080 (container) / 8008 (host)
7. Joins the `anyloom` Docker network

### AnythingLLM Docker Settings

The AnythingLLM container (`anyloom-anythingllm`) requires several Docker-level settings to run reliably:

| Setting | Value | Purpose |
|---------|-------|---------|
| `shm_size` | `"1g"` | Chrome uses `/dev/shm` for IPC. Docker's default 64MB shared memory causes `BUS_ADRERR` crashes when AnythingLLM's embedded Chromium allocates IPC buffers. 1GB provides comfortable headroom. |
| `init: true` | — | Runs `tini` as PID 1, which reaps zombie Chrome processes. Without this, Node.js runs as PID 1 and does not call `wait()` on child processes, leaving defunct Chromium workers accumulating. |
| `ANYTHINGLLM_CHROMIUM_ARGS` | `--no-sandbox,--disable-setuid-sandbox,--disable-dev-shm-usage,--disable-gpu,--no-zygote,--disable-software-rasterizer` | Eliminates sandbox/seccomp crashes without granting the container `SYS_ADMIN` capability. `--no-sandbox` and `--disable-setuid-sandbox` bypass Chrome's user namespace sandbox (which requires privileges unavailable in Docker). `--disable-dev-shm-usage` forces Chrome to use `/tmp` instead of `/dev/shm` as a secondary mitigation. `--no-zygote` and `--disable-software-rasterizer` prevent additional process-spawning and GPU-access failures in headless mode. |

---

## Configuration

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` / `-m` | Required | Path to GGUF model file |
| `--port` | 8080 | API server port (container internal) |
| `--host` | 127.0.0.1 | Bind address (use `0.0.0.0` for Docker) |
| `--ctx-size` / `-c` | 512 | Context length (tokens) |
| `--n-gpu-layers` / `-ngl` | 0 | Number of layers to offload to GPU (use 99 for full offload) |
| `--flash-attn` / `-fa` | auto | Enable flash attention (`on`, `off`, `auto`) |
| `--cache-type-k` | f16 | KV cache key type (f16, q8_0, q4_0) |
| `--cache-type-v` | f16 | KV cache value type (f16, q8_0, q4_0) |
| `--jinja` | false | Enable Jinja templates for tool calling |
| `--threads` / `-t` | auto | Number of CPU threads (generation phase) |
| `--threads-batch` / `-tb` | same as --threads | Number of CPU threads (prompt processing phase) |
| `--batch-size` / `-b` | 2048 | Prompt processing batch size (tokens per iteration) |
| `--ubatch-size` / `-ub` | 512 | Physical batch size per GPU kernel launch (must be ≤ batch-size) |
| `--slots` | false | Enable slot management API (`/slots` endpoint) |
| `-np` / `--parallel` | 1 | Number of parallel slots (enables concurrent requests + prefix sharing) |
| `-sps` / `--slot-prompt-similarity` | 0.0 | Auto-assign requests to slots with N+ prefix overlap (KV cache reuse) |
| `--alias` | model name | Override model name in `/v1/models` responses (e.g., `gpt-4`) |
| `--fit` / `--fit-target` | off | MoE-aware tensor placement (`on` enables automatic expert-aware GPU layer distribution, 20%+ gains for MoE models) |
| `GGML_CUDA_GRAPH_OPT` | 0 | Environment variable: set to `1` to enable CUDA graph optimization (batches kernel launches, requires `ipc: host` in Docker) |

### AnyLoom Default Configuration

Docker Compose command flags:

```
--model /models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
--alias gpt-4
--host 0.0.0.0
--port 8080
--ctx-size 131072
--n-gpu-layers 99
--flash-attn on
--fit on
--cache-type-k q8_0
--cache-type-v q8_0
--batch-size 8192
--ubatch-size 4096
--threads 8
--threads-batch 16
--jinja
--slots
-np 2
-sps 0.5
```

**Rationale:**
- **131K context:** Full model capability enabled by Q4_K_M weights + quantized KV cache
- **Q4_K_M weights:** ~18.6 GiB in VRAM, leaving ~12.4 GiB for KV cache
- **Q8_0/Q8_0 KV cache:** ~52 KiB/token — fits 131K tokens in ~6.7 GiB (q4_0 values can cause broken output; q8_0 is safe default)
- **Flash attention:** Reduces memory overhead for long sequences
- **--fit on:** MoE-aware tensor placement — automatically distributes expert layers for optimal GPU utilization, 20%+ gains for MoE models like Qwen3-30B-A3B
- **GGML_CUDA_GRAPH_OPT=1:** CUDA graph optimization batches kernel launches, reducing CPU-GPU synchronization overhead (up to 1.2x speedup)
- **--alias gpt-4:** llama.cpp reports as `gpt-4` in `/v1/models` responses, which maps cleanly to `cl100k_base` in AnythingLLM's tiktoken tokenizer. Without this, AnythingLLM logs "Unknown model" errors and falls back to inaccurate token counting.
- **Speculative decoding disabled:** N-gram speculation (`--spec-type ngram-mod`) was removed because it produces ~0% acceptance rate on agent/tool-call workloads. JSON outputs are novel tokens that never match input n-grams, so the speculator drafts tokens that are always rejected. This was confirmed by `accept: low acceptance streak` messages in LLM logs. The draft overhead (48-64 tokens verified then discarded) actually *reduced* throughput vs. standard autoregressive decoding.
- **Prompt caching re-enabled:** `--no-cache-prompt` was removed. It was only needed as a workaround for llama.cpp bug [#19231](https://github.com/ggerganov/llama.cpp/issues/19231) where speculative decoding broke after the first request. With spec decoding disabled, prompt caching works correctly and significantly improves prefill performance for repeated system prompts.
- **ipc: host:** Required Docker setting for CUDA graph optimization — allows the container to use shared memory for inter-process communication with the GPU driver
- **Full GPU offload:** All 48 layers on GPU for maximum throughput
- **Jinja templates:** Enables structured tool calling for agent workflows
- **batch-size 8192:** 4× default (2048). Number of tokens queued per prompt processing iteration. With ~7 GiB VRAM headroom after weights + KV cache, 8192 fits comfortably. Larger batch = fewer iterations to process long prompts.
- **ubatch-size 4096:** 8× default (512). Physical tokens per GPU kernel launch. This is the single most impactful prefill optimization — larger ubatch means the GPU processes more tokens per CUDA kernel invocation, dramatically improving compute utilization.
- **threads 8:** CPU threads for token generation. With full GPU offload, the GPU handles all matrix math; CPU threads only manage tokenization and data marshaling. 8 threads balances throughput with two parallel slots active.
- **threads-batch 16:** CPU threads for prompt processing phase (separate from generation). Set to physical core count (Ryzen 9 9950X3D = 16 cores) for maximum throughput during the data pipeline that feeds the GPU.
- **slots:** Enables the `/slots` API endpoint for monitoring slot usage and KV cache state.
- **-np 2 (parallel slots):** Two parallel slots enable concurrent request processing and KV cache prefix sharing between slots. When two requests share the same system prompt prefix (e.g., AnythingLLM's ~21K character prompt), the second request reuses the cached KV state from the first slot instead of recomputing it.
- **-sps 0.5 (slot prompt similarity):** Auto-assigns incoming requests to the slot with the highest prefix overlap, provided at least 50% of the prompt prefix matches. This maximizes KV cache reuse across consecutive requests with shared system prompts.

### Prefill Optimization Notes

The AnythingLLM system prompt (~21K characters) is sent with every request. The llama.cpp server automatically caches KV state per slot — when consecutive requests share a prompt prefix, cached tokens are skipped (visible in response `timings.cache_n`). This is NOT the `--prompt-cache` CLI flag (which is CLI-only, not available in server mode). The server's automatic prefix matching provides equivalent benefits without configuration.

With `-np 2` and `-sps 0.5`, prefix sharing now works **across slots**: when a new request arrives, the server checks all slots for prefix overlap and routes to the best match. This means two concurrent DyTopo agents sharing the same system prompt can both benefit from cached prefill, rather than only sequential requests to the same slot.

**Note on speculative decoding:** N-gram speculation (`--spec-type ngram-mod`) was previously enabled but has been removed. While n-gram speculation is effective for repetitive natural-language output, AnyLoom's agent workloads primarily produce JSON tool calls and structured data — novel tokens that never appear in the input context. The n-gram speculator achieved ~0% acceptance rate (visible as `accept: low acceptance streak` in server logs), meaning all drafted tokens were discarded. The verification overhead of 48-64 rejected draft tokens per step actually degraded throughput compared to standard autoregressive decoding.

Continuous batching (`--cont-batching`) is enabled by default in modern llama.cpp server builds and does not need to be specified explicitly.

---

## VRAM Budget

| Component | VRAM |
|---|---|
| Q4_K_M model weights | ~18.6 GiB |
| llama.cpp overhead | ~1.0 GiB |
| KV cache (K:Q8_0 V:Q8_0) | ~12.3 GiB |
| **Total** | **~31.9 GiB** |

### GGUF Quantization Options

| Quantization | Weight Size | Available KV | Max Context (FP16 KV) | Max Context (Q8/Q4 KV) |
|---|---|---|---|---|
| Q4_K_M | 18.6 GiB | ~12.4 GiB | ~132K | ~325K |
| Q5_K_M | 21.7 GiB | ~9.3 GiB | ~99K | ~244K |
| Q6_K | 25.1 GiB | ~5.9 GiB | ~63K | ~155K |
| Q8_0 | 32.5 GiB | Won't fit | — | — |

Q4_K_M is the recommended quantization for RTX 5090 (32GB). Quality impact is minimal: +0.05 perplexity vs FP16 weights. Combined with Q8_0 key cache (+0.004 perplexity), total quality loss is negligible for instruction-following and tool-calling tasks.

---

## API Usage

llama.cpp server provides an OpenAI-compatible API at `http://localhost:8008/v1`.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completion (primary) |
| `/v1/completions` | POST | Text completion |
| `/health` | GET | Health check |

### Example: Chat Completion

```python
import openai

client = openai.AsyncOpenAI(
    base_url="http://localhost:8008/v1",
    api_key="EMPTY"  # llama.cpp doesn't require auth
)

response = await client.chat.completions.create(
    model="Qwen3-30B-A3B-Instruct-2507",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is llama.cpp?"}
    ],
    temperature=0.3,
    max_tokens=2048
)

print(response.choices[0].message.content)
```

### DyTopo Integration

DyTopo's `InferenceClient` wraps llama.cpp with:
- Connection pooling (20 keepalive, 100 max connections)
- Retry with exponential backoff (3 attempts, 1s→8s)
- Semaphore-controlled concurrency (8 concurrent)
- Per-agent token tracking

---

## Performance Characteristics

### Throughput

| Context Length | Generation (tok/s) | Notes |
|---|---|---|
| Short (<2K) | ~234 | Peak generation speed |
| 8K | ~200 | Comfortable operating range |
| 32K | ~110 | Typical agentic workload |
| 64K | ~80 | Long document analysis |
| 131K | ~50 | Maximum context |

> **Hardware:** RTX 5090, Qwen3-30B-A3B Q4_K_M GGUF, llama.cpp server-cuda, --flash-attn, K:Q8_0/V:Q8_0

### Memory Usage

**VRAM breakdown:**

| Component | Size |
|---|---|
| Q4_K_M weights (all 128 experts) | ~18.6 GiB |
| llama.cpp overhead | ~1.0 GiB |
| KV cache @ 131K (Q8/Q8) | ~6.7 GiB |
| **Total** | **~26.3 GiB / 32 GiB** |

> At full 131K context with Q8_0/Q8_0 KV cache, approximately 5.7 GiB of VRAM headroom remains. Still fits comfortably in 32GB VRAM with 18.6 GiB weights.

### Configuration Options

| Configuration | Value | Trade-off |
|---|---|---|
| **Context Length** | 32K / 64K / 131K | Higher = more KV cache memory, longer prompt processing |
| **KV Cache Type** | Q8_0/Q8_0 (default) / FP16 | Quantized = ~3× more context capacity, negligible quality loss |
| **Quantization** | Q4_K_M (default) / Q5_K_M / Q6_K | Higher = better quality, less room for KV cache |
| **Flash Attention** | On (default) / Off | On = lower memory overhead, faster long-context |
| **Parallel Slots** | 1-8 | More slots = better throughput, more memory per slot |

---

## Troubleshooting

### Server Not Reachable

**Symptom:** `Connection refused to localhost:8008`

**Solutions:**
1. Verify container is running:
   ```bash
   docker compose ps
   ```

2. Check container logs:
   ```bash
   docker compose logs llm
   ```

3. Verify port mapping:
   ```bash
   docker compose port llm 8080
   # Should output: 0.0.0.0:8008
   ```

4. Restart the container:
   ```bash
   docker compose restart llm
   ```

### CUDA Out of Memory

**Symptom:** `CUDA out of memory` or model fails to load

**Root cause:** The GGUF model and KV cache together exceed available VRAM.

**Solutions (in order of impact):**

1. **Use a smaller quantization** (most impact):
   - Q4_K_M (~18.6 GiB) fits comfortably on 32GB
   - Q5_K_M (~21.7 GiB) fits on 32GB with reduced context
   - Q6_K (~25.1 GiB) tight on 32GB

2. **Reduce context length** (`--ctx-size`):
   ```
   --ctx-size 65536   # Half context, half KV cache
   --ctx-size 32768   # Quarter context
   ```

3. **Enable quantized KV cache** (if not already):
   ```
   --cache-type-k q8_0 --cache-type-v q8_0
   ```

4. **Enable flash attention** (if not already):
   ```
   --flash-attn on
   ```

5. Apply changes:
   ```bash
   docker compose down
   docker compose up -d
   ```

### Model Loading Fails

**Symptom:** `failed to load model` or `invalid model file`

**Solutions:**
1. Verify GGUF file exists and is complete:
   ```bash
   ls -lh models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
   # Should be ~18.6 GB
   ```

2. Re-download if corrupted:
   ```bash
   huggingface-cli download unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF \
     Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
     --local-dir ./models --force-download
   ```

3. Verify model is mounted in Docker:
   ```bash
   docker compose exec llm ls -lh /models/
   ```

### Slow First Request

**Symptom:** First request after startup takes 10-30 seconds

**Cause:** Model loading and CUDA kernel warmup.

**Solution:** This is expected. Subsequent requests will be fast. Wait for health check to pass:
```bash
curl http://localhost:8008/health
```

### GPU Not Available in Container

**Symptom:** Container logs show "No CUDA GPUs detected" or fails to start

**Solutions:**
1. Verify NVIDIA Container Toolkit is installed:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
   ```

2. Check Docker Desktop GPU settings:
   - Open Docker Desktop → Settings → Resources → GPUs
   - Enable GPU support

3. Verify NVIDIA driver on Windows:
   ```powershell
   nvidia-smi
   # Should be 535+ for CUDA 12 support
   ```

4. Restart Docker Desktop after enabling GPU support

---

## Integration with DyTopo

### Backend Configuration

DyTopo's configuration file (`dytopo_config.yaml`) specifies the llama.cpp backend:

```yaml
llm:
  base_url: "http://localhost:8008/v1"
  model: "qwen3-30b-a3b-instruct-2507"

concurrency:
  backend: "llama-cpp"
  max_concurrent: 8
```

### InferenceClient Usage

The `InferenceClient` connects to llama.cpp for all agent inference:

```python
from inference.client import get_client, reset_client
from dytopo.config import load_config

cfg = load_config()
client = get_client(cfg)

result = await client.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.3,
    max_tokens=2048,
    agent_id="agent_1"
)

print(f"Backend: {result.backend}")  # "llama-cpp"
print(f"Tokens: {result.tokens_in}/{result.tokens_out}")
```

---

## Knowledge Layer Architecture

Information flowing through AnyLoom falls into three categories, each matched to a different access mechanism:

1. **Archival knowledge (RAG):** Documents that change rarely — API documentation, architectural notes, research papers. Stored in Qdrant, embedded with BGE-M3. The retrieval pipeline surfaces these as context during chat and agent execution.

2. **Live data (Skills/MCP tools):** Information with a short shelf life — market prices, weather, live API responses. Custom agent skills or MCP servers fetch this on demand at query time. Examples: `asset-price` skill (stocks, crypto, commodities, indices), `mcp-server-fetch` (web page content).

3. **Computed artifacts (Agent work):** Information created by agents during execution — summaries, code, analysis. DyTopo swarm round outputs are computed artifacts that may feed into subsequent rounds.

**Principle:** Match the knowledge category to the access mechanism. Archival data belongs in RAG. Live data belongs behind a skill or MCP tool. Computed artifacts are agent work products.

**Custom skill vs MCP server decision:** Start with a custom skill for single HTTP calls with simple argument parsing. Migrate to an MCP server if the capability needs caching, Python dependencies, or access from both AnythingLLM and the Python orchestrator.

---

## Skill Output Conventions

Custom agent skills and MCP tools should follow these output conventions for consistent agent consumption:

**JSON envelope:**

```json
{"status": "success", "source": "api-name", "data": {"..."}}
```

```json
{"status": "error", "source": "api-name", "error": "message with recovery guidance"}
```

**Size limits:** Truncate output to 4,000-8,000 characters. Each AIbitat agent step carries 5,600-7,600 input tokens of overhead, so large payloads crowd out the context window quickly.

**Error messages:** Include actionable recovery guidance, not just status codes. For example, `"error": "Yahoo Finance returned 429 — too many requests, retry in 60 seconds or try a different ticker symbol"` is preferable to `"error": "HTTP 429"`.

---

## Model Compatibility

### Tested Models

| Model | Size | Quantization | Status | Notes |
|---|---|---|---|---|
| Qwen3-30B-A3B-Instruct-2507 | 30.5B | Q4_K_M | ✅ Default | MoE, ~18.6 GiB, 131K context |
| Qwen3-30B-A3B-Instruct-2507 | 30.5B | Q5_K_M | ✅ Supported | Higher quality, ~21.7 GiB |
| Qwen3-Coder-30B-A3B-Instruct | 30.5B | Q4_K_M | ✅ Compatible | Coding-optimized variant |

### Model Format Requirements

- **Supported:** GGUF files (`.gguf`)
- **Quantizations:** Q4_K_M (default), Q5_K_M, Q6_K, Q4_0, Q5_0, Q8_0
- **Not supported:** Native PyTorch formats (`.safetensors`, `.bin`) — use vLLM for those

---

## Maintenance

### Updating llama.cpp

```bash
# Rebuild local Blackwell image from latest source
bash scripts/build_llm_image.sh

# Restart with new image
docker compose down
docker compose up -d
```

To fall back to the official image (non-Blackwell): set `LLM_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda` in `.env`.

Check release notes for breaking changes: https://github.com/ggerganov/llama.cpp/releases

### Model Management

Models are stored in a mounted volume. To switch models:

1. Download the new GGUF file to `./models/`
2. Update the model path in `docker-compose.yml`
3. Restart: `docker compose down && docker compose up -d`

### Log Management

View real-time logs:
```bash
docker compose logs -f llm
```

Save logs to file:
```bash
docker compose logs llm > llm.log
```

---

## Historical Note

AnyLoom previously used vLLM with FP8 quantization. This was not viable on RTX 5090 (32GB): FP8 model weights consumed ~29 GiB, leaving negative VRAM for KV cache. The migration to llama.cpp with Q4_K_M GGUF reduced weight size to ~18.6 GiB, enabling 131K+ context where previously only 32K was possible (and even that was marginal).

---

## Resources

- **GitHub:** https://github.com/ggerganov/llama.cpp
- **GGUF Models:** https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF
- **Server Documentation:** https://github.com/ggerganov/llama.cpp/tree/master/examples/server

---

**Last Updated:** 2026-02-17
**AnyLoom Version:** 1.0
**llama.cpp Image:** local/llama.cpp:server-cuda-blackwell (CUDA 12.8, sm_120)
