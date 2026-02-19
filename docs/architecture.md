# Architecture

## System Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     Docker Network: anyloom                      │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  anyloom-llm                                               │  │
│  │  Qwen3-30B-A3B-Instruct-2507 (Q4_K_M GGUF)               │  │
│  │                                                            │  │
│  │  Model weights (Q4_K_M):      ~18.6 GiB VRAM              │  │
│  │  KV cache (131K, Q8/Q8):      ~7.0 GiB VRAM              │  │
│  │  llama.cpp overhead:          ~1.0 GiB VRAM               │  │
│  │  ─────────────────────────────────────────────             │  │
│  │  Total:                       ~26.6 GiB / 32 GiB          │  │
│  │                                                            │  │
│  │  Container port: 8080                                      │  │
│  │  Host port: 8008                                           │  │
│  │  Internal: http://anyloom-llm:8080/v1                      │  │
│  │  External: http://localhost:8008/v1                         │  │
│  │  Parallel slots: 2                                         │  │
│  └────────────────┬───────────────────────────────────────────┘  │
│                   │                                              │
│  ┌────────────────▼───────────────┐  ┌────────────────────────┐  │
│  │  anyloom-anythingllm           │  │  anyloom-qdrant        │  │
│  │  Port: 3001                    │  │  Port: 6333 (REST)     │  │
│  │                                │  │  Port: 6334 (gRPC)     │  │
│  │  LLM: anyloom-llm:8080        │  │                        │  │
│  │  VectorDB: anyloom-qdrant:6333 │  │  Hybrid dense+sparse   │  │
│  │  Embed: anyloom-embedding:8080 │  │  RRF fusion            │  │
│  │                                │  │  source_dir filtering  │  │
│  │  Workspace RAG:                │◀─┤                        │  │
│  │   2500-char chunks             │  │                        │  │
│  │   (~625 tokens)                │  │                        │  │
│  │   16 snippets, Low sim         │  │                        │  │
│  │   30 msg history               │  │                        │  │
│  │                                │  │                        │  │
│  │  MCP servers (6, stdio):       │  │  Collections:          │  │
│  │   fetch, memory, tavily,       │  │   anythingllm (RAG)    │  │
│  │   context7, filesystem,        │  │   swarm_memory         │  │
│  │   sequential-thinking          │  │   swarm_traces         │  │
│  └────────────────────────────────┘  └────────────────────────┘  │
│                                                                   │
│  ┌────────────────────────────────┐                              │
│  │  anyloom-embedding             │                              │
│  │  BGE-M3 Q8_0 GGUF (GPU)       │                              │
│  │  Port: 8009 (host) / 8080     │                              │
│  │  1024-dim dense, 8192 ctx/slot │                              │
│  │  ~635 MB VRAM, 2 parallel slots│                              │
│  └────────────────────────────────┘                              │
└──────────────────────────────────────────────────────────────────┘
                                │
                                │ localhost:8008, :8009, :6333
                                ▼
         ┌──────────────────────────────────────────────┐
         │  Host (Native Python)                        │
         │                                              │
         │  MCP: qdrant_mcp_server.py (8 tools)         │
         │   • rag_search, rag_status, rag_reindex,     │
         │     rag_sources, rag_file_info               │
         │   • swarm_start, swarm_status, swarm_result  │
         │   • BGE-M3 ONNX INT8 on CPU (~0.6 GB RAM)   │
         │   • Hybrid dense+sparse RRF search           │
         │                                              │
         │  MCP: system_status_mcp.py (6 tools)         │
         │   • service_health, qdrant_collections,      │
         │     gpu_status, llm_slots,                   │
         │     docker_status, stack_config              │
         │                                              │
         │  DyTopo Package (src/dytopo/):               │
         │   • MiniLM-L6-v2 on CPU (routing, ~80 MB)   │
         │   • Stigmergic router (trace-aware topology) │
         │   • 3 domains, 3-5 rounds, τ=0.5            │
         │   • AsyncOpenAI → localhost:8008             │
         │   • Semaphore-based concurrency (2 slots)    │
         │   • Aegean consensus termination             │
         │                                              │
         │  Health Monitor (scripts/health_monitor.py)  │
         │   • Probes all 4 containers every 30s        │
         │   • Auto-restart via docker restart           │
         │   • Crash window: 3 attempts / 15 min        │
         │   • JSONL logs: ~/anyloom-logs/health.jsonl  │
         └──────────────────────────────────────────────┘
```

## VRAM Budget

### LLM (anyloom-llm)

| Component | Size |
|---|---|
| Qwen3-30B-A3B Q4_K_M weights (all 128 experts) | ~18.6 GiB |
| llama.cpp overhead | ~1.0 GiB |
| **Subtotal (fixed)** | **~19.6 GiB** |

| Context Length | KV Cache (Q8_0 K / Q8_0 V) | Total VRAM | Status |
|---|---|---|---|
| 8K | ~0.4 GiB | ~20.0 GiB | Comfortable |
| 32K | ~1.7 GiB | ~21.3 GiB | Comfortable |
| 64K | ~3.4 GiB | ~23.0 GiB | Comfortable |
| 131K (default) | ~7.0 GiB | ~26.6 GiB | Default — fits 32GB with headroom |

llama.cpp loads the Q4_K_M GGUF model file directly (~18.6 GiB on disk, same in VRAM). KV cache uses quantized types (`--cache-type-k q8_0 --cache-type-v q8_0`) at ~55 KiB/token. Flash attention (`--flash-attn on`) is enabled. Default context is 131K tokens. 2 parallel slots with speculative decoding (`-sps 0.5`).

### Embedding (anyloom-embedding)

| Component | Size |
|---|---|
| BGE-M3 Q8_0 GGUF weights | ~635 MB |
| KV cache (16K ctx, 2 slots) | ~50 MB |
| **Total** | **~685 MB VRAM** |

### CPU-only models (host)

| Model | Used by | RAM |
|---|---|---|
| BGE-M3 ONNX INT8 | MCP qdrant-rag server (hybrid search) | ~0.6 GB |
| MiniLM-L6-v2 | DyTopo routing + stigmergic router | ~80 MB |

## Port Map

| Port | Service | Used By |
|---|---|---|
| 8008 | llama.cpp LLM API (Docker host port) | MCP server inference, DyTopo swarm agent inference, direct API calls from host |
| 8009 | llama.cpp Embedding API (Docker host port) | AnythingLLM chunk embedding (via Docker network anyloom-embedding:8080), health monitor probes |
| 8080 | llama.cpp container port (internal) | AnythingLLM inference (anyloom-llm:8080), embedding (anyloom-embedding:8080) |
| 6333 | Qdrant REST (Docker, single hybrid instance) | AnythingLLM workspace RAG, MCP qdrant-rag server, stigmergic router traces, swarm memory |
| 6334 | Qdrant gRPC (Docker) | Optional gRPC access |
| 3001 | AnythingLLM UI (Docker) | User web interface for workspace chat |

## Component Summary

### Docker containers

**llama.cpp LLM** (`anyloom-llm`) is the sole GPU inference backend for all LLM operations in the stack. It runs on the `anyloom` network with container port 8080 mapped to host port 8008. Q4_K_M GGUF quantization (~18.6 GiB weights + ~7 GiB KV cache q8_0/q8_0 + flash attention = ~26.6 GiB in 32GB VRAM at 131K context). 2 parallel slots with speculative decoding. Handles all LLM inference for AnythingLLM, DyTopo swarm, and direct API calls. No fallback inference backend.

**llama.cpp Embedding** (`anyloom-embedding`) runs BGE-M3 Q8_0 GGUF on GPU (~635 MB VRAM) with container port 8080 mapped to host port 8009. Provides OpenAI-compatible `/v1/embeddings` endpoint with 1024-dim dense vectors, 8192 token context per slot, 2 parallel slots. Used by AnythingLLM for document chunk embedding.

**AnythingLLM** (`anyloom-anythingllm`) runs on port 3001, providing a workspace-based RAG interface with document ingestion. Connects to llama.cpp LLM at `http://anyloom-llm:8080/v1`, embedding at `http://anyloom-embedding:8080/v1`, and Qdrant at `http://anyloom-qdrant:6333`. Chunking: 2500-char chunks (~625 tokens) via EmbeddingModelMaxChunkLength, 16 snippets at low similarity, 30-message history. Runs 6 MCP servers as stdio child processes: fetch, memory, tavily, context7, filesystem, sequential-thinking.

**Qdrant** (`anyloom-qdrant`) runs on ports 6333 (REST) and 6334 (gRPC). Single hybrid instance serving multiple collections: AnythingLLM workspace RAG, `swarm_memory` (DyTopo run persistence, 384-dim), and `swarm_traces` (stigmergic routing traces, 384-dim). Supports hybrid dense+sparse vectors with RRF fusion, payload-indexed by source file and source directory for filtered retrieval.

### Host-side processes

**MCP server — qdrant-rag** (`src/qdrant_mcp_server.py`) runs natively as a Python process. Connects to llama.cpp at `localhost:8008` and Qdrant at `localhost:6333`. Exposes 8 tools: 5 RAG tools (`rag_search`, `rag_status`, `rag_reindex`, `rag_sources`, `rag_file_info`) with hybrid dense+sparse search using BGE-M3 ONNX INT8 on CPU (~0.6 GB RAM, 0 VRAM), and 3 DyTopo tools (`swarm_start`, `swarm_status`, `swarm_result`).

**MCP server — system-status** (`src/mcp_servers/system_status_mcp.py`) runs natively as a Python process. Exposes 6 diagnostic tools: `service_health`, `qdrant_collections`, `gpu_status`, `llm_slots`, `docker_status`, `stack_config`. Reuses the `HealthChecker` class from DyTopo for service probes.

**DyTopo swarm** (`src/dytopo/`) is a dedicated Python package with 8 core modules and 6 sub-packages. Core: `models.py` (Pydantic v2), `config.py` (YAML), `agents.py` (prompts, schemas, domain rosters), `router.py` (MiniLM-L6-v2 cosine similarity routing), `stigmergic_router.py` (trace-aware topology with Qdrant-persisted swarm traces and time-decayed boost matrix), `graph.py` (NetworkX DAG, topological tiers), `orchestrator.py` (async parallel swarm with semaphore concurrency, Aegean consensus termination), `governance.py` (convergence/stalling detection), `audit.py` (JSONL logging). Sub-packages: `observability/`, `safeguards/`, `messaging/`, `routing/`, `delegation/`, `documentation/`.

**Health Monitor** (`scripts/health_monitor.py`) is a standalone Python sidecar (no LLM inference). Probes all 4 Docker containers + GPU every 30 seconds, auto-restarts failed containers via `docker restart`, with crash window protection (3 attempts per 15 minutes) and alert cooldown (30 minutes). Logs structured JSONL to `~/anyloom-logs/health.jsonl`. Configurable via `dytopo_config.yaml` or environment variables.
