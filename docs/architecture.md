# Architecture

## System Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     Docker Network: anyloom                      │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  anyloom-llm                                              │  │
│  │  Qwen3-30B-A3B-Instruct-2507 (Q4_K_M GGUF)          │  │
│  │                                                            │  │
│  │  Model weights (Q4_K_M):      ~18.6 GiB VRAM          │  │
│  │  KV cache (131K, Q8/Q4):     ~5.0 GiB VRAM            │  │
│  │  llama.cpp overhead:          ~1.0 GiB VRAM            │  │
│  │  ─────────────────────────────────────────────          │  │
│  │  Total:                       ~24.6 GiB / 32 GiB       │  │
│  │                                                            │  │
│  │  Container port: 8080                                      │  │
│  │  Host port: 8008                                           │  │
│  │  Internal: http://anyloom-llm:8080/v1                     │  │
│  │  External: http://localhost:8008/v1                        │  │
│  └────────────────┬───────────────────────────────────────────┘  │
│                   │                                              │
│  ┌────────────────▼───────────────┐  ┌────────────────────────┐ │
│  │  anyloom-anythingllm         │  │  anyloom-qdrant      │ │
│  │  Port: 3001                    │  │  Port: 6333 (REST)     │ │
│  │                                │  │  Port: 6334 (gRPC)     │ │
│  │  LLM: anyloom-llm:8080       │  │                        │ │
│  │  VectorDB: anyloom-qdrant:6333│ │  Hybrid dense+sparse   │ │
│  │                                │  │  RRF fusion            │ │
│  │  Workspace RAG:                │  │  source_dir filtering  │ │
│  │   2500-char chunks             │◀─┤                        │ │
│  │   (~625 tokens)                │  │                        │ │
│  │   16 snippets, Low sim         │  │                        │ │
│  │   30 msg history               │  │                        │ │
│  └────────────────────────────────┘  └────────────────────────┘ │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
                                │
                                │ localhost:8008, localhost:6333
                                ▼
         ┌──────────────────────────────────────────────┐
         │  Host (Native Python)                        │
         │                                              │
         │  MCP Server: qdrant_mcp_server.py            │
         │   • Connects to llama.cpp: localhost:8008     │
         │   • Connects to Qdrant: localhost:6333       │
         │                                              │
         │  BGE-M3 Embedding (ONNX INT8, CPU):             │
         │   • Dense + TF-sparse vectors                │
         │   • RRF hybrid search                        │
         │   • ~0.6 GB RAM, 0 VRAM                      │
         │                                              │
         │  DyTopo Package (src/dytopo/):               │
         │   • MiniLM-L6-v2 on CPU (routing)            │
         │   • ~80 MB RAM                               │
         │   • 3 domains, 3-5 rounds                    │
         │   • τ=0.5 threshold                          │
         │   • AsyncOpenAI → localhost:8008             │
         │   • temp 0.1 descriptors                     │
         │   • temp 0.3 work output                     │
         │   • Semaphore-based concurrency              │
         │                                              │
         │  MCP Tools (qdrant-rag server):              │
         │   • swarm_start, swarm_status, swarm_result  │
         │   • Hybrid RAG search operations             │
         └──────────────────────────────────────────────┘
```

## VRAM Budget

| Component | Size |
|---|---|
| Qwen3-30B-A3B Q4_K_M weights (all 128 experts) | ~18.6 GiB |
| llama.cpp overhead | ~1.0 GiB |
| **Subtotal (fixed)** | **~19.6 GiB** |

| Context Length | KV Cache (Q8_0 K / Q4_0 V) | Total VRAM | Status |
|---|---|---|---|
| 8K | ~0.3 GiB | ~19.9 GiB | Comfortable |
| 32K | ~1.2 GiB | ~20.8 GiB | Comfortable |
| 64K | ~2.5 GiB | ~22.1 GiB | Comfortable |
| 131K (default) | ~5.0 GiB | ~24.6 GiB | Default — fits 32GB with headroom |

llama.cpp loads the Q4_K_M GGUF model file directly (~18.6 GiB on disk, same in VRAM). KV cache uses quantized types (--cache-type-k q8_0 --cache-type-v q4_0) at ~39 KiB/token. Flash attention (--flash-attn on) is enabled. Default context is 131K tokens.

BGE-M3 on CPU (ONNX INT8, sentence-transformers, MCP server process): ~0.6 GB RAM, 0 VRAM.
MiniLM-L6-v2 on CPU (sentence-transformers, MCP server process, DyTopo routing): ~80 MB RAM, 0 VRAM.

## Port Map

| Port | Service | Used By |
|---|---|---|
| 8008 | llama.cpp API (Docker host port) | MCP server inference, DyTopo swarm agent inference, direct API calls from host |
| 8080 | llama.cpp API (Docker container port) | AnythingLLM inference (via Docker network anyloom-llm:8080) |
| 6333 | Qdrant REST (Docker, single hybrid instance) | AnythingLLM workspace RAG, MCP qdrant-rag server |
| 6334 | Qdrant gRPC (Docker) | Optional gRPC access |
| 3001 | AnythingLLM UI (Docker) | User web interface for workspace chat |

## Component Summary

llama.cpp is the sole GPU inference backend for all LLM operations in the stack. It runs in Docker container `anyloom-llm` on the `anyloom` network, with container port 8080 mapped to host port 8008. It provides high throughput and parallel execution using Q4_K_M GGUF quantization (~18.6 GiB weights + quantized KV cache + flash attention fits in 32GB VRAM at 131K context). llama.cpp handles all LLM inference for both AnythingLLM and DyTopo swarm agent calls. There is no fallback inference backend.

AnythingLLM runs in Docker container `anyloom-anythingllm` on port 3001, providing a workspace-based RAG interface with its own document ingestion pipeline. It connects to llama.cpp via the Docker network at `http://anyloom-llm:8080/v1` for LLM inference, and stores vectors in the Qdrant instance at `http://anyloom-qdrant:6333`. AnythingLLM uses its own embedding provider (configured in the UI). Its chunking strategy uses 2500-character chunks (~625 tokens), controlled by EmbeddingModelMaxChunkLength (the only effective chunk setting via AnythingLLM's API), retrieving 16 snippets at low similarity threshold with 30-message conversation history.

The MCP server (`qdrant_mcp_server.py`) runs natively on the host as a Python process. It connects to llama.cpp at `http://localhost:8008` and Qdrant at `http://localhost:6333`. The server implements hybrid dense+sparse RAG search with RRF fusion using BGE-M3 ONNX INT8 on CPU (~0.6 GB RAM, 0 VRAM), and DyTopo multi-agent swarm orchestration using MiniLM-L6-v2 on CPU for descriptor routing (~80 MB RAM). DyTopo is structured as a dedicated Python package (`src/dytopo/`) with 8 core modules and 6 sub-packages: core modules include `models.py` (Pydantic v2 data models), `config.py` (YAML configuration), `agents.py` (system prompts, JSON schemas, domain rosters), `router.py` (MiniLM embedding and similarity routing), `graph.py` (NetworkX DAG construction with topological tier computation), `orchestrator.py` (async parallel swarm loop with AsyncOpenAI client and semaphore-based concurrency), `governance.py` (convergence/stalling detection), and `audit.py` (JSONL audit logging); sub-packages provide `observability/` (distributed tracing, metrics, profiling), `safeguards/` (rate limiter, token budget, circuit breaker), `messaging/` (typed agent message passing), `routing/` (async routing engine), `delegation/` (subtask delegation with depth control), and `documentation/` (auto-generated living docs from code and execution data). The MCP server exposes 3 thin MCP tools (`swarm_start`, `swarm_status`, `swarm_result`) that delegate to this package, plus hybrid RAG search operations.

Qdrant runs in Docker container `anyloom-qdrant` on ports 6333 (REST) and 6334 (gRPC). It serves both RAG pipelines as a single hybrid instance. It stores hybrid dense+sparse vectors with RRF fusion, payload-indexed by source file and source directory for filtered retrieval across multiple document sources. AnythingLLM accesses it via the Docker network, while the MCP server connects from the host at `localhost:6333`.
