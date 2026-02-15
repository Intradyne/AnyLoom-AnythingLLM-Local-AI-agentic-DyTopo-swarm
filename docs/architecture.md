# Architecture

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        LM Studio (GPU)                          │
│                                                                 │
│  Qwen3-30B-A3B-Instruct-2507 (Q6_K)    25.1 GB VRAM            │
│  ggml-org/bge-m3 (Q8_0) co-loaded       0.6 GB VRAM            │
│  KV cache (80K context, Q8_0)           ~3.75 GB VRAM           │
│  ─────────────────────────────────────────────────────          │
│  Total:                                ~30.5 GB / 32 GB         │
│                                                                 │
│  Endpoints: http://localhost:1234/v1                             │
│    • /chat/completions (LLM — user chat + DyTopo agent calls)   │
│    • /embeddings (BGE-M3 GGUF, dense-only)                      │
└────────┬──────────────────────────────────┬─────────────────────┘
         │                                  │
         │ tool_call JSON                   │ embeddings API
         ▼                                  ▼
┌──────────────────────────┐      ┌─────────────────────────┐
│  MCP Servers             │      │  AnythingLLM             │
│                          │      │                          │
│  Docker Gateway:         │      │  LLM → LM Studio :1234   │
│   • Desktop Commander    │      │  Embed → LM Studio :1234  │
│   • Filesystem           │      │  VectorDB → Qdrant :6333  │
│   • Memory (local)       │      │                          │
│   • Context7             │      │  Workspace RAG:          │
│   • Tavily               │      │   6600-char chunks       │
│   • Fetch                │      │   1000-char overlap      │
│   • Playwright           │      │   16 snippets, Low sim   │
│   • Sequential Think     │      │   30 msg history         │
│   • n8n                  │      └──────────┬───────────────┘
│                          │                 │
│  Python (native):        │                 ▼
│   • qdrant-rag ──────────┼──┐   ┌──────────────────────┐
│     RAG: BGE-M3 on CPU   │  │   │  Qdrant (Docker)     │
│     dense + sparse, RRF  │  │   │  Port 6333 (REST)    │
│     ~2.3 GB RAM          │  │   │  AnythingLLM instance │
│                          │  │   │  Dense vectors only   │
│     DyTopo Package:      │  │   └──────────────────────┘
│     src/dytopo/ (8 mods) │  │
│     MiniLM-L6-v2 on CPU  │  │   ┌──────────────────────┐
│     ~80 MB RAM           │  └──▶│  Qdrant (Docker)     │
│     3 domains, 3–5 rds   │      │  Port 6334 (REST)    │
│     τ=0.3 threshold      │      │  LM Studio instance   │
└──────────────────────────┘      │  Dense + sparse vecs  │
                                  │  RRF hybrid search    │
         ┌────────────────────────│  source_dir filtering  │
         │  DyTopo agent calls    └──────────────────────┘
         │  (AsyncOpenAI → :1234)
         │  temp 0.1 descriptors
         │  temp 0.3 work output
         └──▶ LM Studio /v1/chat/completions
```

## VRAM Budget

| Component | Size |
|---|---|
| Qwen3-30B-A3B Q6_K weights | 25.1 GB |
| BGE-M3 Q8_0 GGUF (co-loaded) | 0.6 GB |
| CUDA overhead | ~1.0 GB |
| **Subtotal (fixed)** | **~26.7 GB** |

| Context Length | KV Cache (Q8_0) | Total VRAM | Status |
|---|---|---|---|
| 32K | ~1.5 GB | ~28.2 GB | ✅ Comfortable |
| 64K | ~3.0 GB | ~29.7 GB | ✅ Conservative |
| 80K | ~3.75 GB | ~30.5 GB | ✅ Default |
| 96K | ~4.5 GB | ~31.2 GB | ⚠️ Tight |
| 128K | ~6.0 GB | ~32.7 GB | ⚠️ Ceiling — use Q5_K_M weights |

BGE-M3 on CPU (FlagEmbedding, MCP server): 0 GB VRAM, ~2.3 GB system RAM.
MiniLM-L6-v2 on CPU (sentence-transformers, DyTopo routing): 0 GB VRAM, ~80 MB system RAM.

## Port Map

| Port | Service | Used By |
|---|---|---|
| 1234 | LM Studio API | AnythingLLM, direct API calls, DyTopo swarm agent inference |
| 6333 | Qdrant REST (container 1) | AnythingLLM workspace RAG |
| 6334 | Qdrant REST (container 2) | MCP qdrant-rag server |

## Component Summary

LM Studio serves as the GPU inference backbone, hosting both the Qwen3-30B-A3B chat model and the BGE-M3 embedding model on a single RTX 5090. All LLM and embedding requests route through its OpenAI-compatible API at `http://localhost:1234/v1`. The chat model handles direct user conversations, MCP tool-augmented interactions, and DyTopo swarm agent inference calls. The co-loaded embedding model provides dense vectors for AnythingLLM's workspace RAG pipeline.

AnythingLLM provides a workspace-based RAG interface with its own document ingestion pipeline. It connects to LM Studio for both LLM inference and embedding generation, and stores its vectors in a dedicated Qdrant instance on port 6333. Its chunking strategy uses 6600-character chunks with 1000-character overlap, retrieving 16 snippets at low similarity threshold with 30-message conversation history.

The MCP server ecosystem splits across two transports. Nine containerized servers run through the Docker MCP Gateway, providing system commands, file operations, a local knowledge graph, library docs, web search, URL fetching, browser automation, multi-step reasoning, and workflow automation. The native Python `qdrant-rag` server runs directly on the host, implementing hybrid dense+sparse RAG search with RRF fusion using BGE-M3 on CPU (~2.3 GB RAM), and DyTopo multi-agent swarm orchestration using MiniLM-L6-v2 for descriptor routing (~80 MB RAM). DyTopo is structured as a dedicated Python package (`src/dytopo/`) with 8 modules: `models.py` (Pydantic v2 data models), `config.py` (YAML configuration loader), `agents.py` (system prompts, JSON schemas, domain rosters), `router.py` (MiniLM embedding and similarity routing), `graph.py` (NetworkX DAG construction), `orchestrator.py` (main swarm loop with singleton AsyncOpenAI client), `governance.py` (convergence/stalling detection), and `audit.py` (JSONL audit logging). The server exposes 3 thin MCP tools (`swarm_start`, `swarm_status`, `swarm_result`) that delegate to this package.

Two independent Qdrant Docker containers serve the two RAG pipelines. The AnythingLLM instance on port 6333 stores dense-only vectors from AnythingLLM's embedding pipeline. The MCP server instance on port 6334 stores hybrid dense+sparse vectors with RRF fusion, payload-indexed by source file and source directory for filtered retrieval across multiple document sources.
