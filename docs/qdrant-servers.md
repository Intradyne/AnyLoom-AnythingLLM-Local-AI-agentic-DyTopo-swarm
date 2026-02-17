# MCP Servers

## Configuration (mcp.json)

```json
{
  "mcpServers": {
    "MCP_DOCKER": {
      "command": "docker",
      "args": ["mcp", "gateway", "run"],
      "env": {
        "LOCALAPPDATA": "C:\\Users\\User\\AppData\\Local",
        "ProgramData": "C:\\ProgramData",
        "ProgramFiles": "C:\\Program Files"
      }
    },
    "qdrant-rag": {
      "command": "python",
      "args": ["<project-root>/src/qdrant_mcp_server.py"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "AnythingLLM_DOCS_DIR": "<project-root>/rag-docs/anythingllm",
        "COLLECTION_NAME": "anyloom_docs"
      }
    }
  }
}
```

## Server Inventory

**Docker MCP Gateway** (`MCP_DOCKER`) — containerized servers:

| Server | Purpose |
|---|---|
| Desktop Commander | System commands, process management |
| Filesystem | File read/write/search |
| Memory | Local knowledge graph (entity-relation store, used by both llama.cpp agent and AnythingLLM) |
| Context7 | Library documentation retrieval |
| Tavily | Web search |
| Fetch | URL content retrieval |
| Playwright | Browser automation |
| Sequential Thinking | Multi-step reasoning scratchpad |
| n8n | Workflow automation webhooks |

**Native Python** (`qdrant-rag`) — runs directly on host:

| Server | Purpose |
|---|---|
| qdrant-rag | Hybrid RAG search + DyTopo swarm orchestration. BGE-M3 (RAG) + MiniLM-L6-v2 (routing) on GPU. Auto-indexes from multiple doc sources with per-file incremental sync. Serves single Qdrant instance on port 6333. |

The `qdrant-rag` server runs natively (not in Docker) and connects to the Docker services via localhost. It connects to Qdrant at `http://localhost:6333` and calls llama.cpp at `http://localhost:8008/v1` for DyTopo agent inference.

## Tool Inventory

**RAG tools (5):**

| Tool | Args | Description |
|---|---|---|
| `rag_search` | `query`, `limit` (1–10), `source` (optional filename) | Hybrid dense+sparse search with RRF fusion. Optional `source` filter scopes results to a single document. |
| `rag_status` | (none) | Collection info, indexed file list, staleness check, performance config. |
| `rag_reindex` | `force` (optional bool) | Trigger re-index. `force=true` deletes collection and rebuilds from scratch. |
| `rag_sources` | (none) | List configured doc sources and their file counts. |
| `rag_file_info` | `filename` | Per-file chunk count, hash, and source directory. |

**DyTopo swarm tools (3):**

| Tool | Args | Description |
|---|---|---|
| `swarm_start` | `task`, `domain`, `tau`, `k_in`, `max_rounds` | Launch a DyTopo multi-agent swarm as a background task. Returns task_id immediately. |
| `swarm_status` | `task_id` | Check progress of a running swarm (round count, active agent, elapsed time). |
| `swarm_result` | `task_id`, `include_topology` | Retrieve completed swarm result with optional per-round topology log. |

## Token Budget

Each MCP tool definition consumes ~300 tokens in the system prompt. The MCP host loads 10 servers (9 Docker + 1 qdrant-rag). The qdrant-rag server exposes 8 tool endpoints (5 RAG + 3 DyTopo). Total tool-definition overhead is ~6.9K tokens.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST endpoint (single hybrid instance) |
| `AnythingLLM_DOCS_DIR` | `../rag-docs/anythingllm` (relative to script dir) | AnythingLLM reference documents |
| `COLLECTION_NAME` | `anyloom_docs` | Qdrant collection name |
| `RAG_CPU_THREADS` | `8` | OMP/MKL/torch thread count (CPU fallback) |
| `RAG_EMBED_BATCH_SIZE` | `16` | Chunks per embedding batch |
| `RAG_EMBED_MAX_LENGTH` | `1024` | Max tokens per chunk sent to BGE-M3 |
| `RAG_MIN_SCORE` | `0.005` | Minimum RRF score to include in results (0 to disable) |
| `LLM_BASE_URL` | `http://localhost:8008/v1` | llama.cpp API endpoint (DyTopo swarm calls, default) |
| `LLM_MODEL` | `qwen3-30b-a3b-instruct-2507` | Model name for DyTopo API calls |

## Server Location

| Path | Contents |
|---|---|
| `src/qdrant_mcp_server.py` | MCP server source (RAG + DyTopo) |
| `rag-docs/anythingllm/` | AnythingLLM reference documents |
| MCP-compatible host config | MCP server configuration (location varies by MCP host implementation) |
| `~/.cache/huggingface/` | BGE-M3 + MiniLM-L6-v2 model weights (auto-downloaded) |

## Embedding Models

| Model | Purpose | Memory | Load time | When active |
|---|---|---|---|---|
| BGE-M3 (ONNX INT8, CPU) | RAG dense+sparse embedding | ~0.6 GB RAM | ~10–20s | During rag_search / rag_reindex |
| MiniLM-L6-v2 (sentence-transformers, CPU) | DyTopo descriptor routing | ~80 MB RAM | <1s | During swarm rounds 2+ |

Both are lazy-loaded (first use only) and share a dedicated `ThreadPoolExecutor(max_workers=2)` to keep CPU-bound work off the async event loop's default pool. CPU threading is coordinated via `OMP_NUM_THREADS`/`MKL_NUM_THREADS` and ONNX session options (default 16 threads — all of 9950X3D's 16 physical cores, since embedding is CPU-only with no GPU contention).
