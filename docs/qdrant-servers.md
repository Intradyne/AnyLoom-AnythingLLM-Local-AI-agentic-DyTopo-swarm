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
      "args": ["C:\\Users\\User\\Qdrant-RAG+Agents\\src\\qdrant_mcp_server.py"],
      "env": {
        "QDRANT_URL": "http://localhost:6334",
        "LMStudio_DOCS_DIR": "C:\\Users\\User\\Qdrant-RAG+Agents\\rag-docs\\lm-studio",
        "AnythingLLM_DOCS_DIR": "C:\\Users\\User\\Qdrant-RAG+Agents\\rag-docs\\anythingllm",
        "COLLECTION_NAME": "lmstudio_docs"
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
| Memory | Shared knowledge graph (entity-relation store, used by both LM Studio and AnythingLLM) |
| Context7 | Library documentation retrieval |
| Tavily | Web search |
| Fetch | URL content retrieval |
| Playwright | Browser automation |
| Sequential Thinking | Multi-step reasoning scratchpad |
| n8n | Workflow automation webhooks |

**Native Python** (`qdrant-rag`) — runs directly on host:

| Server | Purpose |
|---|---|
| qdrant-rag | Hybrid RAG search + DyTopo swarm orchestration. BGE-M3 (RAG) + MiniLM-L6-v2 (routing) on CPU. Auto-indexes from multiple doc sources with per-file incremental sync. |

The `qdrant-rag` server runs natively (not in Docker) because LM Studio spawns MCP servers as child processes via stdio transport. It connects to Qdrant at `http://localhost:6334` and calls LM Studio's LLM at `http://localhost:1234/v1` for DyTopo agent inference.

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

Each MCP tool definition consumes ~300 tokens in the system prompt. With ~15 tools across all servers (9 Docker + 3 built-in LM Studio + 8 qdrant-rag, minus duplicates and disabled), tool definitions total ~6.9K tokens.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6334` | Qdrant REST endpoint |
| `LMStudio_DOCS_DIR` | `./docs` (relative to script dir) | Primary markdown source directory |
| `AnythingLLM_DOCS_DIR` | (empty — disabled) | Secondary markdown source directory |
| `COLLECTION_NAME` | `lmstudio_docs` | Qdrant collection name |
| `RAG_CPU_THREADS` | `8` | OMP/MKL/torch thread count for BGE-M3 |
| `RAG_EMBED_BATCH_SIZE` | `16` | Chunks per embedding batch |
| `RAG_EMBED_MAX_LENGTH` | `1024` | Max tokens per chunk sent to BGE-M3 |
| `RAG_MIN_SCORE` | `0.005` | Minimum RRF score to include in results (0 to disable) |
| `LLM_BASE_URL` | `http://localhost:1234/v1` | LM Studio API endpoint (DyTopo swarm calls) |
| `LLM_MODEL` | `qwen3-30b-a3b-instruct-2507` | Model name for DyTopo API calls |

## Server Location

| Path | Contents |
|---|---|
| `C:\Users\User\Qdrant-RAG+Agents\src\qdrant_mcp_server.py` | MCP server source (RAG + DyTopo) |
| `C:\Users\User\Qdrant-RAG+Agents\rag-docs\lm-studio\` | LM Studio reference documents |
| `C:\Users\User\Qdrant-RAG+Agents\rag-docs\anythingllm\` | AnythingLLM reference documents |
| `C:\Users\User\Qdrant-RAG+Agents\rag-docs\lm-studio\.rag_state.json` | Per-file incremental sync state (v2) |
| `C:\Users\User\.lmstudio\config\mcp.json` | LM Studio MCP server configuration |
| `~/.cache/huggingface/` | BGE-M3 + MiniLM-L6-v2 model weights (auto-downloaded) |

## Embedding Models

| Model | Purpose | Memory | Load time | When active |
|---|---|---|---|---|
| BGE-M3 (FlagEmbedding, CPU) | RAG dense+sparse embedding | ~2.3 GB RAM | ~30–60s | During rag_search / rag_reindex |
| MiniLM-L6-v2 (sentence-transformers, CPU) | DyTopo descriptor routing | ~80 MB RAM | <1s | During swarm rounds 2+ |

Both are lazy-loaded (first use only) and share a dedicated `ThreadPoolExecutor(max_workers=2)` to keep CPU-bound work off the async event loop's default pool. CPU threading is coordinated via `OMP_NUM_THREADS`/`MKL_NUM_THREADS` (default 8 threads — half of 9950X3D's 16 physical cores).
