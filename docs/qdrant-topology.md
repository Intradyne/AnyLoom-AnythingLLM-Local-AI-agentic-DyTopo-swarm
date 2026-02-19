# Qdrant Topology

## Single-Instance Architecture

A single Qdrant Docker container (`anyloom-qdrant`) serves all vector storage needs: AnythingLLM workspace RAG, MCP qdrant-rag server, DyTopo swarm memory persistence, and stigmergic trace routing.

| Instance | Host Port | Container Port | Serves | Collections |
|---|---|---|---|---|
| qdrant | 6333 (REST), 6334 (gRPC) | 6333, 6334 | AnythingLLM + MCP qdrant-rag + DyTopo swarm | `anyloom_docs`, `swarm_memory`, `swarm_traces` + AnythingLLM workspace collections |

The instance serves three embedding models at different dimensionalities:
- **BGE-M3** (1024-dim dense + sparse) — used by both AnythingLLM (via GPU embedding container on port 8009) and MCP qdrant-rag server (via ONNX INT8 on CPU) for document RAG
- **MiniLM-L6-v2** (384-dim) — used by DyTopo for swarm memory persistence and stigmergic trace storage (CPU)

## Docker Configuration

```bash
# Single hybrid instance (port 6333)
docker run -d --name anyloom-qdrant \
  --network anyloom \
  -p 6333:6333 -v anyloom_qdrant_storage:/qdrant/storage \
  --restart always --memory=4g --cpus=4 \
  qdrant/qdrant:latest
```

## Collection Schemas

### `anyloom_docs` (MCP qdrant-rag)

Created and managed by the `qdrant-rag` MCP server:

- **Dense vectors:** Named `"dense"`, 1024-dim, cosine distance (BGE-M3)
- **Sparse vectors:** Named `"sparse"`, inverted index with learned lexical weights (BGE-M3)
- **Quantization:** INT8 scalar (~4× storage reduction, ~95%+ accuracy retention)
- **HNSW index:** m=16, ef_construct=200
- **Payload indexes:** `source` (keyword), `section_header` (keyword), `source_dir` (keyword)
- **Search:** Prefetches 3× candidates per mode, fuses with RRF, filters below MIN_SCORE (default 0.005)

The `source_dir` payload index enables filtered search scoped to a specific doc source (e.g., only AnythingLLM docs).

### `swarm_memory` (DyTopo memory/writer.py)

Persists completed swarm results for semantic retrieval of past solutions:

- **Vectors:** 384-dim, cosine distance (MiniLM-L6-v2)
- **Payload:** `task_summary`, `domain`, `agents`, `rounds`, `key_findings`, `final_answer`, `convergence_status`, `token_count`, `wall_time_ms`, `created_at`
- **Search:** `query_similar(text, limit)` finds past solutions relevant to a new task
- **Created by:** `SwarmMemoryWriter` after successful swarm runs

### `swarm_traces` (DyTopo stigmergic_router.py)

Persists swarm routing patterns for trace-aware topology construction:

- **Vectors:** 384-dim, cosine distance (MiniLM-L6-v2)
- **Payload:** `task_summary`, `task_domain`, `agent_roles`, `active_edges` (with sender/receiver roles and weights), `rounds_to_converge`, `final_answer_quality`, `convergence_method`, `created_at`
- **Quality gate:** Only traces with `final_answer_quality >= min_quality` (default 0.5) are deposited
- **Pruning:** Traces older than `prune_max_age_hours` (default 720h / 30 days) are deleted
- **Created by:** `StigmergicRouter.deposit_trace()` after swarm completion

### AnythingLLM workspace collections

AnythingLLM creates its own collections for workspace RAG (e.g., `anyloom-workspace`). These use 1024-dim BGE-M3 dense vectors from the GPU embedding container (`anyloom-embedding:8080`). Managed entirely by AnythingLLM — not touched by MCP or DyTopo code.

## Multi-Source Indexing

The MCP qdrant-rag server indexes markdown files from configured directories into the `anyloom_docs` collection. Each directory is a "source" with a label. The `source_dir` payload field tracks which directory a chunk came from, enabling filtered search per source.

Source directories:
- `rag-docs/anythingllm/` → label `"anythingllm"`

The per-file incremental sync handles adds, modifications, and deletions independently per source.

## Incremental Sync

The MCP server tracks per-file content hashes in `.rag_state.json` (v2 format with source labels):

```json
{
  "anythingllm/tool-reference.md": "1122334455667788"
}
```

Keys are `source_label/filename`. On each search, the server compares current file hashes against stored state. Only changed, added, or deleted files trigger re-embedding. Modified/deleted files have their chunks purged from Qdrant (filtered by `source` + `source_dir` payload fields) before new chunks are upserted. A single-file edit re-indexes in ~3–10s instead of ~30–60s for the full corpus.
