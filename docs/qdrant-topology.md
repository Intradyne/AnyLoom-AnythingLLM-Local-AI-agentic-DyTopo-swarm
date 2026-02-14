# Qdrant Topology

## Dual-Instance Rationale

Two independent Qdrant Docker containers, one per RAG pipeline.

| Instance | Host Port | Container Port | Serves | Vector Config |
|---|---|---|---|---|
| anythingllm-qdrant | 6333 (REST) | 6333 | AnythingLLM workspace RAG | Dense 1024-dim (cosine) |
| lmstudio-qdrant | 6334 (REST) | 6333 | MCP qdrant-rag server | Dense 1024-dim + sparse lexical (RRF) |

The second container maps host port 6334 to the container's default REST port 6333 via `-p 6334:6333`. Both containers use the standard Qdrant image.

AnythingLLM and the MCP RAG server implement fundamentally different pipelines. AnythingLLM uses workspace-scoped documents, its own SQLite metadata layer, 6600-char chunks with 1000-char overlap, and dense-only vectors. The MCP server uses auto-indexed markdown documents, `##`-header boundary chunking (minimum 80 chars, no fixed upper limit), and hybrid dense+sparse vectors with RRF fusion. Sharing a Qdrant instance would create retrieval pollution — chunks optimized for one pipeline degrading the other.

The only state both agents access is the Memory knowledge graph (MCP server), which stores structured entity-relation data locally, not embedded chunks.

## Docker Configuration

```bash
# AnythingLLM instance (port 6333)
docker run -d --name anythingllm-qdrant \
  -p 6333:6333 -v qdrant_anythingllm:/qdrant/storage \
  --restart always --memory=4g --cpus=4 \
  qdrant/qdrant:latest

# LM Studio / MCP server instance (port 6334)
docker run -d --name lmstudio-qdrant \
  -p 6334:6333 -v qdrant_lmstudio:/qdrant/storage \
  --restart always --memory=4g --cpus=4 \
  qdrant/qdrant:latest
```

## Collection Schema

The `qdrant-rag` MCP server creates and manages the `lmstudio_docs` collection on port 6334:

- **Dense vectors:** Named `"dense"`, 1024-dim, cosine distance
- **Sparse vectors:** Named `"sparse"`, inverted index with learned lexical weights
- **Quantization:** INT8 scalar (~4× storage reduction, ~95%+ accuracy retention)
- **HNSW index:** m=16, ef_construct=200
- **Payload indexes:** `source` (keyword), `section_header` (keyword), `source_dir` (keyword)
- **Search:** Prefetches 3× candidates per mode, fuses with RRF, filters below MIN_SCORE (default 0.005)

The `source_dir` payload index enables filtered search scoped to a specific doc source (e.g., only LM Studio docs, only AnythingLLM docs).

## Multi-Source Indexing

The server indexes markdown files from multiple directories into a single Qdrant collection. Each directory is a "source" with a label. The `source_dir` payload field tracks which directory a chunk came from, enabling filtered search per source.

Configured in `mcp.json` via environment variables:
- `LMStudio_DOCS_DIR` → label `"lmstudio"`
- `AnythingLLM_DOCS_DIR` → label `"anythingllm"`

Both sources are indexed into the same collection (`lmstudio_docs`). The per-file incremental sync handles adds, modifications, and deletions independently per source.

## Incremental Sync

The MCP server tracks per-file content hashes in `.rag_state.json` (v2 format with source labels):

```json
{
  "lmstudio/architecture.md": "a1b2c3d4e5f67890",
  "lmstudio/mcp-servers.md": "0987654321fedcba",
  "anythingllm/tool-reference.md": "1122334455667788"
}
```

Keys are `source_label/filename`. On each search, the server compares current file hashes against stored state. Only changed, added, or deleted files trigger re-embedding. Modified/deleted files have their chunks purged from Qdrant (filtered by `source` + `source_dir` payload fields) before new chunks are upserted. A single-file edit re-indexes in ~3–10s instead of ~30–60s for the full corpus.
