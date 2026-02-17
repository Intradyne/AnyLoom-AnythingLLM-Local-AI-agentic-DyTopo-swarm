# Qdrant Topology

## Single-Instance Architecture

A single Qdrant Docker container serves both AnythingLLM and the MCP qdrant-rag server.

| Instance | Host Port | Container Port | Serves | Vector Config |
|---|---|---|---|---|
| qdrant | 6333 (REST) | 6333 | AnythingLLM + MCP qdrant-rag server | Dense 1024-dim + sparse lexical (RRF) |

The MCP server manages hybrid dense+sparse indexing for multi-source documents (AnythingLLM docs). AnythingLLM connects to the same instance for its workspace RAG, leveraging the hybrid search capabilities for improved retrieval quality. Both systems use BGE-M3 embeddings in the same 1024-dim semantic space, enabling consistent retrieval across all document sources.

## Docker Configuration

```bash
# Single hybrid instance (port 6333)
docker run -d --name anyloom-qdrant \
  --network anyloom \
  -p 6333:6333 -v anyloom_qdrant_storage:/qdrant/storage \
  --restart always --memory=4g --cpus=4 \
  qdrant/qdrant:latest
```

## Collection Schema

The `qdrant-rag` MCP server creates and manages the `anyloom_docs` collection on port 6333:

- **Dense vectors:** Named `"dense"`, 1024-dim, cosine distance
- **Sparse vectors:** Named `"sparse"`, inverted index with learned lexical weights
- **Quantization:** INT8 scalar (~4× storage reduction, ~95%+ accuracy retention)
- **HNSW index:** m=16, ef_construct=200
- **Payload indexes:** `source` (keyword), `section_header` (keyword), `source_dir` (keyword)
- **Search:** Prefetches 3× candidates per mode, fuses with RRF, filters below MIN_SCORE (default 0.005)

The `source_dir` payload index enables filtered search scoped to a specific doc source (e.g., only AnythingLLM docs). This single hybrid instance provides superior retrieval quality for both AnythingLLM workspace queries and MCP RAG tool searches.

## Multi-Source Indexing

The server indexes markdown files from multiple directories into a single Qdrant collection. Each directory is a "source" with a label. The `source_dir` payload field tracks which directory a chunk came from, enabling filtered search per source.

Configured in `mcp.json` via environment variables:
- `AnythingLLM_DOCS_DIR` → label `"anythingllm"`

Both sources are indexed into the same collection (`anyloom_docs`). The per-file incremental sync handles adds, modifications, and deletions independently per source.

## Incremental Sync

The MCP server tracks per-file content hashes in `.rag_state.json` (v2 format with source labels):

```json
{
  "anythingllm/tool-reference.md": "1122334455667788"
}
```

Keys are `source_label/filename`. On each search, the server compares current file hashes against stored state. Only changed, added, or deleted files trigger re-embedding. Modified/deleted files have their chunks purged from Qdrant (filtered by `source` + `source_dir` payload fields) before new chunks are upserted. A single-file edit re-indexes in ~3–10s instead of ~30–60s for the full corpus.
