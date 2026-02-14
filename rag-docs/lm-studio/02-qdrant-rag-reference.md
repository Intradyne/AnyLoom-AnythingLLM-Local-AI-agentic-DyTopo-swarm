# Qdrant RAG Pipeline Reference

## How does hybrid search work in the RAG pipeline?

<!-- Verified: 2026-02-13 -->

The MCP qdrant-rag server on port 6334 implements hybrid search by combining dense semantic vectors and sparse lexical vectors with Reciprocal Rank Fusion (RRF fusion). The BGE-M3 FlagEmbedding model running on CPU generates two vector types for each chunk: a dense 1024-dimensional embedding that captures semantic meaning, and a sparse lexical vector where individual tokens receive learned importance weights for exact term matching. Both vectors are stored as named vectors in the Qdrant collection called `lmstudio_docs`. At query time, `rag_search` encodes the query into both dense and sparse representations, runs parallel searches against both named vector fields, and fuses the ranked results using RRF fusion to produce a single ranked list. This means queries benefit from both semantic understanding (dense vectors find conceptually similar content) and exact term matching (sparse lexical vectors boost chunks containing literal strings like `6334`, `rag_search`, or `QDRANT_URL`). The minimum score threshold is 0.005, configurable via the `RAG_MIN_SCORE` environment variable.

<!-- Related: hybrid search, RRF fusion, dense vectors, sparse lexical vectors, BGE-M3, Qdrant, port 6334, lmstudio_docs, semantic search, keyword matching -->

## How are documents chunked for indexing?

<!-- Verified: 2026-02-13 -->

The qdrant-rag server splits markdown documents using deterministic ##-header chunking. The algorithm extracts the document's `# Title` from the first `# ` header, then splits the content on every `## ` header boundary using a regex lookahead. Each `## ` section becomes one chunk stored as a single point in Qdrant. Sections shorter than 80 characters are discarded to avoid trivially small fragments. The `# Title` is prepended to every chunk as a context anchor, so each chunk reads as `# {doc_title}\n\n## {section_header}\n{section_content}`. Target chunk size is 800-2,300 characters per `## ` section. Each chunk is stored with metadata: `source` (filename), `source_dir` ("lmstudio" or "anythingllm"), `doc_title`, `section_header`, and `chunk_index`. Chunk IDs are deterministic UUID v5 values generated from the key `"{source_label}/{filename}::{section_header}::{chunk_index}"`, which means re-indexing the same content produces the same IDs and updates in place. HNSW indexing uses m=16 and ef_construct=200, with INT8 scalar quantization at quantile 0.99 for efficient storage.

<!-- Related: chunking, ##-header chunking, markdown splitting, chunk size, Qdrant points, UUID, HNSW, quantization, indexing -->

## How does incremental sync detect and process changed files?

<!-- Verified: 2026-02-13 -->

The qdrant-rag server tracks document state using SHA-256 per-file hashes stored in a `.rag_state.json` file alongside the document source directory. When `rag_search` is called or `rag_reindex` is invoked, the server compares the current SHA-256 hash (truncated to 16 hex characters) of each markdown file against the stored hash. Only files whose hashes have changed are re-embedded and re-indexed in Qdrant — unchanged files are skipped entirely. This incremental approach means a typical sync after editing one or two documents takes 3-10 seconds rather than the 30-60 seconds required for a full reindex. The state file uses a v2 format with source labels, storing entries as `{"source_label/filename": hash16}` to enable per-source tracking across the two document directories. Calling `rag_reindex(force=false)` (the default) triggers this incremental sync. Calling `rag_reindex(force=true)` deletes the entire `lmstudio_docs` collection and rebuilds from scratch — use this only when corruption is suspected or after a schema change.

<!-- Related: incremental sync, SHA-256, hash, .rag_state.json, rag_reindex, file tracking, staleness, force reindex -->

## How are multiple document sources organized?

<!-- Verified: 2026-02-13 -->

The qdrant-rag server indexes documents from two source directories into the single `lmstudio_docs` collection on Qdrant port 6334. The "lmstudio" source draws from the `rag-docs/lm-studio/` directory, containing reference documents covering architecture, RAG, tools, DyTopo, models, Memory, system prompt design, pipeline comparison, workflow patterns, and troubleshooting. The "anythingllm" source draws from the `rag-docs/anythingllm/` directory, containing AnythingLLM-specific documentation. Source directories are configured via the `LMStudio_DOCS_DIR` and `AnythingLLM_DOCS_DIR` environment variables. Each chunk carries a `source_dir` payload index recording its origin ("lmstudio" or "anythingllm"). Three payload indexes are created for filtering: `source` (KEYWORD, for filename filtering), `section_header` (KEYWORD), and `source_dir` (KEYWORD, for source directory filtering). The `rag_search` tool accepts an optional `source` parameter that filters results by source directory. The `rag_sources` tool returns the list of configured source directories with file counts. The `rag_file_info(filename)` tool returns metadata for a specific file including its hash and chunk count.

<!-- Related: document sources, lmstudio, anythingllm, source filter, source_dir, rag_sources, rag_file_info, payload index, collection -->

## How should rag_search be queried effectively?

<!-- Verified: 2026-02-13 -->

Effective `rag_search` queries exploit both the dense and sparse components of hybrid search. Write natural language questions rather than keyword fragments — a query like "How does incremental sync work?" scores better against the hybrid pipeline than bare keywords because the dense embedding model captures semantic relationships between question-form queries and answer-form content. Include the key technical term alongside a plain-language description for best results: "What port does the MCP qdrant-rag server use for hybrid search?" activates both the dense match (semantic concept of "port assignment") and the sparse match (literal term "6334" in matching chunks). Use the `source` parameter when the relevant document set is known — `source="01-architecture-reference.md"` scopes to a specific file. Set `limit` to 3-5 for targeted queries where you need a precise answer, or 8-10 for broad surveys requiring comprehensive coverage. The collection name is `lmstudio_docs` on Qdrant port 6334, with dense vectors of 1024 dimensions (COSINE distance) and sparse vectors stored with `on_disk=false` for fast lexical matching.

<!-- Related: rag_search, query technique, natural language, hybrid matching, source filter, limit, search strategy -->

## What does rag_status report and when should rag_reindex be called?

<!-- Verified: 2026-02-13 -->

The `rag_status` tool returns the health and state of the RAG pipeline: whether the `lmstudio_docs` collection exists and is healthy, the total point count (number of indexed chunks), and a staleness check comparing current file hashes against `.rag_state.json`. If any files have been modified since the last index, `rag_status` flags them as stale. This is useful for verifying index freshness after editing documentation and for diagnosing issues when search results seem outdated. Call `rag_reindex` after editing any markdown files in the rag-docs directories. The default invocation with `force=false` performs incremental sync — only changed files are re-embedded, completing in 3-10 seconds. Use `force=true` only when index corruption is suspected or after a collection schema change, as a full rebuild re-embeds all documents and takes 30-60 seconds. The server also checks for staleness automatically on each `rag_search` call, so explicit reindex is primarily useful for ensuring freshness before a search or after a batch of edits.

<!-- Related: rag_status, rag_reindex, collection health, staleness, point count, index freshness, incremental sync -->

## How does the dual Qdrant topology serve two separate RAG pipelines?

<!-- Verified: 2026-02-13 -->

Two independent Qdrant Docker containers serve completely separate RAG pipelines with different retrieval physics. Port 6334 serves the MCP qdrant-rag server using hybrid dense+sparse search with RRF fusion. Documents are chunked on ##-header boundaries (800-2,300 characters, zero overlap), embedded by BGE-M3 FlagEmbedding on CPU into both 1024-dimensional dense vectors and sparse lexical weight vectors, and stored in the `lmstudio_docs` collection. RAG injection is explicit — the agent must call `rag_search` to retrieve context. Port 6333 serves AnythingLLM using dense-only cosine similarity. Documents are chunked into 6,600-character blocks with 1,000-character overlap, embedded by the BGE-M3 Q8_0 GGUF model on GPU (via LM Studio's `/v1/embeddings` endpoint) into dense-only 1024-dimensional vectors, and stored in workspace-scoped collections. RAG injection is automatic — AnythingLLM inserts retrieved context into the system message. The containers use different embedding models, different chunking strategies, different vector configurations, and store separate data. Reindexing one has zero effect on the other.

<!-- Related: dual Qdrant, port 6333, port 6334, AnythingLLM, MCP qdrant-rag, dense-only, hybrid search, RRF, chunking comparison -->

## What Qdrant collection configuration is used for hybrid search?

<!-- Verified: 2026-02-13 -->

The `lmstudio_docs` collection on Qdrant port 6334 is configured with two named vector fields for hybrid search. The dense vector field stores 1024-dimensional embeddings using COSINE distance, indexed with HNSW parameters m=16 and ef_construct=200 for balanced recall and build speed. INT8 scalar quantization is applied at quantile 0.99 with `always_ram=true` for efficient storage while keeping quantized vectors in memory. The sparse vector field stores lexical weight vectors with `on_disk=false` for fast retrieval of exact-match token weights. Points are upserted in batches of 20 during indexing. The collection supports three payload indexes for efficient filtering: `source` (KEYWORD) for filename-level filtering, `section_header` (KEYWORD) for section-level filtering, and `source_dir` (KEYWORD) for source directory filtering ("lmstudio" or "anythingllm"). The collection is created automatically on first index if it does not exist, with the full vector and index configuration applied at creation time.

<!-- Related: lmstudio_docs, collection config, HNSW, INT8 quantization, dense vectors, sparse vectors, COSINE, payload indexes, Qdrant configuration -->
