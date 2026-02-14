# Dual RAG Pipeline Comparison Reference

Reference documentation comparing the two independent RAG retrieval pipelines in the Loom stack — the hybrid dense+sparse pipeline on port 6334 serving the LM Studio agent, and the dense-only pipeline on port 6333 serving the AnythingLLM workspace agent.


## How do the two RAG retrieval pipelines differ in search mechanics?

<!-- Verified: 2026-02-13 -->

The Loom stack runs two completely independent Qdrant instances with architecturally different retrieval strategies. The LM Studio agent's pipeline on port 6334 uses hybrid dense+sparse search with Reciprocal Rank Fusion (RRF), combining BGE-M3's 1024-dimensional dense semantic vectors with learned sparse lexical weights. The AnythingLLM pipeline on port 6333 uses dense-only cosine similarity search with the same BGE-M3 model in GGUF Q8_0 format.

Hybrid search catches exact technical identifiers — port numbers like 6334, tool names like rag_search, environment variables like QDRANT_URL — through the sparse lexical component even when the dense embedding places the query elsewhere semantically. Dense-only relies entirely on cosine similarity between query and chunk embeddings, meaning every retrievable concept must be embedded in prose that semantically overlaps with likely queries.

<!-- Related: dual RAG pipelines, hybrid vs dense-only, port 6333, port 6334, dense-only cosine similarity, hybrid dense+sparse, RRF fusion, Reciprocal Rank Fusion, BGE-M3, sparse lexical weights, exact term matching, semantic embedding -->


## What are the practical advantages of hybrid search over dense-only?

<!-- Verified: 2026-02-13 -->

Hybrid search on port 6334 excels when queries contain exact identifiers. A query for "what does QDRANT_URL default to" finds chunks mentioning QDRANT_URL through sparse term matching even if the dense embedding focuses broadly on "environment variable configuration." A query for "port 6334" finds the exact port number through lexical matching regardless of semantic context. This makes hybrid search more robust for infrastructure-specific queries.

Dense-only search on port 6333 has the advantage of simplicity — no tuning of sparse weight balance, and retrieval quality depends entirely on how well document content matches the semantic space of natural language questions. The trade-off is that vocabulary bridging (including expert terms and plain-language equivalents) becomes critical. A bare table `| 6333 | Qdrant |` embeds generically; a prose sentence "Port 6333 serves AnythingLLM's dense-only workspace RAG" embeds near relevant queries.

<!-- Related: hybrid search advantages, sparse term matching, exact identifiers, vocabulary bridging, dense retrieval quality, infrastructure queries, natural language questions, embedding quality, query specificity -->


## How do the chunking strategies differ between the two pipelines?

<!-- Verified: 2026-02-13 -->

The LM Studio pipeline on port 6334 uses deterministic section-header-based chunking: qdrant_mcp_server.py splits markdown files on every `##` header, prepends the document's `# Title` to each chunk for context anchoring, and discards fragments shorter than 80 characters. This produces chunks of 800 to 2,300 characters with zero overlap. Each `##` heading becomes exactly one Qdrant point — the document author fully controls chunk boundaries.

The AnythingLLM pipeline on port 6333 uses a recursive text splitter targeting 6,600-character chunks with 1,000-character overlap. The splitter respects paragraph boundaries (`\n\n` then `\n`), but the author does not control exact boundaries — they influence them by writing sections that naturally fall within the 4,000 to 6,600 character range. The 1,000-character overlap duplicates trailing content at the start of the next chunk, providing continuity across chunk boundaries.

<!-- Related: chunking strategy, section-header chunking, recursive text splitter, 6600-character chunks, 2300-character limit, 1000-character overlap, zero overlap, deterministic chunking, chunk boundaries, Qdrant points, document authoring -->


## How does embedding computation differ between the two pipelines?

<!-- Verified: 2026-02-13 -->

The LM Studio pipeline uses BGE-M3 loaded via the FlagEmbedding Python library directly on CPU, consuming approximately 2.3 GB system RAM with zero VRAM. Configuration: use_fp16=False, max_length=1024, batch_size=16. This produces both dense 1024-dimensional vectors and sparse lexical weight vectors for hybrid search. The model runs on the Ryzen 9 9950X3D with 8 dedicated threads (RAG_CPU_THREADS). First-time loading takes 30-60 seconds; subsequent embeddings complete in seconds.

The AnythingLLM pipeline uses ggml-org/bge-m3-Q8_0, a GGUF-quantized version co-loaded in LM Studio alongside Qwen3 on the GPU, consuming approximately 0.6 GB VRAM. It produces only dense vectors through the /v1/embeddings endpoint — no sparse component because the GGUF format and LM Studio's endpoint do not support FlagEmbedding's multi-output mode. AnythingLLM's embed chunk length of 32,768 characters (~8,192 tokens) matches BGE-M3's input window.

<!-- Related: BGE-M3, FlagEmbedding CPU, GGUF Q8_0 GPU, dense embeddings, sparse vectors, embedding computation, VRAM 0.6 GB, RAM 2.3 GB, max_length 1024, batch_size 16, /v1/embeddings endpoint, Ryzen 9 9950X3D, CPU threads -->


## How does document injection differ and what does that mean for system prompt design?

<!-- Verified: 2026-02-13 -->

AnythingLLM injects retrieved chunks automatically — they appear after a "Context:" separator in the system message on every relevant query, with zero explicit action from the agent. The agent simply receives workspace knowledge and grounds its answers in it. The LM Studio agent must explicitly call rag_search to retrieve documents — there is no automatic injection.

This creates fundamentally different system prompt requirements. The AnythingLLM prompt includes a detailed HOW TO USE CONTEXT section teaching the model to cite specific values from auto-injected context, acknowledge coverage boundaries, and synthesize across chunks. The LM Studio prompt includes a RAG SEARCH STRATEGY section with domain triggers telling the agent when to call rag_search, query technique guidance, reformulation strategies, and source filter usage. The same information often exists in both pipelines but in format-divergent versions — terser for hybrid (sparse catches keywords), richer prose for dense-only (no keyword fallback).

<!-- Related: automatic injection, explicit rag_search, Context separator, document injection, system prompt design, format-divergent, HOW TO USE CONTEXT, RAG SEARCH STRATEGY, dense-only prose, hybrid terseness, query technique, coverage boundaries -->


## What retrieval configuration settings govern search quality in each pipeline?

<!-- Verified: 2026-02-13 -->

The LM Studio pipeline on port 6334 uses a minimum score threshold of 0.005 (RAG_MIN_SCORE), filtering out results below this combined RRF score. Default search limit is 5 results, adjustable up to 10. Results are ranked by RRF fusion which combines dense and sparse rankings using reciprocal rank weighting. The collection name is lmstudio_docs, and source_dir payload filtering allows scoping searches to "lmstudio" or "anythingllm" document sources.

The AnythingLLM pipeline on port 6333 uses a low document similarity threshold (the lowest available in workspace settings), retrieving up to 16 snippets per query with 30 messages of conversation history. This permissive threshold admits more candidate chunks at the cost of potential noise — a worthwhile trade-off given BGE-M3's embedding quality and the 80K context window that accommodates extra retrieved content. You have plenty of context room. Focus on answer quality, not token counting. For reference, the approximate budget at 80K: system prompt ~2K, tool definitions ~3K, RAG snippets ~8K (16 x ~500 tokens), chat history ~12K, totaling ~25K overhead with ~55K remaining for the current exchange. Note: this ~8K RAG snippet budget is AnythingLLM's auto-injected context only (port 6333). LM Studio's explicit `rag_search` results (port 6334) come from its own separate ~71K remaining budget after ~8.9K system prompt and tool definition overhead. The two RAG budgets are independent — each pipeline has its own context allocation.

<!-- Related: retrieval configuration, minimum score threshold, RAG_MIN_SCORE, 0.005, similarity threshold, 16 snippets, 30 message history, RRF fusion score, source_dir filtering, lmstudio_docs, context budget, 80K tokens, 25K overhead, 55K remaining, search limit, workspace settings -->
