# Model and Embedding Reference

## What chat model does this stack use?

<!-- Verified: 2026-02-13 -->

The stack runs Qwen3-30B-A3B-Instruct-2507 quantized to Q6_K, loaded in LM Studio on an NVIDIA RTX 5090 with 32 GB VRAM. The model uses a Mixture-of-Experts (MoE) architecture with 30.5 billion total parameters but only 3.3 billion active per token, making it efficient despite its large total size. It has 48 layers with 128 experts per MoE layer and 8 experts active per token. The attention mechanism uses Grouped Query Attention (GQA) at an 8:1 ratio — 32 query heads sharing 4 KV heads. The Q6_K quantization occupies approximately 25.1 GB of VRAM for weights and retains roughly 99% of FP16 quality. KV cache uses Q8_0 format at approximately 48 KB per token, consuming about 3.75 GB at the configured 80,000-token context length. All 48 layers are offloaded to GPU — partial offload is avoided because it degrades MoE routing quality. The model achieves a tool calling F1 score of 0.971 and its post-training pipeline is specifically optimized for agent use, tool calling, MCP integration, and structured JSON output.

<!-- Related: Qwen3, 30B-A3B, MoE, Q6_K, VRAM, RTX 5090, quantization, GQA, KV cache, tool calling -->

## How does Qwen3's hybrid thinking mode work?

<!-- Verified: 2026-02-13 -->

Qwen3 supports a hybrid thinking mode controlled by two special tokens that toggle internal chain-of-thought reasoning. Prefixing a message with `/think` activates extended reasoning — the model generates internal chain-of-thought before producing its final answer, improving accuracy on complex multi-step problems, debugging, architecture analysis, and planning tasks. Prefixing with `/no_think` disables internal reasoning, producing fast structured output suitable for tool calls, descriptor generation, and straightforward responses where reasoning overhead is unnecessary. When neither prefix is provided, the model decides automatically whether to engage thinking mode based on query complexity. In API calls, the `/think` or `/no_think` prefix is included at the beginning of the user message content. DyTopo uses `/no_think` for Phase A descriptor generation where speed and determinism matter, and lets the model decide for Phase C work output. The LM Studio agent system prompt routes explicit category lists to each mode: `/think` for multi-step debugging, architecture analysis, code review, and planning 3+ tool sequences; `/no_think` for status checks, single tool calls, direct lookups, formatting, and poll operations.

<!-- Related: /think, /no_think, hybrid thinking, reasoning mode, chain-of-thought, extended reasoning, fast response, Qwen3 -->

## What sampling settings are used for inference?

<!-- Verified: 2026-02-13 -->

LM Studio serves Qwen3-30B-A3B with these sampling parameters: temperature 0.3, top_K 20, top_P 0.85, min_P 0.05. The repeat penalty is 1.0, which effectively disables it — this is critical because values above 1.0 corrupt JSON output by penalizing the repeated structural characters (`{`, `"`, `,`) that valid JSON requires. AnythingLLM overrides temperature to 0.1 per-request for tighter agentic tool calls with less variance. DyTopo descriptor generation (Phase A) uses temperature 0.1 per-request for deterministic output, while DyTopo work execution (Phase C) uses temperature 0.3. The KV cache quantization is Q8_0, balancing memory savings with quality at the 80,000-token context window. Eval batch size is 2048 tokens for parallel prompt evaluation. The context overflow policy is RollingWindow, which drops the oldest messages when the window fills rather than truncating or erroring. Flash Attention is enabled for memory-efficient attention computation. Speculative decoding and structured output mode are both disabled. The model achieves over 110 tokens per second at 32K context and over 52 tokens per second at 128K context on the RTX 5090.

<!-- Related: temperature, sampling, top_K, top_P, min_P, repeat penalty, KV cache, Q8_0, RollingWindow, Flash Attention, inference speed -->

## How does the BGE-M3 GGUF embedding pipeline work in LM Studio?

<!-- Verified: 2026-02-13 -->

Pipeline 1 runs the BGE-M3 model as a GGUF file (bge-m3-Q8_0.gguf) co-loaded in LM Studio alongside Qwen3 on the GPU. It consumes approximately 635 MB (0.6 GB) of VRAM and produces dense-only 1024-dimensional embedding vectors via the `/v1/embeddings` endpoint at localhost:1234. This pipeline serves AnythingLLM's workspace RAG system, which stores vectors in Qdrant on port 6333 with dense-only cosine similarity search. AnythingLLM is configured with the embedding model identifier `ggml-org/bge-m3-Q8_0`, a chunk size of 6,600 characters, chunk overlap of 1,000 characters, and a maximum of 16 context snippets at Low similarity threshold. The Q8_0 quantization is near-lossless for embedding models, preserving vector quality while keeping VRAM usage modest. Because this is a GGUF inference pipeline, it produces dense vectors only — it cannot generate the sparse lexical vectors needed for hybrid search with RRF fusion. This is why a separate CPU pipeline exists for the MCP qdrant-rag server.

<!-- Related: BGE-M3, GGUF, Q8_0, LM Studio, /v1/embeddings, AnythingLLM, port 6333, dense-only, VRAM, Pipeline 1 -->

## How does the BGE-M3 FlagEmbedding CPU pipeline work for hybrid search?

<!-- Verified: 2026-02-13 -->

Pipeline 2 runs the same BGE-M3 model using the FlagEmbedding Python library with full PyTorch weights (BAAI/bge-m3), loaded entirely on CPU. It consumes approximately 2.3 GB of system RAM with zero VRAM usage and produces both dense 1024-dimensional semantic vectors and sparse lexical weight vectors with `return_sparse=True`. This pipeline serves the MCP qdrant-rag server (`qdrant_mcp_server.py`), which stores hybrid vectors in the `lmstudio_docs` collection on Qdrant port 6334. The max_length parameter is 1024 tokens per chunk, sufficient for the stack's ##-header chunked documents (800-2,300 characters). Batch size is 16 (configurable via `RAG_EMBED_BATCH_SIZE`). ColBERT late-interaction vectors are disabled because they incur approximately 50x storage overhead with marginal retrieval benefit for this use case. CPU threading is configured at 8 threads via OMP and MKL environment variables (set by `RAG_CPU_THREADS`). Gradient computation is disabled with `torch.set_grad_enabled(False)` since this is inference-only. The model is loaded on first search request, with a cold start time of approximately 30-60 seconds.

<!-- Related: BGE-M3, FlagEmbedding, CPU pipeline, sparse vectors, dense vectors, hybrid search, port 6334, Pipeline 2, BAAI/bge-m3 -->

## Why does hybrid search require the CPU pipeline instead of the LM Studio GGUF?

<!-- Verified: 2026-02-13 -->

Hybrid search combines dense semantic vectors with sparse lexical vectors using RRF fusion to promote results that score well on both signals. The LM Studio GGUF pipeline can produce dense vectors only — the GGUF format does not support sparse vector generation. The FlagEmbedding library with `return_sparse=True` is required to generate sparse lexical vectors where individual tokens receive learned weights reflecting their importance. This matters for technical documentation retrieval because exact identifiers like port numbers (`6334`), function names (`rag_search`, `create_entities`), environment variables (`QDRANT_URL`, `RAG_CPU_THREADS`), and configuration keys are critical for precise matching. Dense semantic vectors average these identifiers into a single vector representation, potentially losing the specific token that distinguishes one port from another. Sparse lexical vectors preserve individual token weights, so a search for "port 6334" directly matches documents containing that literal string. RRF fusion then combines both ranking signals — semantic similarity from dense vectors and token-level precision from sparse lexical vectors — producing more accurate retrieval for the mixed natural-language and technical-identifier queries typical in this stack.

<!-- Related: GGUF limitation, sparse vectors, FlagEmbedding, return_sparse, hybrid search, RRF fusion, exact matching, lexical weights, dense-only vs hybrid -->

## What is MiniLM-L6-v2 and why is it used for DyTopo routing?

<!-- Verified: 2026-02-13 -->

MiniLM-L6-v2 is a compact sentence-transformer model with 22 million parameters, producing L2-normalized 384-dimensional vectors and consuming approximately 80 MB of system RAM on CPU. Its sole purpose in this stack is DyTopo descriptor routing — it embeds the key and query descriptors that agents generate in Phase A, and the cosine similarity matrix built from these embeddings determines which agents communicate in each round. MiniLM-L6-v2 was chosen over BGE-M3 for this routing task because it has a wider similarity spread in its output distribution, which provides better threshold discrimination at the routing threshold τ boundary. When tau is 0.3, the system needs to cleanly separate "should connect" from "should disconnect" — a model with tighter similarity clustering would make this boundary less reliable and produce unstable topologies. MiniLM is loaded on the first swarm invocation via the `SentenceTransformer` library and persists as an independent singleton. It is separate from the BGE-M3 models and does not interact with the RAG pipeline, Qdrant, or the embedding process.

<!-- Related: MiniLM-L6-v2, sentence-transformer, 384-dim, routing, DyTopo, tau threshold, similarity spread, descriptor embedding -->

## Why was BGE-M3 chosen over alternative embedding models?

<!-- Verified: 2026-02-13 -->

BGE-M3 was selected after evaluating alternatives against the stack's dual-pipeline requirements. Qwen3-Embedding was considered but requires an Instruct-format prefix for each input, which AnythingLLM cannot inject — it sends raw text to the embedding endpoint without prompt templating, making Qwen3-Embedding incompatible with the AnythingLLM workspace RAG pipeline. Nomic-embed was also evaluated but produces 768-dimensional vectors compared to BGE-M3's 1024 dimensions, and supports a maximum context of 2048 tokens versus BGE-M3's 8192 tokens. The decisive advantage of BGE-M3 is its dual capability: the same base model can run as a GGUF in LM Studio for dense-only workspace RAG (Pipeline 1, serving AnythingLLM on port 6333) and as FlagEmbedding on CPU for dense+sparse hybrid search (Pipeline 2, serving the MCP qdrant-rag server on port 6334). Both pipelines produce vectors in the same 1024-dimensional semantic space, ensuring consistency across the two Qdrant collections even though they use different embedding implementations and different chunking strategies.

<!-- Related: BGE-M3 selection, Qwen3-Embedding, nomic-embed, model comparison, dual pipeline, 1024-dim, embedding choice -->
