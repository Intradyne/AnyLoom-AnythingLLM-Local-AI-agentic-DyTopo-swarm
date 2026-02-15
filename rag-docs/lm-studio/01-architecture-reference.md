# Loom Architecture Reference

## What is the overall system topology?

<!-- Verified: 2026-02-13 -->

Loom is a fully local AI agent stack running on a single NVIDIA RTX 5090 with 32 GB GDDR7 VRAM. The inference engine is LM Studio on port 1234, hosting Qwen3-30B-A3B-Instruct-2507 at Q6_K quantization with an 80,000-token context window. Two independent frontends share this one model: LM Studio provides direct MCP tool access with hybrid search (combining dense semantic vectors and sparse lexical vectors via RRF fusion) through Qdrant on port 6334, while AnythingLLM provides workspace-based chat with dense-only RAG through a separate Qdrant container on port 6333. The MCP tool ecosystem consists of a Docker MCP Gateway running 9 containerized servers plus one native Python server (`qdrant_mcp_server.py`) that provides 5 RAG tools and 3 DyTopo multi-agent swarm tools. A local Memory knowledge graph persists facts across both frontends and sessions.

<!-- Related: Loom, system architecture, topology, RTX 5090, LM Studio, AnythingLLM, Qdrant, MCP, single-GPU -->

## What are the port assignments and which services use them?

<!-- Verified: 2026-02-13 -->

Port 1234 is the LM Studio API (OpenAI-compatible), used by AnythingLLM for both chat completions and embeddings (BGE-M3 Q8_0 GGUF on GPU), by DyTopo swarm inference calls, and by direct chat clients. Port 6333 is Qdrant REST for the first Docker container, used exclusively by AnythingLLM for workspace RAG with dense-only cosine similarity vectors from the BGE-M3 Q8_0 GGUF model loaded in LM Studio. Port 6334 is Qdrant REST for the second Docker container (host port 6334 mapped to the container's default port 6333 via `-p 6334:6333`), used exclusively by the MCP qdrant-rag server (`qdrant_mcp_server.py`) for hybrid dense+sparse search with RRF fusion, using the BGE-M3 FlagEmbedding model running on CPU. These two Qdrant containers are completely independent — they use different embedding models, different chunking strategies (6,600-char auto-split with 1,000-char overlap for port 6333 versus deterministic ##-header chunking with zero overlap for port 6334), and store separate data.

<!-- Related: port 1234, port 6333, port 6334, Qdrant, LM Studio API, OpenAI-compatible, AnythingLLM, MCP qdrant-rag -->

## What is the VRAM budget on the RTX 5090?

<!-- Verified: 2026-02-13 -->

The RTX 5090 has 32 GB GDDR7 VRAM. The Qwen3-30B-A3B model at Q6_K quantization occupies 25.1 GB for its weights. The BGE-M3 Q8_0 GGUF embedding model co-loaded in LM Studio for AnythingLLM's dense-only pipeline uses an additional 0.6 GB. CUDA runtime overhead accounts for approximately 1.0 GB, bringing the fixed allocation to roughly 26.7 GB. The KV cache for the context window scales with context length: at the default 80,000-token context, the KV cache (Q8_0 quantized) occupies approximately 3.75 GB, for a total of about 30.5 GB out of 32 GB. Scaling up to 96K context pushes the total to approximately 31.2 GB (tight but functional), while 128K context would exceed available VRAM and require dropping to Q5_K_M quantization. The MCP RAG pipeline's BGE-M3 FlagEmbedding runs entirely on CPU (~2.3 GB system RAM) and consumes zero VRAM.

<!-- Related: VRAM, GPU memory, RTX 5090, 32 GB, KV cache, Q6_K, 80K context, Qwen3, BGE-M3 GGUF, memory budget -->

## How is the 80K token context window allocated?

<!-- Verified: 2026-02-13 -->

The 80,000-token context window has approximately 8,900 tokens of fixed overhead: the system prompt consumes roughly 2,000 tokens and LM Studio's Jinja-injected tool definitions add approximately 6,900 tokens for 15 tools across the Docker MCP Gateway and qdrant-rag server. This leaves about 71,000 tokens available for conversation history, RAG search results, tool output, and model responses. The RollingWindow context manager in LM Studio drops the oldest messages when the window fills, which means key findings must be summarized into response text to persist. Targeted `rag_search` queries with a limit of 3-5 results conserve context more effectively than broad surveys at 8-10 results. At Q8_0 KV cache quantization, each 1,000 tokens of context costs approximately 47 MB of VRAM.

<!-- Related: context window, 80K tokens, token budget, overhead, system prompt, tool definitions, RollingWindow, KV cache -->

## What CPU and system RAM resources does the stack use?

<!-- Verified: 2026-02-13 -->

The host CPU is an AMD Ryzen 9 9950X3D with 16 cores and 32 threads, paired with 94 GB DDR5 system RAM. Two CPU-resident models run alongside GPU inference. The BGE-M3 FlagEmbedding model for the MCP RAG pipeline (hybrid search on port 6334) uses 8 dedicated CPU threads (half of physical cores, set via the `RAG_CPU_THREADS` environment variable) and occupies approximately 2.3 GB of system RAM. It generates both 1024-dimensional dense semantic vectors and sparse lexical weight vectors for each chunk. The MiniLM-L6-v2 model for DyTopo swarm routing uses approximately 80 MB of system RAM and generates 384-dimensional embeddings for computing cosine similarity between agent descriptors. Both models are lazy-loaded on first use — BGE-M3 loads in `qdrant_mcp_server.py` on the first `rag_search` call with a 30-60 second cold start, and MiniLM-L6-v2 loads in `src/dytopo/router.py` on the first swarm invocation in under 1 second — then remain resident as singletons. Tokenizer parallelism is disabled (`TOKENIZERS_PARALLELISM=false`) to prevent thread contention with the embedding batch processing.

<!-- Related: CPU, Ryzen 9 9950X3D, RAM, DDR5, BGE-M3, FlagEmbedding, MiniLM, CPU threads, system resources -->

## How do the LM Studio and AnythingLLM frontends differ?

<!-- Verified: 2026-02-13 -->

LM Studio and AnythingLLM are two independent frontends that share the same Qwen3-30B-A3B inference on port 1234 but differ in their RAG pipelines, tool access, and safety mechanisms. LM Studio provides direct MCP tool access to 9 Docker-hosted servers plus the native qdrant-rag server, uses hybrid dense+sparse search with RRF fusion on Qdrant port 6334, employs deterministic ##-header chunking (800-2,300 character chunks with zero overlap), and has explicit RAG injection (the agent must call `rag_search` to retrieve documents). LM Studio provides raw access with zero built-in safety nets — the system prompt must compensate for the lack of iteration limits, JSON repair, context compression, loop deduplication, and error recovery. AnythingLLM provides workspace-based chat with its AIbitat agent framework, uses dense-only cosine similarity on Qdrant port 6333, employs 6,600-character auto-split chunking with 1,000-character overlap, and automatically injects RAG context into the system message. AnythingLLM includes built-in safety nets: an 8-round iteration cap, JSON repair, context compression, and deduplication. The default inference temperature is 0.3 for LM Studio and 0.1 for AnythingLLM.

<!-- Related: LM Studio, AnythingLLM, frontend comparison, hybrid search, dense-only, safety nets, AIbitat, MCP, temperature -->
