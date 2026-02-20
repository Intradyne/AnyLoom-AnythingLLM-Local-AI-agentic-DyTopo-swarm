# Infrastructure and Port Topology

## Hardware (Laptop Profile)

- **GPU:** RTX 2070 Max-Q (8 GB VRAM, Turing sm_75)
- **CPU:** i7-10750H (6-core, 12-thread)
- **RAM:** 32 GB DDR4

## VRAM Budget

| Component | VRAM |
|-----------|------|
| Qwen2.5-Coder-7B Q4_K_M weights | ~4.5 GB |
| KV cache 16K ctx (Q8_0/Q8_0) | ~1.0 GB |
| llama.cpp overhead | ~0.3 GB |
| OS / display compositor | ~0.5 GB |
| **Total** | **~6.3 GB** |
| **Headroom** | **~1.7 GB** |

Embedding (BGE-M3) runs on CPU — zero GPU footprint.

## Port Topology

| Port | Service | Container | Notes |
|------|---------|-----------|-------|
| 8008 (host) → 8080 (container) | llama.cpp inference (OpenAI-compatible) | `anyloom-llm` | Chat completions + embeddings |
| 8009 (host) → 8080 (container) | llama.cpp embedding (CPU, BGE-M3) | `anyloom-embedding` | Dedicated CPU embedding server |
| 6333 (REST), 6334 (gRPC) | Qdrant vector database | `anyloom-qdrant` | All vector storage |
| 3001 | AnythingLLM web UI | `anyloom-anythingllm` | Workspace management |

- **From host/scripts:** `http://localhost:8008/v1`
- **From Docker containers:** `http://anyloom-llm:8080/v1`
- **Embedding from containers:** `http://anyloom-embedding:8080/v1`

## Model Configuration

- **LLM:** Qwen2.5-Coder-7B-Instruct Q4_K_M (~4.5 GB VRAM)
- **Context window:** 16,384 tokens (--ctx-size 16384)
- **KV cache:** Q8_0/Q8_0 (Q4_0 causes garbled output on 7B models)
- **Parallel slots:** 1 (single context — VRAM constraint)
- **Temperature:** 0.3 (workspace setting)
- **Embedding:** BGE-M3 Q8_0 GGUF on CPU (~635 MB RAM, 0 VRAM)

## Inference Bottleneck

All consumers share the llama.cpp endpoint. Requests process sequentially — one at a time.
- Normal interactive use: negligible impact (individual calls complete in seconds)
- During DyTopo swarms: 10-60 second delays for interactive chat
- Generation speed: ~30-40 tokens/sec
- Embedding on CPU: ~15-50ms per query, ~100-200ms per chunk during indexing

## RAG Pipelines

### AnythingLLM Pipeline (passive, automatic)
- Dense-only cosine similarity search on Qdrant port 6333
- BGE-M3 embeddings via CPU embedding server (port 8009)
- Chunks injected automatically into system message after "Context:" separator
- Zero agent action required

### MCP Hybrid Pipeline (active, explicit)
- Hybrid dense+sparse search with RRF fusion on same Qdrant instance
- Requires explicit rag_search MCP tool calls
- Separate collection from AnythingLLM workspace docs
- Source_dir payload filtering for scoped searches

Both pipelines query the same `anyloom-qdrant` container but use separate collections.

## DyTopo Swarm System

- MCP-exclusive (not available in AnythingLLM)
- Tools: swarm_start, swarm_status, swarm_result
- Three domains: code (Developer, Researcher, Tester, Designer), math (ProblemParser, Solver, Verifier), general (Analyst, Critic, Synthesizer)
- Routing: MiniLM-L6-v2 on CPU (~80 MB RAM) for semantic descriptor matching
- tau parameter controls communication density (default 0.45 for laptop)
- max_rounds: 3 (laptop constraint — 7B model converges quickly)
- All inference routes through shared llama.cpp endpoint

## Thread Allocation (i7-10750H)

| Service | Threads | When Active |
|---------|---------|-------------|
| llama.cpp LLM (generation) | 4 | During token generation |
| llama.cpp LLM (batch/prefill) | 6 | During prompt processing |
| llama.cpp Embedding (CPU) | 4 | During document indexing |
| DyTopo MiniLM-L6-v2 | 2 | During routing (<50ms) |

## Troubleshooting

- **OOM/CUDA errors:** Check `nvidia-smi` — should be ~5.0-5.5 GB. Close GPU-accelerated apps.
- **Slow generation:** Check thermal throttling. Ensure laptop is plugged in.
- **Embedding failures:** Check `docker logs anyloom-embedding`. Verify bge-m3-q8_0.gguf exists.
- **High latency:** Check for active DyTopo swarms. Start fresh conversation to reset KV cache.
