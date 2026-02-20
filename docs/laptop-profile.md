# Laptop Constraint Profile — RTX 2070 Max-Q

## Overview

This branch adapts AnyLoom from its desktop target (RTX 5090, 32GB VRAM) to run
on an RTX 2070 Max-Q laptop with 8GB VRAM. The core philosophy is
"VRAM Tetris" — sacrificing concurrency and context window for a functional
swarm that fits in constrained GPU memory.

## Hardware Target

| Component | Spec |
|-----------|------|
| CPU | i7-10750H (6-core, 12-thread) |
| RAM | 32GB DDR4 |
| GPU | RTX 2070 Max-Q (8GB VRAM, Turing sm_75) |

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

## Model Choice

**Qwen2.5-Coder-7B-Instruct Q4_K_M** was selected because:
- Fits in ~4.5GB VRAM, leaving comfortable headroom
- Strong instruction following and JSON schema compliance
- Code-optimized but competent at general tasks
- Dense model (not MoE) — predictable, stable VRAM usage

## Key Configuration Differences from Desktop

| Setting | Desktop | Laptop | Why |
|---------|---------|--------|-----|
| LLM model | Qwen3-30B-A3B | Qwen2.5-Coder-7B | VRAM constraint |
| Context window | 131K | 16K | @agent mode needs headroom over 8K |
| KV cache type | Q8_0 | Q8_0 | Q4_0 causes garbled output on 7B models |
| Parallel slots | 2 | 1 | Can't afford 2 contexts |
| max_concurrent | 2 | 1 | Matches single slot |
| Embedding | GPU | CPU | Free VRAM for LLM |
| T_max (rounds) | 5 | 3 | Token budget constraint |
| tau (routing) | 0.5 | 0.45 | 7B produces noisier descriptors |
| K_in (indegree) | 3 | 2 | Less context per agent |
| Batch size | 8192 | 2048 | Less VRAM headroom |
| LLM threads | 8 | 4 | 6-core CPU (leave 2 for OS) |
| Docker image | local Blackwell build | Official CUDA image | Turing GPU compatibility |

## Thread Allocation (i7-10750H)

| Service | Threads | When Active |
|---------|---------|-------------|
| llama.cpp LLM (generation) | 4 | During token generation |
| llama.cpp LLM (batch/prefill) | 6 | During prompt processing |
| llama.cpp Embedding (CPU) | 4 | During document indexing |
| DyTopo MiniLM-L6-v2 | 2 | During routing (<50ms) |
| Health monitor | 1 | Every 60s |

## Performance Expectations

- **Generation speed:** ~30-40 tokens/sec (vs ~60-80 on desktop)
- **Context window:** 8K tokens limits multi-turn depth
- **Swarm rounds:** 3 max — 7B model converges quickly or not at all
- **Embedding:** Slightly slower on CPU (~15-50ms per query, ~100-200ms per chunk during indexing)

## Model Download

```bash
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
  qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --local-dir models
```

## Troubleshooting

### OOM / CUDA Out of Memory
- Check `nvidia-smi` — VRAM should be ~5.5-5.8 GB
- Ensure no other GPU processes are running
- Reduce `LLM_CONTEXT_SIZE` in `.env` to 4096 as last resort

### Slow Generation
- Close GPU-accelerated browsers/apps competing for VRAM
- Monitor thermal throttling: `nvidia-smi -l 1`
- Ensure laptop is plugged in (GPU power management)

### Embedding Failures
- Embedding runs on CPU — check `docker logs anyloom-embedding`
- Verify model file exists: `ls models/bge-m3-q8_0.gguf`
- CPU embedding is slower; increase timeout if needed

### Garbled / Degenerate Output
- **Q4_0 KV cache causes garbled output on 7B models** — use Q8_0 instead
- Q4_0 KV cache quantization is too aggressive for 7B parameter models, producing
  word merging ("packageockerize"), wrong math (15*23=3.5), and repetition loops
- Q8_0 adds ~0.2 GB VRAM but produces clean, coherent output
- `--flash-attn on` is required for quantized KV cache (Q4_0 or Q8_0)

### AnythingLLM Token Budget
- System prompt: ~715 tokens (condensed laptop version)
- RAG chunks: ~2,500 tokens (4 chunks x ~625 tokens at 2,500-char max)
- Chat history: ~400 tokens (1 message history)
- Total prompt: ~3,800 tokens — leaves ~4,400 tokens for generation
- If prompt > 6,000 tokens: reduce topN or clear chat threads
