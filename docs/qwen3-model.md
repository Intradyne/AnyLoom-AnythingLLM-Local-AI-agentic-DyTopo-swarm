# Qwen3-30B-A3B-Instruct-2507

**Author:** Qwen Team, Alibaba Cloud
**Model Card:** https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507
**GGUF Source:** https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF
**Architecture:** Mixture-of-Experts — 30.5B total, 3.3B active per token
**Experts:** 128 per MoE layer, 8 active per token
**Layers:** 48
**KV Heads:** 4 (32 query heads, 8:1 GQA ratio)
**Native Context:** 256K (via YaRN in the 2507 update)

---

## Deployment

AnyLoom deploys this model via llama.cpp server with GGUF quantization:

### llama.cpp (Port 8008 host / 8080 container)
- **Format:** Q4_K_M GGUF (~18.6 GiB weights in VRAM)
- **Use case:** All LLM inference (DyTopo swarm agents, AnythingLLM chat)
- **Context:** 131K tokens with quantized KV cache
- **Throughput:** ~234 tok/s generation, ~110 tok/s at 32K context
- **Platform:** Docker container (`ghcr.io/ggml-org/llama.cpp:server-cuda`)
- **Memory:** ~24.6 GiB VRAM at 131K context (Q8/Q4 KV cache, flash attention)

---

## Why this model

This MoE model delivers 70B-class quality and tool calling accuracy rivaling GPT-4, while fitting in the RTX 5090's 32GB VRAM through Q4_K_M GGUF quantization (~18.6 GiB weights) with quantized KV cache and flash attention.

**Tool calling.** Docker's practical evaluation (3,570 test cases, 21 models) measured Qwen3 at 0.971 F1 on tool selection — within 0.003 of GPT-4 (0.974). The four-stage post-training pipeline explicitly optimizes for agent/tool-use, MCP integration, and structured JSON output.

**Speed.** MoE activates only 3.3B of 30.5B params per token. On the RTX 5090: ~234 tok/s generation, ~110 tok/s at 32K context, ~50 tok/s at 131K. Faster than a dense 32B model despite higher overall quality.

**KV cache efficiency.** 4 KV heads across 48 layers produces roughly 3x smaller cache than a dense 32B model. With quantized KV cache (Q8_0 keys / Q4_0 values, ~39 KiB/token), 131K context fits in ~5 GiB.

**Precision.** Q4_K_M quantization maintains high quality (+0.05 perplexity vs FP16) for structured JSON output and tool calls. All 128 experts per layer are stored in VRAM — only the activation path is sparse.

## MoE architecture notes

Despite having 30.5B total parameters, inference touches only 3.3B per token. All expert weights must reside in VRAM (no selective loading), so the model occupies ~18.6 GiB VRAM (Q4_K_M) regardless of how few experts activate. The architectural savings show up in speed and KV cache, not in weight storage.

The 48-layer architecture uses GPU offload 48/48 (not 64 like dense Qwen models). Every layer must be on GPU — partial offload degrades MoE routing.

## Hybrid thinking mode

Qwen3 supports per-turn thinking control:

- **Default:** Model decides whether to reason based on query complexity
- **`/think`:** Forces chain-of-thought reasoning. Use for multi-step planning, complex tool chains, debugging.
- **`/no_think`:** Skips reasoning for fast structured output. Use for simple tool calls, status checks, direct answers.

For API calls, prepend the tag to the user message content.

## Sampling rationale

**Temperature → 0.3.** Hybrid thinking handles determinism internally. At 0.3, `/no_think` responses are near-deterministic for tool calls while `/think` retains enough diversity for reasoning. Use 0.1 if tool calls are occasionally malformed; 0.5 for creative work. AnythingLLM uses 0.1 for tighter agentic control. This value provides a good balance for most use cases.

**Repeat Penalty → 1.0 (OFF).** Any value >1.0 corrupts tool-call JSON by penalizing repeated structural tokens (`{`, `"`, `,`).

**Top K → 20 / Top P → 0.85 / Min P → 0.05.** Conservative sampling that constrains token selection without over-restricting the model's vocabulary. Standard for Qwen3 instruction-tuned models.

**Context Overflow → RollingWindow.** System prompt stays pinned, oldest messages drop first. With 131K context, the token budget is generous — see README.md for the budget breakdown in typical agentic sessions.

**Eval Batch Size → 2048.** RTX 5090's 1,792 GB/s bandwidth handles large batches easily. MoE benefits from large batch prefill — multiple experts execute in parallel during prompt evaluation.

**KV Cache → Q8_0 keys / Q4_0 values.** Quantized KV cache at ~39 KiB/token dramatically extends context capacity with negligible quality loss. Q8_0 keys add only +0.004 perplexity. Configured via --cache-type-k q8_0 --cache-type-v q4_0 in llama.cpp.

## Alternative configurations

**128K context with higher quality:** Use Q5_K_M weights (~21.7 GiB). Total VRAM at 128K: ~28.7 GiB. Speed holds above 52 tok/s.

**Coding focus:** Consider Qwen3-Coder-30B-A3B-Instruct (same architecture, optimized for multi-file agentic coding). Q4_K_M fits 131K context comfortably.

**24GB GPUs:** Q4_K_M weights (~18.6 GiB) fit on 24GB GPUs with reduced context. Use --ctx-size 32768 or lower to stay within VRAM budget.
