# Qwen3-30B-A3B-Instruct-2507

**Author:** Qwen Team, Alibaba Cloud
**Model Card:** https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507
**GGUF Source:** https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF
**Architecture:** Mixture-of-Experts — 30.5B total, 3.3B active per token
**Experts:** 128 per MoE layer, 8 active per token
**Layers:** 48
**KV Heads:** 4 (32 query heads, 8:1 GQA ratio)
**Native Context:** 256K (via YaRN in the 2507 update)
**Quantization:** Q6_K (~25.1 GB weights)

---

## Why this model

This MoE model fills the RTX 5090's 32 GB VRAM with a near-lossless quantization (Q6_K, ~99% FP16 quality) while delivering 70B-class quality and tool calling accuracy rivaling GPT-4.

**Tool calling.** Docker's practical evaluation (3,570 test cases, 21 models) measured Qwen3 at 0.971 F1 on tool selection — within 0.003 of GPT-4 (0.974). The four-stage post-training pipeline explicitly optimizes for agent/tool-use, MCP integration, and structured JSON output.

**Speed.** MoE activates only 3.3B of 30.5B params per token. On the RTX 5090: 110+ tok/s at 32K context, 52+ tok/s at 128K. Faster than a dense 32B model despite higher overall quality.

**KV cache efficiency.** 4 KV heads across 48 layers produces ~48 KB/token at Q8_0 — roughly 3× smaller than a dense 32B model (~160 KB/token). At the 80K default context, KV cache is ~3.75 GB — fitting comfortably in the remaining VRAM after weights.

**Quantization.** Q6_K retains ~99% of FP16 quality (vs ~95% for Q4_K_M). For structured JSON output, this eliminates subtle attention-layer corruption that causes malformed tool calls. All 128 experts per layer are stored in VRAM — only the activation path is sparse.

## MoE architecture notes

Despite having 30.5B total parameters, inference touches only 3.3B per token. All expert weights must reside in VRAM (no selective loading), so the model file is ~25 GB regardless of how few experts activate. The architectural savings show up in speed and KV cache, not in weight storage.

The 48-layer architecture uses GPU offload 48/48 (not 64 like dense Qwen models). Every layer must be on GPU — partial offload degrades MoE routing.

## Hybrid thinking mode

Qwen3 supports per-turn thinking control:

- **Default:** Model decides whether to reason based on query complexity
- **`/think`:** Forces chain-of-thought reasoning. Use for multi-step planning, complex tool chains, debugging.
- **`/no_think`:** Skips reasoning for fast structured output. Use for simple tool calls, status checks, direct answers.

For API calls, prepend the tag to the user message content.

## Sampling rationale

**Temperature → 0.3.** Hybrid thinking handles determinism internally. At 0.3, `/no_think` responses are near-deterministic for tool calls while `/think` retains enough diversity for reasoning. Use 0.1 if tool calls are occasionally malformed; 0.5 for creative work. AnythingLLM uses 0.1 for tighter agentic control.

**Repeat Penalty → 1.0 (OFF).** Any value >1.0 corrupts tool-call JSON by penalizing repeated structural tokens (`{`, `"`, `,`).

**Top K → 20 / Top P → 0.85 / Min P → 0.05.** Conservative sampling that constrains token selection without over-restricting the model's vocabulary. Standard for Qwen3 instruction-tuned models.

**Context Overflow → RollingWindow.** System prompt stays pinned, oldest messages drop first. With 80K context, overflow is rare in typical agentic sessions.

**Eval Batch Size → 2048.** RTX 5090's 1,792 GB/s bandwidth handles large batches easily. MoE benefits from large batch prefill — multiple experts execute in parallel during prompt evaluation.

**KV Cache → Q8_0.** Halves cache memory with negligible quality loss. At 80K context this saves ~3.75 GB vs unquantized. Avoid Q4_0 — MoE router decisions are sensitive to KV precision, and aggressive quantization causes expert selection drift over long contexts.

## Alternative configurations

**128K context:** Drop to Q5_K_M weights (~21.7 GB). Total VRAM at 128K: ~28.7 GB. Speed holds above 52 tok/s.

**Coding focus:** Consider Qwen3-Coder-30B-A3B-Instruct (same architecture, optimized for multi-file agentic coding). Q5_K_M fits 128K context comfortably.
