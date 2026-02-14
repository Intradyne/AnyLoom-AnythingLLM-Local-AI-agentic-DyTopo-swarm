# LM Studio Settings

## Chat Model

**Model:** Qwen3-30B-A3B-Instruct-2507 (Q6_K)
**GGUF:** https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF

| Setting                 | Value          |
| ----------------------- | -------------- |
| Context Length          | 80000          |
| GPU Offload             | 48/48          |
| CPU Threads             | 12             |
| Eval Batch Size         | 2048           |
| Unified KV Cache        | On             |
| RoPE Freq Base          | Auto           |
| RoPE Freq Scale         | Auto           |
| Offload KV Cache to GPU | On             |
| Keep in Memory          | On             |
| Try mmap                | On             |
| Seed                    | Random         |
| Flash Attention         | On             |
| KV Cache Quant          | Q8_0 (K and V) |

| Sampling | Value |
|---|---|
| Temperature | 0.3 |
| Response Length Limit | Off |
| Context Overflow | RollingWindow |
| Stop Strings | (none) |
| Top K | 20 |
| Repeat Penalty | 1.0 (OFF) |
| Top P | 0.85 |
| Min P | 0.05 |
| Structured Output | Off |
| Speculative Decoding | Off |

See `qwen3-model.md` for model selection rationale and setting explanations.

---

## Embedding Model (co-loaded)

**Model:** ggml-org/bge-m3-Q8_0
**GGUF:** https://huggingface.co/ggml-org/bge-m3-GGUF

| Setting | Value |
|---|---|
| Context Length | 8192 |
| GPU Offload | 24/24 |
| Eval Batch Size | 2048 |

Load the embedding model first, then load the chat model. Both stay resident simultaneously. The embedding model serves AnythingLLM via the `/v1/embeddings` endpoint.

See `bge-m3-embedding.md` for model details, the second (CPU) embedding pipeline, and installation.

---

## API consumers

The `/v1/chat/completions` endpoint serves three callers:

| Caller | Temperature | Notes |
|---|---|---|
| LM Studio chat (user) | 0.3 (UI setting) | Interactive use with hybrid thinking |
| AnythingLLM | 0.1 (AnythingLLM setting) | Agentic tool calls need max determinism |
| DyTopo swarm agents | 0.1 or 0.3 (per-request) | Descriptors at 0.1, work output at 0.3 — set via API `temperature` param, overrides UI setting |

DyTopo agent calls use `AsyncOpenAI(base_url="http://localhost:1234/v1")` and set temperature per-request in the API body. These calls happen in the background when a swarm is running and share the same model/KV cache as interactive chat. See `dytopo-swarm.md` for DyTopo architecture details.

LLM mcp.json

```json
{
  "mcpServers": {
    "MCP_DOCKER": {
      "command": "docker",
      "args": [
        "mcp",
        "gateway",
        "run"
      ],
      "env": {
        "LOCALAPPDATA": "C:\\Users\\User\\AppData\\Local",
        "ProgramData": "C:\\ProgramData",
        "ProgramFiles": "C:\\Program Files"
      }
    },
    "qdrant-rag": {
      "command": "python",
      "args": [
        "C:\\Users\\User\\Qdrant-RAG+Agents\\src\\qdrant_mcp_server.py"
      ],
      "env": {
        "QDRANT_URL": "http://localhost:6334",
        "LMStudio_DOCS_DIR": "C:\\Users\\User\\Qdrant-RAG+Agents\\rag-docs\\lm-studio",
        "AnythingLLM_DOCS_DIR": "C:\\Users\\User\\Qdrant-RAG+Agents\\rag-docs\\anythingllm",
        "COLLECTION_NAME": "lmstudio_docs"
      }
    }
  }
}
```