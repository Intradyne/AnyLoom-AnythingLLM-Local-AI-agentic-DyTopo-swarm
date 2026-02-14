# BGE-M3 Embedding Model

**Author:** Beijing Academy of Artificial Intelligence (BAAI)
**Model Card:** https://huggingface.co/BAAI/bge-m3
**Architecture:** XLM-RoBERTa-large, 568M parameters
**Output Dimensions:** 1024
**Max Input:** 8192 tokens
**Capabilities:** Dense retrieval, sparse lexical retrieval, ColBERT multi-vector (sparse used, ColBERT disabled)

This stack runs BGE-M3 in two independent pipelines serving different systems. Both produce 1024-dim vectors in the same semantic space.

---

## Pipeline 1: GGUF in LM Studio (serves AnythingLLM)

**GGUF:** https://huggingface.co/ggml-org/bge-m3-GGUF
**File:** bge-m3-Q8_0.gguf (~635 MB)
**Output:** Dense vectors only (GGUF does not support sparse)
**Endpoint:** `http://localhost:1234/v1/embeddings`
**Serves:** AnythingLLM workspace RAG → Qdrant port 6333
**VRAM:** ~635 MB, co-loaded alongside the chat model

The GGUF stays loaded in LM Studio alongside the Qwen3 chat model. Load the embedding model first, then load the chat model. Both remain resident — no model swapping needed.

Q8_0 quantization is near-lossless for embedding models. Embedding quantization affects weight precision, not output dimensions — output is always 1024-dim float32.

---

## Pipeline 2: FlagEmbedding on CPU (serves MCP RAG server)

**Runtime:** FlagEmbedding Python library on CPU
**Output:** Dense (1024-dim) + sparse lexical vectors (hybrid)
**Serves:** Qdrant port 6334 via `qdrant_mcp_server.py`
**VRAM:** 0 GB
**System RAM:** ~2.3 GB (FP32 weights + PyTorch runtime)
**Disk:** ~1.1 GB (auto-downloaded to `~/.cache/huggingface/`)
**Latency:** ~50–200ms per embedding
**Cold start:** ~30–60s on first search (lazy-loaded, not at server startup)

### Installation

```bash
pip install FlagEmbedding torch --index-url https://download.pytorch.org/whl/cpu
pip install qdrant-client>=1.12.0 mcp[cli]>=1.0.0
```

The `--index-url` flag installs CPU-only PyTorch (~300 MB instead of ~2 GB with CUDA). Model weights auto-download from HuggingFace on first use.

### CPU threading

The MCP server sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` before importing torch, and calls `torch.set_num_threads()` at model load. Default is 8 threads — half the 9950X3D's 16 physical cores — to avoid starving LM Studio and Qdrant. Gradient computation is globally disabled (`torch.set_grad_enabled(False)`) since the server only does inference.

Configurable via `RAG_CPU_THREADS` environment variable in `mcp.json`.

### Coexistence with MiniLM-L6-v2

The same `qdrant_mcp_server.py` process also lazy-loads MiniLM-L6-v2 (~80 MB RAM) for DyTopo descriptor routing. Both models share the same dedicated `ThreadPoolExecutor(max_workers=2)` to keep CPU-bound embedding work off the async event loop's default pool. BGE-M3 and MiniLM are independent singletons — each has its own `threading.Lock()` for thread-safe initialization.

### Why this runs outside LM Studio

The GGUF embedding endpoint (Pipeline 1) returns dense vectors only. BGE-M3's sparse lexical vectors — the primary advantage for technical documentation RAG — require the full PyTorch model via FlagEmbedding with `return_sparse=True`. The MCP server uses Reciprocal Rank Fusion (RRF) in Qdrant to combine dense and sparse signals.

### Encoding configuration

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', device='cpu', use_fp16=False)
output = model.encode(
    texts,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False,
    max_length=1024,
    batch_size=16,
)
# output['dense_vecs']: ndarray [N, 1024]
# output['lexical_weights']: list of dicts {token_id: weight}
```

`max_length=1024` caps the token count sent to the encoder per chunk, not BGE-M3's full 8192-token window — saves substantial CPU compute. Chunks exceeding 1024 tokens embed only their first 1024 tokens. ColBERT disabled (50× storage overhead, marginal benefit for a small corpus).

`use_fp16=False` because FP16 requires CUDA. On CPU, the model runs FP32.

`batch_size=16` (up from 12) leverages the 9950X3D's 3D V-Cache for larger matrix multiplications per pass. Configurable via `RAG_EMBED_BATCH_SIZE`.

### Dependencies

- `torch>=2.0.0` (CPU-only variant)
- `FlagEmbedding>=1.2.0`
- `qdrant-client>=1.12.0`
- `mcp[cli]>=1.0.0`

---

## Why BGE-M3 for both pipelines

Using the same model family across both pipelines means AnythingLLM's Qdrant (port 6333) and the MCP server's Qdrant (port 6334) embed content into the same 1024-dim semantic space. Queries that retrieve well from one store produce comparable results from the other. The MCP server adds sparse vectors for hybrid search, but the dense component is identical.

## Why hybrid search matters (Pipeline 2)

The LM Studio reference documents are dense with exact technical identifiers: port numbers (6333, 6334), tool names (`sequential_thinking`, `create_entities`), config keys (`QDRANT_URL`, `contextLength`), model names. Dense embeddings compress these specific tokens into an averaged 1024-dim vector — a query for "port 6334" might not surface the right chunk if the dense embedding blends it with surrounding prose about Qdrant configuration generally. Sparse vectors assign individual learned weights to literal tokens like "6334", and RRF fusion promotes results that score well on both signals.

## Why not BAAI/bge-m3 in LM Studio search

The BAAI/bge-m3 repo on HuggingFace contains PyTorch/SafeTensors weights, not GGUF. LM Studio only indexes GGUF files, so it doesn't appear. The ggml-org conversion is the GGUF version that appears in LM Studio. FlagEmbedding (Pipeline 2) downloads the PyTorch weights directly from HuggingFace — no LM Studio involvement.

## Why not Qwen3-Embedding-0.6B

Qwen3-Embedding requires an `Instruct:` prefix on every query for optimal retrieval (Qwen team reports 1–5% performance drop without it). AnythingLLM sends raw text to the embedding endpoint without instruction formatting — no way to inject the prefix. BGE-M3 needs no instruction prefix and produces the same 1024-dim output at 8192-token context.

## Why not nomic-embed-text-v1.5

BGE-M3 provides 1024-dim vectors (vs nomic's 768), 8192-token context (vs nomic's 2048), and stronger multilingual/technical retrieval benchmarks. The 4× context increase means AnythingLLM's 6600-character chunks (~1650 tokens) embed with full attention and zero truncation. nomic would work but leaves quality on the table at no cost savings.
