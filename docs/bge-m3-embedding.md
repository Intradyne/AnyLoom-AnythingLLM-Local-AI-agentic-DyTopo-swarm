# BGE-M3 Embedding Model

**Author:** Beijing Academy of Artificial Intelligence (BAAI)
**Model Card:** https://huggingface.co/BAAI/bge-m3
**Architecture:** XLM-RoBERTa-large, 568M parameters
**Output Dimensions:** 1024
**Max Input:** 8192 tokens
**Capabilities:** Dense retrieval, sparse lexical retrieval, ColBERT multi-vector (sparse used, ColBERT disabled)

This stack uses BGE-M3 for hybrid RAG with dense + sparse vectors, providing superior retrieval quality for both AnythingLLM and MCP RAG pipelines.

---

## Single Hybrid Pipeline: ONNX INT8 on CPU

**Runtime:** sentence-transformers ONNX INT8 backend on CPU (AVX-512 VNNI)
**Output:** Dense (1024-dim, ONNX) + TF-sparse lexical vectors (CRC32 hash-based)
**Serves:** Qdrant port 6333 via `qdrant_mcp_server.py` (AnythingLLM + MCP)
**VRAM:** 0 (CPU-only, ~0.6 GB RAM)
**System RAM:** ~0.6 GB (ONNX Runtime)
**Disk:** ~1.1 GB (auto-downloaded to `~/.cache/huggingface/`)
**Latency:** ~15–50ms per query embedding
**Cold start:** ~30–60s on first search (lazy-loaded, not at server startup)

The single Qdrant instance on port 6333 uses hybrid dense+sparse indexing for all document sources. Both AnythingLLM workspace queries and MCP RAG tool searches benefit from RRF fusion.

### Installation

```bash
pip install sentence-transformers[onnx] onnxruntime  # ONNX INT8 CPU embedding
pip install qdrant-client>=1.12.0 mcp[cli]>=1.0.0
```

Uses ONNX Runtime for INT8 inference on CPU. No GPU or PyTorch required. Model weights auto-download from HuggingFace on first use.

### CPU threading

The MCP server sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` before loading ONNX Runtime, and configures `intra_op_num_threads` in the ONNX session options. Default is 16 threads — all of the 9950X3D's 16 physical cores — to maximize throughput since embedding runs on CPU only (no GPU contention). Gradient computation is not needed (ONNX Runtime is inference-only).

Configurable via `RAG_CPU_THREADS` environment variable in `mcp.json`.

### Coexistence with MiniLM-L6-v2

The same `qdrant_mcp_server.py` process also lazy-loads MiniLM-L6-v2 (~80 MB RAM) for DyTopo descriptor routing on CPU. Both models share the same dedicated `ThreadPoolExecutor(max_workers=2)` to keep CPU-bound embedding work off the async event loop's default pool. BGE-M3 and MiniLM are independent singletons — each has its own `threading.Lock()` for thread-safe initialization. Both run on CPU with zero VRAM usage.

### Why ONNX INT8 (vs FlagEmbedding or GGUF)

BGE-M3's ONNX export only produces dense vectors (the learned sparse head isn't exported). Sparse/lexical matching is provided by a lightweight TF-weighted CRC32 hash approach that catches exact keywords and identifiers. This is faster than FlagEmbedding (which requires PyTorch + CUDA) and eliminates GPU contention during active LLM inference. ONNX INT8 on the Ryzen 9950X3D's AVX-512 VNNI gives 3-5x speedup over PyTorch FP32.

### Encoding configuration

```python
from sentence_transformers import SentenceTransformer
import onnxruntime as ort

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 16
sess_options.inter_op_num_threads = 2
model = SentenceTransformer(
    "BAAI/bge-m3", backend="onnx",
    model_kwargs={"provider": "CPUExecutionProvider", "session_options": sess_options},
)
dense_vecs = model.encode(texts, normalize_embeddings=True)
# dense_vecs: ndarray [N, 1024]
# Sparse vectors generated separately via TF-weighted CRC32 hashing
```

ONNX INT8 inference is significantly faster than PyTorch, making the full 8192 context practical for document indexing. For queries, `max_length=256` is used since queries rarely exceed 50 tokens. ColBERT disabled (50x storage overhead, marginal benefit for a small corpus).

`batch_size=16` balances throughput and memory on the ONNX INT8 pipeline. Configurable via `RAG_EMBED_BATCH_SIZE`.

### Dependencies

- `sentence-transformers[onnx]>=3.0`
- `onnxruntime>=1.17`
- `qdrant-client>=1.12.0`
- `mcp[cli]>=1.0.0`

---

## Why hybrid search matters

The reference documents are dense with exact technical identifiers: port numbers (6333, 8000), tool names (`sequential_thinking`, `create_entities`), config keys (`QDRANT_URL`, `contextLength`), model names. Dense embeddings compress these specific tokens into an averaged 1024-dim vector — a query for "port 6333" might not surface the right chunk if the dense embedding blends it with surrounding prose about Qdrant configuration generally. Sparse vectors assign individual learned weights to literal tokens like "6333", and RRF fusion promotes results that score well on both signals.

This hybrid approach benefits both AnythingLLM workspace queries and MCP RAG tool searches, providing superior retrieval quality compared to dense-only embeddings.

## Why not Qwen3-Embedding-0.6B

Qwen3-Embedding requires an `Instruct:` prefix on every query for optimal retrieval (Qwen team reports 1–5% performance drop without it). AnythingLLM sends raw text to the embedding endpoint without instruction formatting — no way to inject the prefix. BGE-M3 needs no instruction prefix and produces the same 1024-dim output at 8192-token context.

## Why not nomic-embed-text-v1.5

BGE-M3 provides 1024-dim vectors (vs nomic's 768), 8192-token context (vs nomic's 2048), and stronger multilingual/technical retrieval benchmarks. The 4x context increase means AnythingLLM's 2500-character chunks (~625 tokens) embed with full attention and zero truncation. nomic would work but leaves quality on the table at no cost savings.
