# Embedding Performance Optimization Plan

## Purpose

Move all embedding inference from CPU to GPU. Currently the MCP RAG server loads BGE-M3 (568M params) on CPU via FlagEmbedding (~2.3 GB RAM, slow), and DyTopo routing loads MiniLM-L6-v2 on CPU. Both should run on the RTX 5090 GPU for 10-30x speedup.

## Hardware context

- GPU: RTX 5090, 32 GB VRAM
- Current VRAM usage: ~26 GB (Qwen3-30B-A3B Q4_K_M weights + KV cache + BGE-M3 GGUF container)
- Free VRAM: ~6 GB
- CPU: Ryzen 9 9950X3D (16c/32t), 94 GB DDR5
- OS: Windows (native Python, Docker containers via WSL2)

## VRAM budget for this change

| Component | Current | After | Delta |
|---|---|---|---|
| BGE-M3 FlagEmbedding (MCP RAG) | 0 MB VRAM (CPU) | ~1,200 MB (fp16 GPU) | +1,200 MB |
| MiniLM-L6-v2 (DyTopo routing) | 0 MB VRAM (CPU) | ~50 MB (fp16 GPU) | +50 MB |
| **Total new VRAM** | | | **+1,250 MB** |
| **New total** | | **~27.25 GB / 32 GB** | 85% util |

Comfortable headroom. The existing Docker llama.cpp embedding container (BGE-M3 GGUF, ~635 MB VRAM) stays as-is for AnythingLLM.

---

## Change 1: Move BGE-M3 FlagEmbedding to GPU (fp16)

**File:** `C:\Users\User\AnyLoom\src\qdrant_mcp_server.py`

This is the single biggest performance win. CPU inference on a 568M-param transformer is the bottleneck for every `rag_search`, `rag_reindex`, and `swarm_start` with context.

### 1a. Change the `_get_embedder()` function (around line 122-138)

Replace:

```python
def _get_embedder():
    """Lazy-load BGE-M3 on first use. Thread-safe via module-level lock."""
    global _embedder
    if _embedder is not None:
        return _embedder
    with _embedder_lock:
        if _embedder is None:
            logger.info("Loading BGE-M3 on CPU (first use — may take 30-60s)...")
            from FlagEmbedding import BGEM3FlagModel
            import torch
            torch.set_grad_enabled(False)
            _embedder = BGEM3FlagModel(
                "BAAI/bge-m3", device="cpu", use_fp16=False,
            )
            torch.set_num_threads(CPU_THREADS)
            logger.info("BGE-M3 loaded successfully")
    return _embedder
```

With:

```python
def _get_embedder():
    """Lazy-load BGE-M3 on first use. Thread-safe via module-level lock."""
    global _embedder
    if _embedder is not None:
        return _embedder
    with _embedder_lock:
        if _embedder is None:
            import torch
            if torch.cuda.is_available():
                _device = "cuda"
                _fp16 = True
                logger.info("Loading BGE-M3 on GPU (fp16, first use — may take 10-20s)...")
            else:
                _device = "cpu"
                _fp16 = False
                logger.info("Loading BGE-M3 on CPU fallback (first use — may take 30-60s)...")
            from FlagEmbedding import BGEM3FlagModel
            torch.set_grad_enabled(False)
            _embedder = BGEM3FlagModel(
                "BAAI/bge-m3", device=_device, use_fp16=_fp16,
            )
            if _device == "cpu":
                torch.set_num_threads(CPU_THREADS)
            logger.info(f"BGE-M3 loaded on {_device} successfully")
    return _embedder
```

### 1b. Increase batch size for GPU throughput (around line 94)

Change:

```python
EMBED_BATCH_SIZE = int(os.environ.get("RAG_EMBED_BATCH_SIZE", "16"))
EMBED_MAX_LENGTH = int(os.environ.get("RAG_EMBED_MAX_LENGTH", "1024"))
```

To:

```python
EMBED_BATCH_SIZE = int(os.environ.get("RAG_EMBED_BATCH_SIZE", "64"))
EMBED_MAX_LENGTH = int(os.environ.get("RAG_EMBED_MAX_LENGTH", "8192"))
```

Rationale: GPU has enough memory to batch 64 sequences at once (vs 16 on CPU). BGE-M3 supports 8192 token context — no reason to truncate at 1024 when GPU can handle full-length chunks.

### 1c. Update the docstring header (around line 88-92)

Change:

```
Hybrid dense+sparse search using BGE-M3 embeddings on CPU,
```

To:

```
Hybrid dense+sparse search using BGE-M3 embeddings on GPU (fp16, CUDA),
```

And change:

```
- BGE-M3 loads via FlagEmbedding on CPU (~2.3 GB system RAM, 0 GB VRAM)
```

To:

```
- BGE-M3 loads via FlagEmbedding on GPU (fp16, ~1.2 GB VRAM)
```

### 1d. Update the dependencies comment (around line 117)

Change:

```
  pip install FlagEmbedding torch --index-url https://download.pytorch.org/whl/cpu
```

To:

```
  pip install FlagEmbedding torch  # CUDA torch required for GPU embedding
```

### 1e. Update `rag_status()` output (around the end of that function)

The status tool currently reports CPU threads and embed batch. Find the lines:

```python
        lines.append(f"\nCPU threads: {CPU_THREADS}")
        lines.append(f"Embed batch: {EMBED_BATCH_SIZE}, max length: {EMBED_MAX_LENGTH}")
```

Replace with:

```python
        import torch
        _dev = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        lines.append(f"\nEmbedding device: {_dev}")
        lines.append(f"Embed batch: {EMBED_BATCH_SIZE}, max length: {EMBED_MAX_LENGTH}")
```

---

## Change 2: Move MiniLM-L6-v2 routing to GPU

**File:** `C:\Users\User\AnyLoom\src\dytopo\router.py`

Smaller win but free — 50 MB VRAM for faster per-round routing during swarm execution.

### 2a. Change `_get_routing_model()` (around line 24-31)

Replace:

```python
def _get_routing_model():
    """Lazy-load MiniLM-L6-v2 for descriptor embedding. ~80 MB RAM, <1s load."""
    global _routing_model
    if _routing_model is None:
        from sentence_transformers import SentenceTransformer
        _routing_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        logger.info("MiniLM-L6-v2 loaded for DyTopo routing (~80 MB)")
    return _routing_model
```

With:

```python
def _get_routing_model():
    """Lazy-load MiniLM-L6-v2 for descriptor embedding. GPU if available."""
    global _routing_model
    if _routing_model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _routing_model = SentenceTransformer("all-MiniLM-L6-v2", device=_device)
        logger.info(f"MiniLM-L6-v2 loaded for DyTopo routing on {_device}")
    return _routing_model
```

### 2b. Update the file docstring (line 7)

Change:

```
Key constraint: Uses MiniLM-L6-v2 (22M params, 384-dim) on CPU.
```

To:

```
Key constraint: Uses MiniLM-L6-v2 (22M params, 384-dim) on GPU (CUDA) with CPU fallback.
```

---

## Change 3: Tune the Docker llama.cpp embedding container

**File:** `C:\Users\User\AnyLoom\docker-compose.yml`

This is the embedding path used by AnythingLLM for workspace RAG. The current config is functional but not tuned.

### 3a. Update the embedding service command block (around line 74-80)

Replace:

```yaml
    command: >
      --model /models/bge-m3-q8_0.gguf
      --embeddings
      --ctx-size 8192
      --batch-size 8192
      --n-gpu-layers 99
      --host 0.0.0.0
      --port 8080
```

With:

```yaml
    command: >
      --model /models/bge-m3-q8_0.gguf
      --embeddings
      --ctx-size 8192
      --batch-size 8192
      --ubatch-size 2048
      --n-gpu-layers 99
      --flash-attn
      --parallel 4
      --host 0.0.0.0
      --port 8080
```

What each flag does:
- `--flash-attn`: Enables flash attention — reduces memory bandwidth bottleneck, faster inference.
- `--parallel 4`: Handles up to 4 concurrent embedding requests (AnythingLLM may send parallel chunk batches during document ingestion).
- `--ubatch-size 2048`: Physical CUDA batch size (default 512). Larger = better GPU utilization for long sequences.

---

## Change 4: Install CUDA-enabled PyTorch

The current install uses CPU-only torch:
```
pip install FlagEmbedding torch --index-url https://download.pytorch.org/whl/cpu
```

The RTX 5090 (Blackwell, SM_100) requires CUDA 12.8+ and PyTorch 2.6+.

### 4a. Uninstall CPU torch, install CUDA torch

Run in the Python environment that runs `qdrant_mcp_server.py`:

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

If `cu128` wheels are not yet available for the installed Python version, fall back to:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

The key requirement is that `torch.cuda.is_available()` returns `True` after installation.

### 4b. Verify CUDA torch is working

```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
```

Expected output: `CUDA: True, Device: NVIDIA GeForce RTX 5090`

If this fails, the code falls back to CPU gracefully (both `_get_embedder()` and `_get_routing_model()` have CPU fallback paths).

### 4c. Also install CUDA-aware sentence-transformers

sentence-transformers uses torch under the hood. Once torch is CUDA-enabled, sentence-transformers automatically uses GPU when `device="cuda"` is passed. No separate install needed — just verify torch has CUDA.

---

## Change 5: Update documentation

### 5a. `C:\Users\User\AnyLoom\.env`

Add optional override for embedding device. Append:

```env
# ============================================================================
# Embedding Performance Tuning
# ============================================================================
# RAG_EMBED_BATCH_SIZE=64       # GPU batch size (default 64, was 16 on CPU)
# RAG_EMBED_MAX_LENGTH=8192     # BGE-M3 full context (default 8192, was 1024)
```

### 5b. `C:\Users\User\AnyLoom\README.md`

In the Architecture section where BGE-M3 is mentioned, update any reference from "CPU" to "GPU (fp16, CUDA)".

### 5c. `C:\Users\User\AnyLoom\INSTALL.md`

Update the dependencies section to note CUDA torch is required:

Change the torch install instruction from:
```
pip install FlagEmbedding torch --index-url https://download.pytorch.org/whl/cpu
```
To:
```
pip install FlagEmbedding torch --index-url https://download.pytorch.org/whl/cu128
```

Add a note: "CUDA-enabled PyTorch is required for GPU embedding. The RTX 5090 requires cu128 (CUDA 12.8). If cu128 wheels are unavailable, try cu124."

---

## Verification steps

After all changes, verify:

1. **CUDA torch works:**
   ```
   python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
   ```

2. **MCP server starts without error:**
   ```
   python src/qdrant_mcp_server.py
   ```
   Look for log line: `BGE-M3 loaded on cuda successfully`

3. **rag_status reports GPU:**
   Call `rag_status()` tool — should show `Embedding device: GPU (CUDA)`

4. **rag_search works end-to-end:**
   Call `rag_search("test query")` — should return results

5. **Docker embedding container healthy:**
   ```
   docker compose up -d embedding
   docker compose logs embedding --tail 20
   ```
   Look for: flash attention enabled, 4 parallel slots

6. **VRAM usage is within budget:**
   ```
   nvidia-smi
   ```
   Total should be ~27-28 GB / 32 GB

7. **DyTopo routing on GPU:**
   Start a swarm and check logs for: `MiniLM-L6-v2 loaded for DyTopo routing on cuda`

---

## What NOT to change

- Do NOT remove the Docker llama.cpp embedding container — AnythingLLM depends on it for workspace RAG.
- Do NOT change the Qdrant collection schema or vector dimensions — existing indexed data must remain valid.
- Do NOT change any MCP tool signatures — callers must not break.
- Do NOT remove CPU fallback paths — graceful degradation if CUDA is unavailable.
- Do NOT change the `_embed_executor` ThreadPoolExecutor pattern — async wrapping of GPU calls still needs run_in_executor to avoid blocking the event loop.

## Future optimization (not in this change)

Consolidate the two BGE-M3 instances (Docker GGUF + Python FlagEmbedding) into a single GPU process by building an OpenAI-compatible embedding endpoint wrapper around FlagEmbedding. This would save ~635 MB VRAM by eliminating the Docker container. Deferred because it requires AnythingLLM to point at a new endpoint and adds operational complexity.

---

## Files modified by this plan

- `C:\Users\User\AnyLoom\src\qdrant_mcp_server.py` (updated — GPU embedding, batch size, docstrings)
- `C:\Users\User\AnyLoom\src\dytopo\router.py` (updated — GPU routing model)
- `C:\Users\User\AnyLoom\docker-compose.yml` (updated — flash-attn, parallel 4, ubatch-size)
- `C:\Users\User\AnyLoom\.env` (updated — embedding tuning comments)
- `C:\Users\User\AnyLoom\README.md` (updated — GPU embedding references)
- `C:\Users\User\AnyLoom\INSTALL.md` (updated — CUDA torch install instructions)
