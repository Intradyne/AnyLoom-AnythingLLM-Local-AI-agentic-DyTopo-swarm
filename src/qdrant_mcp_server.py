"""
Qdrant RAG + DyTopo Swarm MCP Server
=====================================
Hybrid dense+sparse search using BGE-M3 embeddings on CPU,
plus DyTopo multi-agent orchestration with semantic routing.

Architecture:
- BGE-M3 loads via FlagEmbedding on CPU (~2.3 GB system RAM, 0 GB VRAM)
- MiniLM-L6-v2 loads on CPU (~80 MB) for DyTopo descriptor routing
- LM Studio keeps all 32 GB VRAM for Qwen3-30B-A3B-Instruct-2507
- Qdrant collection uses named vectors: "dense" (1024-dim) + "sparse" (lexical)
- Search fuses both with Reciprocal Rank Fusion (RRF)
- Auto-indexes markdown docs from multiple source dirs with per-file incremental sync
- DyTopo swarm runs as async background task, calls LM Studio API directly

Hardware target: RTX 5090 (32 GB VRAM), Ryzen 9 9950X3D (16c/32t), 94 GB DDR5
File location:   C:\\Users\\User\\Qdrant-RAG+Agents\\src\\qdrant_mcp_server.py

DyTopo paper: arXiv 2602.06039, "Dynamic Topology Routing for Multi-Agent
Reasoning via Semantic Matching" (Lu et al., Feb 2026)

MCP Tools (8 total):
  rag_search(query, limit, source)       — hybrid semantic+lexical search
  rag_status()                           — collection info, indexed files, staleness
  rag_reindex(force)                     — trigger manual re-index
  rag_sources()                          — list configured doc sources + file counts
  rag_file_info(filename)                — per-file chunk count and hash
  swarm_start(task, domain, tau, ...)    — launch DyTopo multi-agent swarm
  swarm_status(task_id)                  — check swarm progress
  swarm_result(task_id)                  — retrieve completed swarm result

Dependencies:
  pip install FlagEmbedding torch --index-url https://download.pytorch.org/whl/cpu
  pip install qdrant-client>=1.12.0 mcp[cli]>=1.0.0
  pip install sentence-transformers>=3.0 networkx>=3.0 openai>=1.40
  pip install tenacity>=9.0 json-repair>=0.39
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from mcp.server.fastmcp import FastMCP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LOGGING — stderr only (stdout is JSON-RPC for MCP)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("qdrant-rag")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIGURATION — environment variables
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6334")
COLLECTION = os.environ.get("COLLECTION_NAME", "lmstudio_docs")

# Multi-source doc directories (canvas state: LMStudio + AnythingLLM)
_project_root = os.path.join(os.path.dirname(__file__), "..")
LMStudio_DOCS_DIR = os.environ.get(
    "LMStudio_DOCS_DIR",
    os.path.join(_project_root, "rag-docs", "lm-studio"),
)
AnythingLLM_DOCS_DIR = os.environ.get(
    "AnythingLLM_DOCS_DIR",
    os.path.join(_project_root, "rag-docs", "anythingllm"),
)

# Build source map: label → path (non-empty paths only)
DOC_SOURCES: dict[str, str] = {}
if LMStudio_DOCS_DIR:
    DOC_SOURCES["lmstudio"] = LMStudio_DOCS_DIR
if AnythingLLM_DOCS_DIR:
    DOC_SOURCES["anythingllm"] = AnythingLLM_DOCS_DIR

# State file lives next to the primary docs dir
STATE_FILE = os.path.join(LMStudio_DOCS_DIR, ".rag_state.json")

# CPU tuning — Ryzen 9 9950X3D: 16 physical cores, use half for embedding
CPU_THREADS = int(os.environ.get("RAG_CPU_THREADS", "8"))
EMBED_BATCH_SIZE = int(os.environ.get("RAG_EMBED_BATCH_SIZE", "16"))
EMBED_MAX_LENGTH = int(os.environ.get("RAG_EMBED_MAX_LENGTH", "1024"))
MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.005"))

# LLM endpoint for DyTopo swarm calls
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3-30b-a3b-instruct-2507")

# Apply CPU thread limits before any torch/numpy import path kicks in
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  THREAD-SAFE SINGLETONS — module-level locks (FIX #2: lock initialization)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_embedder_lock = threading.Lock()

_embedder = None      # BGE-M3 for RAG
_qdrant = None

# Dedicated thread pool for CPU-bound work (FIX #4: consistent executor)
_embed_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embed")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BGE-M3 EMBEDDING — CPU, lazy singleton for RAG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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


def _embed_texts_sync(texts: list[str]) -> dict:
    """Embed texts producing dense (1024-dim) + sparse (lexical weights)."""
    model = _get_embedder()
    output = model.encode(
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
        max_length=EMBED_MAX_LENGTH,
        batch_size=EMBED_BATCH_SIZE,
    )
    return {"dense": output["dense_vecs"], "sparse": output["lexical_weights"]}


async def _embed_texts(texts: list[str]) -> dict:
    """Run CPU-bound BGE-M3 embedding in dedicated thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_embed_executor, _embed_texts_sync, texts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  QDRANT CLIENT — async, lazy-initialized
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def _get_qdrant():
    """Get or create async Qdrant client (REST on port 6334)."""
    global _qdrant
    if _qdrant is None:
        from qdrant_client import AsyncQdrantClient
        _qdrant = AsyncQdrantClient(url=QDRANT_URL, timeout=30)
        logger.info(f"Qdrant client connected to {QDRANT_URL}")
    return _qdrant


async def _ensure_collection(client):
    """Create collection with named dense + sparse vectors if missing."""
    from qdrant_client import models

    if await client.collection_exists(COLLECTION):
        return

    logger.info(f"Creating collection '{COLLECTION}' with hybrid vectors")
    await client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=200),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8, quantile=0.99, always_ram=True
            )
        ),
    )
    # Payload indexes for filtered search
    await client.create_payload_index(
        COLLECTION, "source", models.PayloadSchemaType.KEYWORD
    )
    await client.create_payload_index(
        COLLECTION, "section_header", models.PayloadSchemaType.KEYWORD
    )
    # FIX #10: source_dir payload index for multi-source filtering
    await client.create_payload_index(
        COLLECTION, "source_dir", models.PayloadSchemaType.KEYWORD
    )
    logger.info(f"Collection '{COLLECTION}' created with 3 payload indexes")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MARKDOWN CHUNKING — section-aware, ## header boundaries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _chunk_markdown(filepath: Path, source_label: str) -> list[dict]:
    """Split markdown on ## headers. Tags each chunk with source_dir label."""
    content = filepath.read_text(encoding="utf-8")

    title_match = re.match(r"^#\s+(.+)", content, re.MULTILINE)
    doc_title = title_match.group(1).strip() if title_match else filepath.stem

    sections = re.split(r"(?=^## )", content, flags=re.MULTILINE)

    chunks = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section or len(section) < 80:
            continue

        header_match = re.match(r"^##\s+(.+)", section)
        section_header = (
            header_match.group(1).strip() if header_match else "Introduction"
        )

        if section.startswith("# "):
            chunk_text = section
        else:
            chunk_text = f"# {doc_title}\n\n{section}"

        chunks.append({
            "text": chunk_text,
            "source": filepath.name,
            "source_dir": source_label,
            "doc_title": doc_title,
            "section_header": section_header,
            "chunk_index": i,
        })

    return chunks


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AUTO-INDEXING — per-file incremental sync with .rag_state.json (v2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _file_hash(filepath: Path) -> str:
    """SHA-256 of a single file's contents, truncated to 16 hex chars."""
    return hashlib.sha256(filepath.read_bytes()).hexdigest()[:16]


def _deterministic_id(source_label: str, filename: str, section_header: str, chunk_index: int) -> str:
    """Deterministic UUID from source_label/filename/header/index for idempotent upserts."""
    key = f"{source_label}/{filename}::{section_header}::{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def _load_state() -> dict:
    """Load per-file hash state from .rag_state.json."""
    try:
        return json.loads(Path(STATE_FILE).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(state: dict):
    """Persist per-file hash state."""
    try:
        Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
        Path(STATE_FILE).write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.warning(f"Could not write state file: {e}")


def _scan_all_sources() -> dict[str, dict[str, Path]]:
    """Scan all configured doc sources. Returns {source_label: {filename: Path}}."""
    result = {}
    for label, dir_path in DOC_SOURCES.items():
        p = Path(dir_path)
        if p.exists():
            result[label] = {f.name: f for f in sorted(p.glob("*.md"))}
        else:
            logger.warning(f"Doc source '{label}' dir not found: {dir_path}")
    return result


async def _sync_index() -> tuple[bool, dict]:
    """Incremental sync: add/update changed files, remove deleted files.

    Returns (is_ready, stats_dict).
    State format (v2): { "source_label/filename": file_hash, ... }
    """
    all_sources = _scan_all_sources()
    if not all_sources:
        return False, {"error": "No doc sources configured or found"}

    # Build current file map: "label/name" → (Path, hash)
    current_files: dict[str, tuple[Path, str, str]] = {}
    for label, files in all_sources.items():
        for name, fpath in files.items():
            state_key = f"{label}/{name}"
            current_files[state_key] = (fpath, _file_hash(fpath), label)

    if not current_files:
        return False, {"error": "No .md files found in any source"}

    prev_state = _load_state()
    client = await _get_qdrant()
    await _ensure_collection(client)

    # Determine changes
    added = [k for k in current_files if k not in prev_state]
    modified = [k for k in current_files if k in prev_state and prev_state[k] != current_files[k][1]]
    removed = [k for k in prev_state if k not in current_files]
    changed = added + modified

    stats = {
        "added": len(added),
        "modified": len(modified),
        "removed": len(removed),
        "unchanged": len(current_files) - len(changed),
        "total_files": len(current_files),
    }

    if not changed and not removed:
        # Check collection actually has points
        try:
            info = await client.get_collection(COLLECTION)
            if info.points_count and info.points_count > 0:
                return True, stats
        except Exception:
            pass
        # Fall through to full re-index if collection empty
        changed = list(current_files.keys())

    from qdrant_client import models

    # Remove deleted files' points
    for state_key in removed:
        parts = state_key.split("/", 1)
        if len(parts) == 2:
            source_label, filename = parts
            try:
                await client.delete(
                    collection_name=COLLECTION,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="source", match=models.MatchValue(value=filename)
                                ),
                                models.FieldCondition(
                                    key="source_dir", match=models.MatchValue(value=source_label)
                                ),
                            ]
                        )
                    ),
                )
            except Exception as e:
                logger.warning(f"Failed to delete points for {state_key}: {e}")

    # Index changed files
    if changed:
        all_chunks = []
        for state_key in changed:
            fpath, fhash, label = current_files[state_key]
            chunks = _chunk_markdown(fpath, label)
            all_chunks.extend(chunks)

        if all_chunks:
            # Delete old points for changed files first
            for state_key in modified:
                parts = state_key.split("/", 1)
                if len(parts) == 2:
                    source_label, filename = parts
                    try:
                        await client.delete(
                            collection_name=COLLECTION,
                            points_selector=models.FilterSelector(
                                filter=models.Filter(
                                    must=[
                                        models.FieldCondition(
                                            key="source",
                                            match=models.MatchValue(value=filename),
                                        ),
                                        models.FieldCondition(
                                            key="source_dir",
                                            match=models.MatchValue(value=source_label),
                                        ),
                                    ]
                                )
                            ),
                        )
                    except Exception as e:
                        logger.warning(f"Delete before re-index for {state_key}: {e}")

            # Embed all chunks
            texts = [c["text"] for c in all_chunks]
            logger.info(f"Embedding {len(texts)} chunks from {len(changed)} files...")
            embeddings = await _embed_texts(texts)

            # Build points
            points = []
            for i, chunk in enumerate(all_chunks):
                dense_vec = embeddings["dense"][i].tolist()
                sparse_dict = embeddings["sparse"][i]
                sparse_indices = [int(k) for k in sparse_dict.keys()]
                sparse_values = list(sparse_dict.values())

                point_id = _deterministic_id(
                    chunk["source_dir"], chunk["source"],
                    chunk["section_header"], chunk["chunk_index"],
                )

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={
                            "dense": dense_vec,
                            "sparse": models.SparseVector(
                                indices=sparse_indices, values=sparse_values
                            ),
                        },
                        payload={
                            "content": chunk["text"],
                            "source": chunk["source"],
                            "source_dir": chunk["source_dir"],
                            "doc_title": chunk["doc_title"],
                            "section_header": chunk["section_header"],
                            "chunk_index": chunk["chunk_index"],
                        },
                    )
                )

            # Upsert in batches
            BATCH = 20
            for start in range(0, len(points), BATCH):
                await client.upsert(
                    collection_name=COLLECTION,
                    points=points[start: start + BATCH],
                )

            stats["chunks_indexed"] = len(points)
            logger.info(f"Indexed {len(points)} chunks from {len(changed)} files")

    # Save new state
    new_state = {k: current_files[k][1] for k in current_files}
    _save_state(new_state)

    return True, stats


async def _ensure_indexed() -> bool:
    """Ensure index is ready, sync if needed. Returns True if ready."""
    try:
        ready, stats = await _sync_index()
        return ready
    except Exception as e:
        logger.error(f"Index sync error: {e}", exc_info=True)
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HYBRID SEARCH — dense + sparse with RRF fusion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def _hybrid_search(
    query: str, limit: int = 5, source: str | None = None
) -> list:
    """Query with both dense and sparse vectors, fuse results with RRF."""
    from qdrant_client import models

    emb = await _embed_texts([query])
    dense_vec = emb["dense"][0].tolist()
    sparse_dict = emb["sparse"][0]
    sparse_indices = [int(k) for k in sparse_dict.keys()]
    sparse_values = list(sparse_dict.values())

    client = await _get_qdrant()

    # Optional source filename filter
    query_filter = None
    if source:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source", match=models.MatchValue(value=source)
                )
            ]
        )

    results = await client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(
                query=dense_vec, using="dense", limit=limit * 3,
                filter=query_filter,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_indices, values=sparse_values
                ),
                using="sparse", limit=limit * 3,
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )

    return results.points


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  JSON PARSING — Qwen3 response extraction pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _strip_thinking_tags(raw: str) -> str:
    """Remove Qwen3 <think>...</think> blocks from response."""
    stripped = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    return stripped.strip()


def _extract_json(raw: str) -> dict:
    """Full pipeline: strip thinking → find JSON → parse → repair → fallback."""
    content = _strip_thinking_tags(raw)

    # Direct parse
    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        pass

    # Markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # Brace-depth extraction for nested objects
    brace_start = content.find("{")
    if brace_start != -1:
        depth = 0
        for i in range(brace_start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = content[brace_start: i + 1]
                    try:
                        return json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        break

    # json-repair fallback
    try:
        from json_repair import repair_json
        repaired = repair_json(content, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass

    # Final fallback
    return {
        "key": "Agent produced unstructured output",
        "query": "Unable to determine information needs",
        "work": content if content else raw,
        "_parse_failed": True,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MCP SERVER INSTANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
mcp = FastMCP("qdrant-rag")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RAG TOOLS (5)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@mcp.tool()
async def rag_search(query: str, limit: int = 5, source: str = "") -> str:
    """Search reference docs with hybrid semantic+lexical search.

    Args:
        query: Natural language search query.
        limit: Max results (1-10, default 5).
        source: Optional filename filter (e.g., 'architecture.md').
    """
    limit = max(1, min(10, limit))
    source_filter = source.strip() or None

    try:
        indexed = await _ensure_indexed()
        if not indexed:
            return "No documents indexed. Check DOCS_DIR paths."
    except Exception as e:
        return f"Indexing error: {e}"

    try:
        points = await _hybrid_search(query, limit, source_filter)
    except Exception as e:
        return f"Search error: {e}"

    if not points:
        return "No matching documents found for this query."

    results = []
    for i, pt in enumerate(points, 1):
        payload = pt.payload or {}
        score = pt.score if pt.score is not None else 0.0
        if score < MIN_SCORE:
            continue
        source_name = payload.get("source", "unknown")
        source_dir = payload.get("source_dir", "")
        section = payload.get("section_header", "")
        content = payload.get("content", "")
        prefix = f"[{source_dir}] " if source_dir else ""
        results.append(
            f"--- Result {i} (score: {score:.3f}) ---\n"
            f"Source: {prefix}{source_name} > {section}\n"
            f"{content}"
        )

    return "\n\n".join(results) if results else "All results below minimum score threshold."


@mcp.tool()
async def rag_status() -> str:
    """Collection info, indexed files, and staleness check."""
    try:
        client = await _get_qdrant()
        lines = [f"Qdrant: {QDRANT_URL}", f"Collection: {COLLECTION}"]

        if await client.collection_exists(COLLECTION):
            info = await client.get_collection(COLLECTION)
            lines.append(f"Points: {info.points_count}")
            lines.append(f"Status: {info.status}")
        else:
            lines.append("Collection does not exist yet (will create on first search)")

        # Source dirs
        lines.append(f"\nDoc sources:")
        for label, dir_path in DOC_SOURCES.items():
            p = Path(dir_path)
            count = len(list(p.glob("*.md"))) if p.exists() else 0
            lines.append(f"  {label}: {dir_path} ({count} .md files)")

        # Staleness check
        prev_state = _load_state()
        if prev_state:
            all_sources = _scan_all_sources()
            current_keys = set()
            changes = []
            for label, files in all_sources.items():
                for name, fpath in files.items():
                    state_key = f"{label}/{name}"
                    current_keys.add(state_key)
                    fh = _file_hash(fpath)
                    if state_key not in prev_state:
                        changes.append(f"  + {state_key} (new)")
                    elif prev_state[state_key] != fh:
                        changes.append(f"  ~ {state_key} (modified)")
            for state_key in prev_state:
                if state_key not in current_keys:
                    changes.append(f"  - {state_key} (deleted)")

            if changes:
                lines.append(f"\nIndex is STALE — {len(changes)} change(s):")
                lines.extend(changes[:20])
                if len(changes) > 20:
                    lines.append(f"  ... and {len(changes) - 20} more")
            else:
                lines.append("\nIndex is UP TO DATE")
        else:
            lines.append("\nState file missing — next search will build full index")

        lines.append(f"\nCPU threads: {CPU_THREADS}")
        lines.append(f"Embed batch: {EMBED_BATCH_SIZE}, max length: {EMBED_MAX_LENGTH}")
        lines.append(f"Min score: {MIN_SCORE}")

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Status error: {e}", exc_info=True)
        return f"Status check error: {e}"


@mcp.tool()
async def rag_reindex(force: bool = False) -> str:
    """Trigger re-index. If force=True, delete collection and rebuild from scratch."""
    try:
        if force:
            client = await _get_qdrant()
            if await client.collection_exists(COLLECTION):
                await client.delete_collection(COLLECTION)
                logger.info("Collection deleted for forced re-index")
            # Clear state file
            _save_state({})

        ready, stats = await _sync_index()
        return f"Re-index {'complete' if ready else 'failed'}. Stats: {json.dumps(stats, indent=2)}"
    except Exception as e:
        return f"Re-index error: {e}"


@mcp.tool()
async def rag_sources() -> str:
    """List configured document sources and their file counts."""
    lines = []
    for label, dir_path in DOC_SOURCES.items():
        p = Path(dir_path)
        if p.exists():
            files = sorted(p.glob("*.md"))
            lines.append(f"{label} ({dir_path}): {len(files)} files")
            for f in files:
                lines.append(f"  {f.name} ({f.stat().st_size:,} bytes)")
        else:
            lines.append(f"{label} ({dir_path}): DIRECTORY NOT FOUND")
    return "\n".join(lines) if lines else "No doc sources configured."


@mcp.tool()
async def rag_file_info(filename: str) -> str:
    """Show per-file details: chunk count, hash, source directory."""
    state = _load_state()
    matches = []
    for state_key, fhash in state.items():
        if filename in state_key:
            matches.append(f"  {state_key}: hash={fhash}")

    if not matches:
        return f"No indexed file matching '{filename}'"

    # Count chunks in collection
    try:
        client = await _get_qdrant()
        from qdrant_client import models
        count = await client.count(
            collection_name=COLLECTION,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source", match=models.MatchValue(value=filename)
                    )
                ]
            ),
        )
        return f"File '{filename}':\n" + "\n".join(matches) + f"\nChunks in collection: {count.count}"
    except Exception as e:
        return f"File '{filename}':\n" + "\n".join(matches) + f"\nChunk count error: {e}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DYTOPO — Swarm task store + MCP tools
#  All orchestration logic lives in the dytopo/ package.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAX_TASKS = 20
_swarm_tasks: dict[str, dict] = {}


def _cleanup_tasks():
    """Evict oldest completed tasks when storage exceeds MAX_TASKS."""
    if len(_swarm_tasks) <= MAX_TASKS:
        return
    completed = sorted(
        [(k, v) for k, v in _swarm_tasks.items()
         if v.get("status") != "running"],
        key=lambda x: x[1].get("completed_at", 0),
    )
    for k, _ in completed[: len(completed) // 2]:
        del _swarm_tasks[k]


async def _log_progress(event_type: str, data: dict):
    """Progress callback — writes to stderr via logger."""
    logger.info("[DyTopo] %s: %s", event_type, data)


@mcp.tool()
async def swarm_start(
    task: str,
    domain: str = "code",
    tau: float = 0.3,
    k_in: int = 3,
    max_rounds: int = 5,
) -> str:
    """Launch a DyTopo multi-agent reasoning swarm. Returns a task_id to poll.

    DyTopo dynamically constructs agent communication topology each round
    using semantic similarity between agent descriptors. Agents only receive
    messages that are semantically relevant to their needs.

    Args:
        task: The problem to solve (code challenge, math proof, analysis, etc.)
        domain: Agent team — "code" (Developer/Researcher/Tester/Designer),
                "math" (ProblemParser/Solver/Verifier),
                or "general" (Analyst/Critic/Synthesizer)
        tau: Routing threshold 0.0-1.0 (default 0.3). Lower = more connections.
        k_in: Max incoming messages per agent per round (default 3).
        max_rounds: Maximum reasoning rounds before forced termination (default 5).
    """
    from dytopo.models import SwarmDomain, SwarmStatus, SwarmTask
    from dytopo.agents import get_worker_names
    from dytopo.orchestrator import run_swarm

    try:
        domain = domain.strip().lower()
        try:
            swarm_domain = SwarmDomain(domain)
        except ValueError:
            valid = ", ".join(d.value for d in SwarmDomain)
            return f"Unknown domain '{domain}'. Use: {valid}"

        tau = max(0.0, min(1.0, tau))
        k_in = max(1, min(5, k_in))
        max_rounds = max(1, min(10, max_rounds))

        _cleanup_tasks()

        swarm = SwarmTask(
            task=task,
            domain=swarm_domain,
            tau=tau,
            K_in=k_in,
            T_max=max_rounds,
        )
        _swarm_tasks[swarm.task_id] = {
            "status": "running",
            "swarm": swarm,
            "progress": "Initializing...",
            "wall_clock_sec": 0,
        }

        async def _safe_run():
            try:
                await run_swarm(swarm, on_progress=_log_progress)
                elapsed = (swarm.end_time or time.monotonic()) - swarm.start_time
                _swarm_tasks[swarm.task_id] = {
                    "status": "complete",
                    "swarm": swarm,
                    "wall_clock_sec": round(elapsed, 1),
                    "completed_at": time.monotonic(),
                }
            except Exception as e:
                logger.error("Swarm %s failed: %s", swarm.task_id, e, exc_info=True)
                _swarm_tasks[swarm.task_id] = {
                    "status": "error",
                    "swarm": swarm,
                    "error": str(e),
                    "wall_clock_sec": 0,
                    "completed_at": time.monotonic(),
                }

        asyncio.create_task(_safe_run())

        agent_names = ", ".join(get_worker_names(swarm_domain))
        return (
            f"Swarm launched: {swarm.task_id}\n"
            f"Domain: {domain} ({agent_names})\n"
            f"Config: tau={tau}, K_in={k_in}, max_rounds={max_rounds}\n"
            f"Use swarm_status('{swarm.task_id}') to check progress."
        )
    except Exception as e:
        logger.error("swarm_start failed: %s", e, exc_info=True)
        return f"Failed to start swarm: {e}"


@mcp.tool()
async def swarm_status(task_id: str) -> str:
    """Check progress of a running DyTopo swarm.

    Args:
        task_id: The ID returned by swarm_start.
    """
    task_id = task_id.strip()
    if task_id not in _swarm_tasks:
        active = ", ".join(_swarm_tasks.keys()) or "none"
        return f"No swarm found with ID '{task_id}'. Active IDs: {active}"

    info = _swarm_tasks[task_id]
    status = info.get("status", "unknown")
    swarm = info.get("swarm")

    if status == "running" and swarm:
        elapsed = time.monotonic() - swarm.start_time
        return (
            f"Status: RUNNING ({elapsed:.1f}s elapsed)\n"
            f"Rounds completed: {len(swarm.rounds)}/{swarm.T_max}\n"
            f"LLM calls: {swarm.total_llm_calls}\n"
            f"Progress: {swarm.progress_message}"
        )
    if status == "complete" and swarm:
        elapsed = info.get("wall_clock_sec", 0)
        return (
            f"Status: COMPLETE ({elapsed}s)\n"
            f"Rounds: {len(swarm.rounds)}\n"
            f"Termination: {swarm.termination_reason}\n"
            f"Tokens: {swarm.total_tokens}\n"
            f"Use swarm_result('{task_id}') to retrieve the full answer."
        )
    if status == "error":
        return (
            f"Status: ERROR\n"
            f"Error: {info.get('error', 'unknown')}"
        )
    return f"Status: {status}"


@mcp.tool()
async def swarm_result(task_id: str, include_topology: bool = True) -> str:
    """Retrieve completed swarm result with optional topology log.

    Args:
        task_id: The ID returned by swarm_start.
        include_topology: Include per-round routing graph details (default True).
    """
    from dytopo.agents import get_role_name
    from dytopo.models import AgentRole

    task_id = task_id.strip()
    if task_id not in _swarm_tasks:
        return f"No swarm found with ID '{task_id}'."

    info = _swarm_tasks[task_id]
    if info.get("status") == "running":
        return (
            f"Swarm '{task_id}' is still running. "
            "Use swarm_status to check progress."
        )
    if info.get("status") == "error":
        return f"Swarm '{task_id}' failed: {info.get('error', 'unknown')}"

    swarm = info.get("swarm")
    if not swarm:
        return f"No result data for swarm '{task_id}'."

    elapsed = (swarm.end_time or time.monotonic()) - swarm.start_time
    parts = [
        "=== DyTopo Swarm Result ===",
        f"Task: {swarm.task[:200]}",
        f"Domain: {swarm.domain.value}",
        f"Rounds: {len(swarm.rounds)}, LLM calls: {swarm.total_llm_calls}",
        f"Termination: {swarm.termination_reason}",
        f"Tokens: {swarm.total_tokens:,}",
        f"Wall time: {elapsed:.1f}s",
        "",
        swarm.final_answer or "[No answer produced]",
    ]

    # Add comprehensive metrics
    metrics = swarm.swarm_metrics
    parts.append("\n=== Swarm Metrics ===")
    parts.append(f"Total rounds: {metrics.total_rounds}")
    parts.append(f"Total LLM calls: {metrics.total_llm_calls}")
    parts.append(f"Total tokens: {metrics.total_tokens:,}")
    parts.append(f"Wall time: {metrics.total_wall_time_ms:,}ms ({metrics.total_wall_time_ms/1000:.1f}s)")
    parts.append(f"Agent failures: {metrics.agent_failures}")
    parts.append(f"Re-delegations: {metrics.redelegations}")

    if metrics.convergence_detected_at:
        parts.append(f"Convergence detected at round: {metrics.convergence_detected_at}")
    else:
        parts.append("Convergence: not detected")

    if metrics.routing_density_per_round:
        densities = [f"{d:.2f}" for d in metrics.routing_density_per_round]
        parts.append(f"Routing density per round: {', '.join(densities)}")
        avg_density = sum(metrics.routing_density_per_round) / len(metrics.routing_density_per_round)
        parts.append(f"Average routing density: {avg_density:.2f}")

    # Per-agent metrics if available
    if metrics.per_agent:
        parts.append("\n=== Per-Agent Metrics ===")
        for agent_id, agent_metrics in metrics.per_agent.items():
            parts.append(f"\n{agent_id}:")
            parts.append(f"  Success rate: {agent_metrics.success_rate:.1%}")
            parts.append(f"  Avg latency: {agent_metrics.avg_latency_ms:.0f}ms")
            parts.append(f"  Avg tokens: {agent_metrics.avg_tokens_per_round:.0f}")
            parts.append(f"  Times cited: {agent_metrics.times_cited}")
            parts.append(f"  Times isolated: {agent_metrics.times_isolated}")

    if include_topology:
        parts.append("\n=== Topology Log ===")
        for rnd in swarm.rounds:
            parts.append(f"--- Round {rnd.round_num} ---")
            parts.append(f"Goal: {rnd.goal}")
            if rnd.edges:
                parts.append(f"Edges ({len(rnd.edges)}):")
                for src, tgt, w in rnd.edges:
                    parts.append(f"  {src} -> {tgt} (sim: {w:.3f})")
            if rnd.execution_order:
                parts.append(f"Order: {' -> '.join(rnd.execution_order)}")
            if rnd.duration_sec:
                parts.append(f"Duration: {rnd.duration_sec:.1f}s")

    return "\n".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    logger.info(f"Starting Qdrant RAG + DyTopo MCP Server")
    logger.info(f"Qdrant: {QDRANT_URL}, Collection: {COLLECTION}")
    for label, path in DOC_SOURCES.items():
        logger.info(f"Doc source '{label}': {path}")
    logger.info(f"LLM endpoint: {LLM_BASE_URL} ({LLM_MODEL})")
    logger.info(f"CPU threads: {CPU_THREADS}, Embed batch: {EMBED_BATCH_SIZE}")
    mcp.run(transport="stdio")
