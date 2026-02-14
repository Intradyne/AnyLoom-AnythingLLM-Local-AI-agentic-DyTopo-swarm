"""Standalone re-embed script for port 6334 (LM Studio hybrid RAG).

Reads .md files from both RAG doc directories, chunks on ## headers,
embeds with BGE-M3 (dense + sparse), and upserts to Qdrant.

Usage:
    python src/reindex.py
"""

import asyncio
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6334")
COLLECTION = os.environ.get("COLLECTION_NAME", "lmstudio_docs")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOC_SOURCES = {
    "lmstudio": PROJECT_ROOT / "rag-docs" / "lm-studio",
    "anythingllm": PROJECT_ROOT / "rag-docs" / "anythingllm",
}

CPU_THREADS = int(os.environ.get("RAG_CPU_THREADS", "8"))
EMBED_BATCH_SIZE = int(os.environ.get("RAG_EMBED_BATCH_SIZE", "16"))
EMBED_MAX_LENGTH = int(os.environ.get("RAG_EMBED_MAX_LENGTH", "1024"))

os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_markdown(filepath: Path, source_label: str) -> list[dict]:
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
        section_header = header_match.group(1).strip() if header_match else "Introduction"
        chunk_text = section if section.startswith("# ") else f"# {doc_title}\n\n{section}"
        chunks.append({
            "text": chunk_text,
            "source": filepath.name,
            "source_dir": source_label,
            "doc_title": doc_title,
            "section_header": section_header,
            "chunk_index": i,
        })
    return chunks


def deterministic_id(source_label: str, filename: str, section_header: str, chunk_index: int) -> str:
    key = f"{source_label}/{filename}::{section_header}::{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print(f"Re-embed script for {QDRANT_URL} / collection '{COLLECTION}'")
    print(f"Doc sources: {', '.join(f'{k}: {v}' for k, v in DOC_SOURCES.items())}")

    # 1. Scan docs
    all_chunks = []
    for label, dir_path in DOC_SOURCES.items():
        if not dir_path.exists():
            print(f"  WARNING: {label} dir not found: {dir_path}")
            continue
        files = sorted(dir_path.glob("*.md"))
        print(f"  {label}: {len(files)} .md files")
        for f in files:
            chunks = chunk_markdown(f, label)
            all_chunks.extend(chunks)
            print(f"    {f.name}: {len(chunks)} chunks")

    if not all_chunks:
        print("ERROR: No chunks produced. Check doc directories.")
        sys.exit(1)

    print(f"\nTotal: {len(all_chunks)} chunks to embed")

    # 2. Load BGE-M3
    print("\nLoading BGE-M3 embedder (CPU, first load may download ~2.3 GB)...")
    import torch
    torch.set_grad_enabled(False)
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel("BAAI/bge-m3", device="cpu", use_fp16=False)
    torch.set_num_threads(CPU_THREADS)
    print("BGE-M3 loaded.")

    # 3. Embed all chunks
    texts = [c["text"] for c in all_chunks]
    print(f"\nEmbedding {len(texts)} chunks (batch_size={EMBED_BATCH_SIZE}, max_length={EMBED_MAX_LENGTH})...")
    output = model.encode(
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
        max_length=EMBED_MAX_LENGTH,
        batch_size=EMBED_BATCH_SIZE,
    )
    print("Embedding complete.")

    # 4. Connect to Qdrant and recreate collection
    from qdrant_client import AsyncQdrantClient, models
    client = AsyncQdrantClient(url=QDRANT_URL, timeout=60)

    if await client.collection_exists(COLLECTION):
        print(f"\nDeleting existing collection '{COLLECTION}'...")
        await client.delete_collection(COLLECTION)

    print(f"Creating collection '{COLLECTION}' with hybrid dense+sparse vectors...")
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
    await client.create_payload_index(COLLECTION, "source", models.PayloadSchemaType.KEYWORD)
    await client.create_payload_index(COLLECTION, "section_header", models.PayloadSchemaType.KEYWORD)
    await client.create_payload_index(COLLECTION, "source_dir", models.PayloadSchemaType.KEYWORD)

    # 5. Build and upsert points
    points = []
    for i, chunk in enumerate(all_chunks):
        dense_vec = output["dense_vecs"][i].tolist()
        sparse_dict = output["lexical_weights"][i]
        sparse_indices = [int(k) for k in sparse_dict.keys()]
        sparse_values = list(sparse_dict.values())

        point_id = deterministic_id(
            chunk["source_dir"], chunk["source"],
            chunk["section_header"], chunk["chunk_index"],
        )
        points.append(models.PointStruct(
            id=point_id,
            vector={
                "dense": dense_vec,
                "sparse": models.SparseVector(indices=sparse_indices, values=sparse_values),
            },
            payload={
                "content": chunk["text"],
                "source": chunk["source"],
                "source_dir": chunk["source_dir"],
                "doc_title": chunk["doc_title"],
                "section_header": chunk["section_header"],
                "chunk_index": chunk["chunk_index"],
            },
        ))

    BATCH = 20
    print(f"\nUpserting {len(points)} points in batches of {BATCH}...")
    for start in range(0, len(points), BATCH):
        batch = points[start:start + BATCH]
        await client.upsert(collection_name=COLLECTION, points=batch)
        print(f"  Batch {start // BATCH + 1}: {len(batch)} points")

    # 6. Save state file (compatible with MCP server's incremental sync)
    state_file = DOC_SOURCES["lmstudio"] / ".rag_state.json"
    state = {}
    for label, dir_path in DOC_SOURCES.items():
        if dir_path.exists():
            for f in sorted(dir_path.glob("*.md")):
                state_key = f"{label}/{f.name}"
                state[state_key] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    # 7. Verify
    info = await client.get_collection(COLLECTION)
    print(f"\nDone! Collection '{COLLECTION}' has {info.points_count} points.")
    print(f"State file saved: {state_file}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
