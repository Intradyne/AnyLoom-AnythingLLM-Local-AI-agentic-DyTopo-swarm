"""
DyTopo Semantic Router
======================

Embedding, similarity matrix computation, thresholding, and degree cap enforcement.

Key constraint: Uses MiniLM-L6-v2 (22M params, 384-dim) on CPU.
Tiny model (~80 MB), not worth GPU overhead.  BGE-M3 (used for RAG) is NOT used for routing.
"""

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("dytopo.router")

# Lazy singleton for MiniLM
_routing_model = None


def _get_routing_model():
    """Lazy-load MiniLM-L6-v2 for descriptor embedding. CPU — tiny model, not worth GPU."""
    global _routing_model
    if _routing_model is None:
        from sentence_transformers import SentenceTransformer
        _routing_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        logger.info("MiniLM-L6-v2 loaded for DyTopo routing on CPU (~80 MB)")
    return _routing_model


def prepare_descriptor_for_embedding(
    descriptor: str,
    agent_role: str = "",
    round_context: str = "",
) -> str:
    """Enrich a descriptor with role and context for better semantic routing.

    Prefixes the descriptor with role and context information to improve
    embedding separation between agents with similar descriptions.

    Args:
        descriptor: Raw descriptor text (key or query)
        agent_role: Agent's role name (e.g., "developer", "researcher")
        round_context: Brief context for the current round goal

    Returns:
        Enriched descriptor string, kept concise for MiniLM's 256-token limit
    """
    parts = []
    if agent_role:
        parts.append(f"As a {agent_role}:")
    if round_context:
        parts.append(f"[{round_context[:50]}]")
    parts.append(descriptor)
    return " ".join(parts)


def embed_descriptors(
    agent_ids: list[str],
    descriptors: dict[str, dict],  # {agent_id: {"key": str, "query": str}}
    agent_roles: dict[str, str] | None = None,
    round_context: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Embed all keys and queries. Returns (key_vectors, query_vectors), each shape (N, 384).

    When agent_roles is provided, descriptors are enriched via
    prepare_descriptor_for_embedding() before encoding to improve semantic
    separation between agents with similar raw descriptions.

    Args:
        agent_ids: Ordered list of agent IDs
        descriptors: Dict mapping agent_id to {"key": str, "query": str}
        agent_roles: Optional dict mapping agent_id to role name for enrichment
        round_context: Optional round context string for enrichment

    Returns:
        Tuple of (key_vectors, query_vectors), both shape (N, 384), dtype float32
    """
    model = _get_routing_model()
    keys = []
    queries = []
    for aid in agent_ids:
        key_text = descriptors[aid]["key"]
        query_text = descriptors[aid]["query"]
        if agent_roles and aid in agent_roles:
            key_text = prepare_descriptor_for_embedding(key_text, agent_roles[aid], round_context)
            query_text = prepare_descriptor_for_embedding(query_text, agent_roles[aid], round_context)
        keys.append(key_text)
        queries.append(query_text)

    # Batch encode — all keys then all queries in one call for efficiency
    all_texts = keys + queries
    all_vectors = model.encode(
        all_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    N = len(agent_ids)
    key_vectors = all_vectors[:N]      # shape (N, 384)
    query_vectors = all_vectors[N:]    # shape (N, 384)

    return key_vectors.astype(np.float32), query_vectors.astype(np.float32)


def compute_similarity_matrix(
    query_vectors: np.ndarray,
    key_vectors: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity matrix S where S[i][j] = sim(query_i, key_j).

    Since vectors are already L2-normalized by encode(), dot product = cosine similarity.

    Args:
        query_vectors: Shape (N, 384), normalized
        key_vectors: Shape (N, 384), normalized

    Returns:
        S with shape (N, N), self-loops zeroed out, dtype float32
    """
    S = query_vectors @ key_vectors.T  # (N, N) cosine similarities
    np.fill_diagonal(S, 0.0)           # no self-loops
    return S.astype(np.float32)


def apply_threshold(
    S: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Apply hard threshold: A[i][j] = 1 if S[i][j] >= tau, else 0.

    A[i][j] = 1 means agent j's key matches agent i's query.
    Edge direction: j → i (j sends to i).

    Args:
        S: Similarity matrix, shape (N, N)
        tau: Threshold value (typically 0.5 for MiniLM on well-crafted descriptors)

    Returns:
        Adjacency matrix A, shape (N, N), dtype int32
    """
    A = (S >= tau).astype(np.int32)
    np.fill_diagonal(A, 0)  # Ensure no self-loops
    return A


def enforce_max_indegree(
    A: np.ndarray,
    S: np.ndarray,
    K_in: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Cap each agent's incoming connections to the top-K_in by similarity.

    For each agent i, if it has more than K_in incoming edges, keep only the
    K_in with highest similarity scores.

    Args:
        A: Adjacency matrix, shape (N, N)
        S: Similarity matrix, shape (N, N)
        K_in: Maximum incoming edges per agent

    Returns:
        Tuple of (A_capped, removed_edges) where removed_edges is list of (receiver, sender) pairs
    """
    removed = []
    N = A.shape[0]
    for i in range(N):
        # Find all incoming edges to agent i (A[i][j] == 1 means j→i)
        active = np.where(A[i] == 1)[0]
        if len(active) > K_in:
            # Get similarities for all incoming edges
            sims = S[i, active]
            # Sort by similarity descending
            sorted_idx = np.argsort(sims)[::-1]
            # Remove edges beyond top K_in
            to_remove = active[sorted_idx[K_in:]]
            for j in to_remove:
                A[i, j] = 0
                removed.append((i, j))
    return A, removed


def build_routing_result(
    agent_ids: list[str],
    descriptors: dict[str, dict],
    tau: float,
    K_in: int,
    agent_roles: dict[str, str] | None = None,
    round_context: str = "",
) -> dict:
    """Full routing pipeline: embed → similarity → threshold → degree cap.

    When agent_roles and/or round_context are provided, descriptors are enriched
    before embedding to improve semantic separation between similar agents.

    Args:
        agent_ids: Ordered list of agent IDs
        descriptors: Dict mapping agent_id to {"key": str, "query": str}
        tau: Similarity threshold
        K_in: Max incoming edges per agent
        agent_roles: Optional dict mapping agent_id to role name for enrichment
        round_context: Optional round context string for enrichment

    Returns:
        Dict with keys:
            agent_ids: list[str]
            similarity_matrix: np.ndarray (N, N)
            adjacency_matrix: np.ndarray (N, N)
            edges: list[tuple[str, str, float]]  — (source_id, target_id, weight)
            isolated: list[str]  — agents with no incoming edges
            removed_edges: list[tuple[str, str]]
            stats: dict with edge_count, density, mean_sim, max_sim, min_active_sim
    """
    key_vecs, query_vecs = embed_descriptors(agent_ids, descriptors, agent_roles, round_context)
    S = compute_similarity_matrix(query_vecs, key_vecs)
    A = apply_threshold(S, tau)
    A, removed_idx = enforce_max_indegree(A, S, K_in)

    N = len(agent_ids)
    edges = []
    for i in range(N):
        for j in range(N):
            if A[i][j] == 1:
                # A[i][j] == 1 means j→i (j sends to i)
                edges.append((agent_ids[j], agent_ids[i], float(S[i][j])))

    isolated = [agent_ids[i] for i in range(N) if A[i].sum() == 0]
    removed_named = [(agent_ids[i], agent_ids[j]) for i, j in removed_idx]

    active_sims = S[A == 1] if A.sum() > 0 else np.array([0.0])
    stats = {
        "edge_count": int(A.sum()),
        "max_possible_edges": N * (N - 1),
        "density": float(A.sum()) / max(1, N * (N - 1)),
        "mean_similarity": float(S[S > 0].mean()) if (S > 0).any() else 0.0,
        "max_similarity": float(S.max()),
        "min_active_similarity": float(active_sims.min()) if len(active_sims) > 0 else 0.0,
        "isolated_count": len(isolated),
        "removed_edge_count": len(removed_idx),
    }

    return {
        "agent_ids": agent_ids,
        "similarity_matrix": S,
        "adjacency_matrix": A,
        "edges": edges,
        "isolated": isolated,
        "removed_edges": removed_named,
        "stats": stats,
    }


def validate_descriptor_separation(
    agent_ids: list[str],
    descriptors: dict[str, dict],
    agent_roles: dict[str, str] | None = None,
    round_context: str = "",
) -> dict:
    """Check if descriptors have sufficient semantic separation for routing.

    Warns about potential routing issues:
    - Key-key similarity > 0.7 (agents offering same thing)
    - All key-query similarities < 0.4 (no meaningful routing possible)

    Args:
        agent_ids: List of agent IDs
        descriptors: Agent descriptors with key/query
        agent_roles: Optional role mapping for enrichment
        round_context: Optional round context

    Returns:
        Dict with "warnings" list and "mean_similarity" float
    """
    key_vecs, query_vecs = embed_descriptors(agent_ids, descriptors, agent_roles, round_context)
    S = compute_similarity_matrix(query_vecs, key_vecs)

    # Key-key similarity (agents offering similar things)
    key_sim = key_vecs @ key_vecs.T
    np.fill_diagonal(key_sim, 0.0)

    warnings = []
    N = len(agent_ids)

    for i in range(N):
        for j in range(i + 1, N):
            if key_sim[i][j] > 0.7:
                warnings.append(
                    f"High key-key similarity ({key_sim[i][j]:.2f}) between "
                    f"{agent_ids[i]} and {agent_ids[j]} — they may offer redundant capabilities"
                )

    # Check if any meaningful routing is possible
    if S.max() < 0.4 and N > 1:
        warnings.append(
            f"Max query-key similarity is only {S.max():.2f} — "
            f"no edges will form at typical tau=0.5. Consider revising descriptors."
        )

    for w in warnings:
        logger.warning(f"Descriptor separation: {w}")

    mean_sim = float(S[S > 0].mean()) if (S > 0).any() else 0.0
    return {"warnings": warnings, "mean_similarity": mean_sim}


def log_routing_round(
    task_id: str,
    round_num: int,
    routing_result: dict,
    log_dir: str = "~/dytopo-logs",
    save_similarity_matrix: bool = True,
):
    """Persist routing data for debugging and analysis.

    Args:
        task_id: Unique task identifier
        round_num: Round number
        routing_result: Output from build_routing_result()
        log_dir: Directory to save logs
        save_similarity_matrix: Whether to include full similarity matrix in log
    """
    log_path = Path(log_dir).expanduser() / task_id
    log_path.mkdir(parents=True, exist_ok=True)

    record = {
        "task_id": task_id,
        "round": round_num,
        "timestamp": datetime.now().isoformat(),
        "stats": routing_result["stats"],
        "edges": routing_result["edges"],
        "isolated": routing_result["isolated"],
        "removed_edges": routing_result["removed_edges"],
    }

    if save_similarity_matrix and "similarity_matrix" in routing_result:
        record["similarity_matrix"] = routing_result["similarity_matrix"].tolist()

    filepath = log_path / f"round_{round_num:02d}_routing.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    logger.info(f"Routing log saved: {filepath}")
