"""
Stigmergic Router
=================

Trace-aware topology construction for DyTopo swarms.

Wraps the existing functional routing pipeline in ``router.py`` and adds
persistence of past swarm routing patterns ("traces") in Qdrant.  When a
new task arrives, similar historical traces are retrieved and blended into
the similarity matrix as a *boost*, so proven routing patterns are favoured
for recurring task families.

Key design constraints:
- Uses MiniLM-L6-v2 (384-dim) via the same lazy singleton as ``router.py``.
- All Qdrant operations are wrapped in try/except with graceful degradation
  so trace failures never crash a swarm.
- The boost matrix maps role-pairs (not agent IDs) from historical traces
  to current agent positions, making it portable across roster changes.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from typing import Any

import numpy as np

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from dytopo.router import (
    _get_routing_model,
    apply_threshold,
    compute_similarity_matrix,
    embed_descriptors,
    enforce_max_indegree,
)

logger = logging.getLogger("dytopo.stigmergic_router")


# ---------------------------------------------------------------------------
#  Helper: convert routing edges to trace format
# ---------------------------------------------------------------------------

def build_trace_edges(
    edges: list[tuple[str, str, float]],
    agent_roles: dict[str, str],
    round_num: int,
) -> list[dict]:
    """Convert routing edges to the trace-deposit format.

    Args:
        edges: List of ``(source_id, target_id, weight)`` tuples produced
            by :func:`dytopo.router.build_routing_result`.
        agent_roles: Mapping of ``agent_id -> role_name`` for the current
            swarm roster.
        round_num: The round number these edges belong to.

    Returns:
        List of dicts, each with keys ``sender``, ``receiver``,
        ``sender_role``, ``receiver_role``, ``weight``, and ``round``.
    """
    trace_edges: list[dict] = []
    for source_id, target_id, weight in edges:
        trace_edges.append({
            "sender": source_id,
            "receiver": target_id,
            "sender_role": agent_roles.get(source_id, source_id),
            "receiver_role": agent_roles.get(target_id, target_id),
            "weight": float(weight),
            "round": round_num,
        })
    return trace_edges


# ---------------------------------------------------------------------------
#  StigmergicRouter
# ---------------------------------------------------------------------------

class StigmergicRouter:
    """Trace-aware topology construction for DyTopo swarms.

    Wraps the existing functional routing pipeline and adds:

    - **Retrieval** of similar past traces from Qdrant before routing.
    - **Time-decayed boost matrix** blended with the cosine-similarity
      matrix before thresholding.
    - **Deposit** of successful swarm traces after completion so future
      runs benefit from the accumulated routing knowledge.

    All Qdrant operations degrade gracefully -- a warning is logged and
    the operation returns an empty / default value so that trace
    unavailability never crashes a swarm.

    Parameters
    ----------
    qdrant_url : str
        Qdrant gRPC / REST endpoint.
    model_name : str
        Name of the sentence-transformer model (must match router.py).
    enable_traces : bool
        Master switch -- when ``False``, ``build_topology`` falls back to
        the plain similarity-only pipeline.
    trace_boost_weight : float
        Blend weight for the trace boost matrix (``alpha`` in the formula
        ``S_blended = (1 - alpha) * S + alpha * B``).
    trace_half_life_hours : float
        Time-decay half-life for historical traces, in hours.  A 168-hour
        (1-week) half-life means a week-old trace contributes 50 % of its
        original boost.
    top_k : int
        Maximum number of similar traces to retrieve per query.
    min_quality : float
        Minimum ``final_answer_quality`` a swarm run must achieve before
        its trace is eligible for deposit.
    prune_max_age_hours : float
        Traces older than this are eligible for pruning.
    """

    COLLECTION = "swarm_traces"
    VECTOR_DIM = 384

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        model_name: str = "all-MiniLM-L6-v2",
        enable_traces: bool = True,
        trace_boost_weight: float = 0.15,
        trace_half_life_hours: float = 168.0,
        top_k: int = 5,
        min_quality: float = 0.5,
        prune_max_age_hours: float = 720.0,
    ) -> None:
        self._client = AsyncQdrantClient(url=qdrant_url)
        self._model_name = model_name
        self.enable_traces = enable_traces
        self.trace_boost_weight = trace_boost_weight
        self.trace_half_life_hours = trace_half_life_hours
        self.top_k = top_k
        self.min_quality = min_quality
        self.prune_max_age_hours = prune_max_age_hours
        self._collection_ready = False

    # ------------------------------------------------------------------
    # Collection bootstrap
    # ------------------------------------------------------------------

    async def ensure_collection(self) -> None:
        """Create the ``swarm_traces`` collection if it does not exist.

        Idempotent -- safe to call on every read or write.  Sets cosine
        distance and the correct vector dimension (384).
        """
        if self._collection_ready:
            return

        try:
            collections = await self._client.get_collections()
            existing_names = [c.name for c in collections.collections]

            if self.COLLECTION not in existing_names:
                await self._client.create_collection(
                    collection_name=self.COLLECTION,
                    vectors_config=VectorParams(
                        size=self.VECTOR_DIM,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection '%s'", self.COLLECTION)
            else:
                logger.debug(
                    "Qdrant collection '%s' already exists", self.COLLECTION
                )

            self._collection_ready = True

        except Exception:
            logger.warning(
                "Could not ensure Qdrant collection '%s' -- Qdrant may be unavailable",
                self.COLLECTION,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    @staticmethod
    def _embed(text: str) -> list[float]:
        """Embed *text* into a 384-dim vector using MiniLM-L6-v2.

        Reuses the lazy-loaded singleton from :func:`dytopo.router._get_routing_model`
        so that only one copy of the model is ever loaded.
        """
        model = _get_routing_model()
        vector = model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector.tolist()

    # ------------------------------------------------------------------
    # Time decay
    # ------------------------------------------------------------------

    def _time_decay(self, created_at: float) -> float:
        """Compute an exponential decay weight for a trace.

        .. math::

            w = 2^{-\\text{age\\_hours} / \\text{half\\_life\\_hours}}

        Returns a value in ``(0, 1]``.  Very old traces approach 0.
        """
        age_hours = (time.time() - created_at) / 3600.0
        if age_hours <= 0:
            return 1.0
        return float(2.0 ** (-age_hours / self.trace_half_life_hours))

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    async def _retrieve_traces(self, task_summary: str) -> list[dict]:
        """Find the top-k similar past traces from Qdrant.

        Each returned dict contains the full Qdrant payload augmented with
        ``score`` (cosine similarity to *task_summary*) and ``decay``
        (time-decay weight).

        Returns an empty list if Qdrant is unavailable or the collection
        is empty.
        """
        vector = self._embed(task_summary)

        try:
            await self.ensure_collection()

            results = await self._client.query_points(
                collection_name=self.COLLECTION,
                query=vector,
                limit=self.top_k,
                with_payload=True,
            )

            traces: list[dict] = []
            for hit in results.points:
                payload = dict(hit.payload) if hit.payload else {}
                payload["score"] = float(hit.score) if hit.score is not None else 0.0
                created_at = payload.get("created_at", 0.0)
                payload["decay"] = self._time_decay(created_at)
                traces.append(payload)

            logger.debug(
                "Retrieved %d traces for task summary (top score=%.3f)",
                len(traces),
                traces[0]["score"] if traces else 0.0,
            )
            return traces

        except Exception:
            logger.warning(
                "Failed to retrieve traces -- Qdrant may be unavailable",
                exc_info=True,
            )
            return []

    # ------------------------------------------------------------------
    # Boost matrix
    # ------------------------------------------------------------------

    def _compute_trace_boost(
        self,
        traces: list[dict],
        agent_ids: list[str],
        agent_roles: dict[str, str],
    ) -> np.ndarray | None:
        """Build an NxN boost matrix from historical traces.

        For each past trace, the method inspects which *role-pairs* had
        active edges.  Those role-pairs are mapped to the *current* agent
        positions and weighted by ``trace_similarity * time_decay * quality``.

        Args:
            traces: Payloads returned by :meth:`_retrieve_traces`, each
                containing ``active_edges``, ``score``, ``decay``, and
                ``final_answer_quality``.
            agent_ids: Ordered list of current agent IDs.
            agent_roles: Mapping of ``agent_id -> role_name`` for the
                current roster.

        Returns:
            An ``(N, N)`` float32 boost matrix normalised to ``[0, 1]``, or
            ``None`` if no applicable traces were found.
        """
        N = len(agent_ids)
        if N == 0:
            return None

        # Build reverse mapping: role_name -> list of indices
        role_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, aid in enumerate(agent_ids):
            role = agent_roles.get(aid, aid)
            role_to_indices[role].append(idx)

        boost = np.zeros((N, N), dtype=np.float64)
        total_weight = 0.0

        for trace in traces:
            edges = trace.get("active_edges", [])
            if not edges:
                continue

            score = trace.get("score", 0.0)
            decay = trace.get("decay", 1.0)
            quality = trace.get("final_answer_quality", 0.0)
            trace_weight = score * decay * quality
            if trace_weight <= 0:
                continue

            for edge in edges:
                sender_role = edge.get("sender_role", "")
                receiver_role = edge.get("receiver_role", "")
                edge_weight = edge.get("weight", 1.0)

                sender_indices = role_to_indices.get(sender_role, [])
                receiver_indices = role_to_indices.get(receiver_role, [])

                for si in sender_indices:
                    for ri in receiver_indices:
                        if si != ri:
                            # A[ri][si] convention: si -> ri (sender sends to receiver)
                            boost[ri, si] += trace_weight * edge_weight
                            total_weight += trace_weight * edge_weight

        if total_weight <= 0:
            return None

        # Normalise to [0, 1]
        max_val = boost.max()
        if max_val > 0:
            boost /= max_val

        return boost.astype(np.float32)

    # ------------------------------------------------------------------
    # Main topology builder
    # ------------------------------------------------------------------

    async def build_topology(
        self,
        agent_ids: list[str],
        descriptors: dict[str, dict],
        task_summary: str = "",
        threshold: float = 0.5,
        max_in_degree: int = 3,
    ) -> dict:
        """Full routing pipeline with optional trace boost.

        Steps:

        1. Embed descriptors and compute the cosine-similarity matrix
           using :func:`embed_descriptors` and :func:`compute_similarity_matrix`
           from ``router.py``.
        2. If traces are enabled and *task_summary* is provided:
           a. Retrieve similar past traces from Qdrant.
           b. Compute the trace boost matrix from historical role-pair edges.
           c. Blend:
              ``S_boosted = (1 - boost_weight) * S + boost_weight * boost``
        3. Apply threshold and enforce max in-degree.
        4. Build and return a result dict matching the format of
           :func:`build_routing_result`, with an added ``trace_context`` key.

        Args:
            agent_ids: Ordered list of agent IDs.
            descriptors: Dict mapping ``agent_id`` to
                ``{"key": str, "query": str}``.
            task_summary: Free-text description of the current task, used to
                retrieve similar traces.  If empty, trace retrieval is skipped.
            threshold: Cosine-similarity threshold (``tau``).
            max_in_degree: Maximum incoming edges per agent (``K_in``).

        Returns:
            Dict with the same keys as ``build_routing_result`` plus
            ``trace_context`` (dict with ``traces_retrieved``,
            ``boost_applied``, and ``traces_used``).
        """
        # Step 1 -- standard embedding + similarity
        key_vecs, query_vecs = embed_descriptors(agent_ids, descriptors)
        S = compute_similarity_matrix(query_vecs, key_vecs)

        # Step 2 -- optional trace boost
        trace_context: dict[str, Any] = {
            "traces_retrieved": 0,
            "boost_applied": False,
            "traces_used": 0,
        }

        if self.enable_traces and task_summary:
            # Build agent_roles mapping from descriptors or agent_ids
            # Use agent_id as fallback role name
            agent_roles: dict[str, str] = {}
            for aid in agent_ids:
                # Try to extract a role from the descriptor key, otherwise use id
                agent_roles[aid] = aid

            traces = await self._retrieve_traces(task_summary)
            trace_context["traces_retrieved"] = len(traces)

            if traces:
                boost = self._compute_trace_boost(traces, agent_ids, agent_roles)
                if boost is not None:
                    alpha = self.trace_boost_weight
                    S = ((1.0 - alpha) * S + alpha * boost).astype(np.float32)
                    # Re-zero diagonal after blending
                    np.fill_diagonal(S, 0.0)
                    trace_context["boost_applied"] = True
                    trace_context["traces_used"] = len(traces)
                    logger.info(
                        "Applied trace boost (alpha=%.2f) from %d traces",
                        alpha,
                        len(traces),
                    )

        # Step 3 -- threshold and degree cap
        A = apply_threshold(S, threshold)
        A, removed_idx = enforce_max_indegree(A, S, max_in_degree)

        # Step 4 -- build result dict (mirrors build_routing_result format)
        N = len(agent_ids)
        edges: list[tuple[str, str, float]] = []
        for i in range(N):
            for j in range(N):
                if A[i][j] == 1:
                    # A[i][j] == 1 means j -> i (j sends to i)
                    edges.append((agent_ids[j], agent_ids[i], float(S[i][j])))

        isolated = [agent_ids[i] for i in range(N) if A[i].sum() == 0]
        removed_named = [(agent_ids[i], agent_ids[j]) for i, j in removed_idx]

        active_sims = S[A == 1] if A.sum() > 0 else np.array([0.0])
        stats: dict[str, Any] = {
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
            "trace_context": trace_context,
        }

    # ------------------------------------------------------------------
    # Deposit
    # ------------------------------------------------------------------

    async def deposit_trace(
        self,
        task_summary: str,
        active_edges: list[dict],
        agent_roles: list[str],
        rounds_to_converge: int,
        final_answer_quality: float,
        convergence_method: str = "",
        task_domain: str = "",
    ) -> str:
        """Persist a completed swarm's routing pattern as a trace.

        A trace is only deposited if *final_answer_quality* meets or
        exceeds :attr:`min_quality`.  This ensures the trace store
        accumulates only high-quality routing patterns.

        Args:
            task_summary: Free-text description of the task.
            active_edges: List of edge dicts (as produced by
                :func:`build_trace_edges`) with keys ``sender``,
                ``receiver``, ``sender_role``, ``receiver_role``,
                ``weight``, ``round``.
            agent_roles: List of role names that participated.
            rounds_to_converge: Number of rounds the swarm ran.
            final_answer_quality: Quality score in ``[0, 1]``.
            convergence_method: How convergence was determined (e.g.
                ``"manager_halt"``, ``"convergence"``, ``"aegean_consensus"``).
            task_domain: The :class:`SwarmDomain` value (``"code"``,
                ``"math"``, ``"general"``).

        Returns:
            The UUID of the deposited trace point, or ``""`` on failure
            or if quality is below the minimum threshold.
        """
        if final_answer_quality < self.min_quality:
            logger.debug(
                "Trace not deposited: quality %.3f < min_quality %.3f",
                final_answer_quality,
                self.min_quality,
            )
            return ""

        vector = self._embed(task_summary)
        point_id = str(uuid.uuid4())

        payload: dict[str, Any] = {
            "task_summary": task_summary,
            "task_domain": task_domain,
            "agent_roles": agent_roles,
            "rounds_to_converge": rounds_to_converge,
            "active_edges": active_edges,
            "final_answer_quality": final_answer_quality,
            "convergence_method": convergence_method,
            "created_at": time.time(),
        }

        try:
            await self.ensure_collection()

            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )

            await self._client.upsert(
                collection_name=self.COLLECTION,
                points=[point],
            )

            logger.info(
                "Deposited trace %s (quality=%.2f, edges=%d, rounds=%d)",
                point_id,
                final_answer_quality,
                len(active_edges),
                rounds_to_converge,
            )
            return point_id

        except Exception:
            logger.warning(
                "Failed to deposit trace -- Qdrant may be unavailable",
                exc_info=True,
            )
            return ""

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_trace_stats(self) -> dict:
        """Return collection point count and configuration summary.

        Returns a dict with ``point_count``, ``collection``, and key
        configuration values.  Returns an empty-ish default on failure.
        """
        stats: dict[str, Any] = {
            "collection": self.COLLECTION,
            "point_count": 0,
            "enable_traces": self.enable_traces,
            "boost_weight": self.trace_boost_weight,
            "half_life_hours": self.trace_half_life_hours,
            "top_k": self.top_k,
            "min_quality": self.min_quality,
            "prune_max_age_hours": self.prune_max_age_hours,
        }

        try:
            await self.ensure_collection()
            info = await self._client.get_collection(self.COLLECTION)
            stats["point_count"] = info.points_count or 0
        except Exception:
            logger.warning(
                "Failed to fetch trace stats -- Qdrant may be unavailable",
                exc_info=True,
            )

        return stats

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    async def prune_old_traces(self) -> int:
        """Delete traces older than :attr:`prune_max_age_hours`.

        Uses a Qdrant filter on the ``created_at`` payload field to
        select traces that have exceeded the maximum age.

        Returns:
            The number of points deleted, or ``0`` on failure.
        """
        cutoff = time.time() - (self.prune_max_age_hours * 3600.0)

        try:
            await self.ensure_collection()

            # Get count before deletion for reporting
            info_before = await self._client.get_collection(self.COLLECTION)
            count_before = info_before.points_count or 0

            await self._client.delete(
                collection_name=self.COLLECTION,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="created_at",
                                range=Range(lt=cutoff),
                            ),
                        ]
                    )
                ),
            )

            info_after = await self._client.get_collection(self.COLLECTION)
            count_after = info_after.points_count or 0

            deleted = max(0, count_before - count_after)
            if deleted > 0:
                logger.info(
                    "Pruned %d old traces (cutoff=%.0f hours ago, %d remaining)",
                    deleted,
                    self.prune_max_age_hours,
                    count_after,
                )
            else:
                logger.debug("No traces to prune (all within age limit)")

            return deleted

        except Exception:
            logger.warning(
                "Failed to prune old traces -- Qdrant may be unavailable",
                exc_info=True,
            )
            return 0
