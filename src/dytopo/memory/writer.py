"""
Swarm Memory Writer
====================

Embeds and persists completed swarm results to Qdrant so that future
swarm runs can retrieve similar past solutions via semantic search.

Uses MiniLM-L6-v2 (384-dim) for embedding -- the same model used by
the DyTopo router, loaded via the same lazy singleton pattern.
"""

from __future__ import annotations

import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from dytopo.models import SwarmDomain, SwarmMemoryRecord

logger = logging.getLogger("dytopo.memory")

# ---------------------------------------------------------------------------
# Lazy singleton for MiniLM -- mirrors src/dytopo/router.py:_get_routing_model
# ---------------------------------------------------------------------------
_embedding_model = None


def _get_embedding_model():
    """Lazy-load MiniLM-L6-v2 for memory embedding. CPU -- tiny model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        logger.info("MiniLM-L6-v2 loaded for swarm memory embedding on CPU (~80 MB)")
    return _embedding_model


class SwarmMemoryWriter:
    """Write and query compressed swarm results in Qdrant.

    Every completed swarm run is embedded (task description + key findings)
    into a 384-dim vector and stored with its full metadata as payload.
    Future runs can call ``query_similar`` to find relevant prior solutions.

    Qdrant unavailability is handled gracefully -- a warning is logged and
    the operation returns without raising, so memory failures never crash
    a swarm.
    """

    COLLECTION_NAME = "swarm_memory"
    VECTOR_DIM = 384  # MiniLM-L6-v2 output dimension

    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self._client = AsyncQdrantClient(url=qdrant_url)
        self._collection_ready = False

    # ------------------------------------------------------------------
    # Collection bootstrap
    # ------------------------------------------------------------------

    async def ensure_collection(self) -> None:
        """Create the ``swarm_memory`` collection if it doesn't exist.

        Idempotent -- safe to call on every write.  Sets cosine distance
        and the correct vector dimension (384).
        """
        if self._collection_ready:
            return

        try:
            collections = await self._client.get_collections()
            existing_names = [c.name for c in collections.collections]

            if self.COLLECTION_NAME not in existing_names:
                await self._client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.VECTOR_DIM,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection '%s'", self.COLLECTION_NAME)
            else:
                logger.debug("Qdrant collection '%s' already exists", self.COLLECTION_NAME)

            self._collection_ready = True

        except Exception:
            logger.warning(
                "Could not ensure Qdrant collection '%s' -- Qdrant may be unavailable",
                self.COLLECTION_NAME,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    @staticmethod
    def _embed(text: str) -> list[float]:
        """Embed *text* into a 384-dim vector using MiniLM-L6-v2."""
        model = _get_embedding_model()
        vector = model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector.tolist()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def write(
        self,
        task_description: str,
        domain: str,
        agent_roles: list[str],
        round_count: int,
        key_findings: list[str],
        final_answer: str,
        convergence_achieved: bool,
        total_tokens: int,
        wall_time_ms: int,
        metadata: dict | None = None,
    ) -> str:
        """Compress and persist a swarm result. Returns the point ID.

        The task description and key findings are concatenated and embedded
        to form the search vector.  All fields are stored as a flat Qdrant
        payload so they are returned by ``query_similar``.

        If Qdrant is unreachable the error is logged and an empty string
        is returned -- the swarm continues unaffected.
        """
        record = SwarmMemoryRecord(
            task_description=task_description,
            domain=SwarmDomain(domain),
            agent_roles=agent_roles,
            round_count=round_count,
            key_findings=key_findings,
            final_answer_summary=final_answer,
            convergence_achieved=convergence_achieved,
            total_tokens=total_tokens,
            wall_time_ms=wall_time_ms,
            metadata=metadata or {},
        )

        # Build the embedding text
        embed_text = f"{task_description} {' '.join(key_findings)}"
        vector = self._embed(embed_text)

        try:
            await self.ensure_collection()

            point = PointStruct(
                id=record.id,
                vector=vector,
                payload=record.model_dump(mode="json"),
            )

            await self._client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[point],
            )

            logger.info("Persisted swarm memory point %s", record.id)
            return record.id

        except Exception:
            logger.warning(
                "Failed to write swarm memory -- Qdrant may be unavailable",
                exc_info=True,
            )
            return ""

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def query_similar(self, query: str, limit: int = 5) -> list[dict]:
        """Find similar past swarm results by semantic search.

        Embeds *query* and performs a nearest-neighbor search against the
        stored vectors, returning up to *limit* payloads ordered by
        descending similarity.

        Returns an empty list if Qdrant is unavailable.
        """
        vector = self._embed(query)

        try:
            await self.ensure_collection()

            results = await self._client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=vector,
                limit=limit,
                with_payload=True,
            )

            return [hit.payload for hit in results.points]

        except Exception:
            logger.warning(
                "Failed to query swarm memory -- Qdrant may be unavailable",
                exc_info=True,
            )
            return []
