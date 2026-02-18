"""
Swarm Memory Tests
==================

Tests for the SwarmMemoryWriter and SwarmMemoryRecord.

Run:
    pytest tests/test_swarm_memory.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from dytopo.models import SwarmDomain, SwarmMemoryRecord
from dytopo.memory.writer import SwarmMemoryWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_kwargs() -> dict:
    """Return a minimal set of kwargs for ``SwarmMemoryWriter.write``."""
    return {
        "task_description": "Solve the traveling salesman problem for 10 cities",
        "domain": "code",
        "agent_roles": ["manager", "developer", "tester"],
        "round_count": 3,
        "key_findings": [
            "Dynamic programming approach chosen",
            "O(n^2 * 2^n) complexity",
            "Optimal tour found for all test cases",
        ],
        "final_answer": "def solve_tsp(cities): ...",
        "convergence_achieved": True,
        "total_tokens": 12500,
        "wall_time_ms": 8400,
        "metadata": {"model": "qwen-2.5-coder", "temperature": 0.7},
    }


def _fake_vector(dim: int = 384) -> list[float]:
    """Return a deterministic fake embedding vector."""
    return [0.01] * dim


def _make_mock_qdrant_client(*, collections=None, query_results=None):
    """Build a fully-wired AsyncMock standing in for AsyncQdrantClient.

    Parameters
    ----------
    collections : list[str] | None
        Names of collections that already exist.  ``None`` means empty.
    query_results : list[MagicMock] | None
        Fake scored-point objects returned by ``query_points``.
    """
    client = AsyncMock()

    # -- get_collections
    coll_response = MagicMock()
    if collections:
        coll_objs = []
        for name in collections:
            obj = MagicMock()
            obj.name = name
            coll_objs.append(obj)
        coll_response.collections = coll_objs
    else:
        coll_response.collections = []
    client.get_collections = AsyncMock(return_value=coll_response)

    # -- create_collection / upsert
    client.create_collection = AsyncMock()
    client.upsert = AsyncMock()

    # -- query_points
    qr = MagicMock()
    qr.points = query_results or []
    client.query_points = AsyncMock(return_value=qr)

    return client


@pytest.fixture
def mock_embed():
    """Patch the embedding model so sentence-transformers is not needed."""
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array(_fake_vector(), dtype=np.float32)

    with patch("dytopo.memory.writer._get_embedding_model", return_value=fake_model):
        yield fake_model


# ---------------------------------------------------------------------------
# SwarmMemoryRecord construction
# ---------------------------------------------------------------------------

class TestSwarmMemoryRecord:
    """Verify that SwarmMemoryRecord can be built from typical swarm output."""

    def test_construction_basic(self):
        record = SwarmMemoryRecord(
            task_description="What is 2+2?",
            domain=SwarmDomain.MATH,
            agent_roles=["solver", "verifier"],
            round_count=2,
            key_findings=["The answer is 4"],
            final_answer_summary="4",
            convergence_achieved=True,
            total_tokens=500,
            wall_time_ms=1200,
        )

        assert record.task_description == "What is 2+2?"
        assert record.domain == SwarmDomain.MATH
        assert record.agent_roles == ["solver", "verifier"]
        assert record.round_count == 2
        assert record.convergence_achieved is True
        assert record.total_tokens == 500
        assert record.wall_time_ms == 1200
        assert record.metadata == {}
        assert record.created_at > 0

    def test_auto_generated_id(self):
        r1 = SwarmMemoryRecord(
            task_description="t1",
            domain=SwarmDomain.GENERAL,
            agent_roles=[],
            round_count=1,
            key_findings=[],
            final_answer_summary="",
            convergence_achieved=False,
            total_tokens=0,
            wall_time_ms=0,
        )
        r2 = SwarmMemoryRecord(
            task_description="t2",
            domain=SwarmDomain.GENERAL,
            agent_roles=[],
            round_count=1,
            key_findings=[],
            final_answer_summary="",
            convergence_achieved=False,
            total_tokens=0,
            wall_time_ms=0,
        )
        assert r1.id != r2.id, "Each record should get a unique UUID"

    def test_model_dump_produces_flat_dict(self):
        record = SwarmMemoryRecord(
            task_description="Sort a list",
            domain=SwarmDomain.CODE,
            agent_roles=["developer"],
            round_count=1,
            key_findings=["Used quicksort"],
            final_answer_summary="def sort(lst): ...",
            convergence_achieved=False,
            total_tokens=1000,
            wall_time_ms=3000,
            metadata={"note": "test"},
        )
        d = record.model_dump(mode="json")

        assert isinstance(d, dict)
        assert d["task_description"] == "Sort a list"
        assert d["domain"] == "code"
        assert d["agent_roles"] == ["developer"]
        assert d["key_findings"] == ["Used quicksort"]
        assert d["metadata"] == {"note": "test"}

    def test_all_domains_accepted(self):
        for domain in SwarmDomain:
            record = SwarmMemoryRecord(
                task_description="test",
                domain=domain,
                agent_roles=[],
                round_count=1,
                key_findings=[],
                final_answer_summary="",
                convergence_achieved=False,
                total_tokens=0,
                wall_time_ms=0,
            )
            assert record.domain == domain


# ---------------------------------------------------------------------------
# SwarmMemoryWriter.write() -- mocked Qdrant
# ---------------------------------------------------------------------------

class TestSwarmMemoryWrite:
    """Test write() with a mocked Qdrant client."""

    @pytest.mark.asyncio
    async def test_write_returns_point_id(self, mock_embed):
        mock_client = _make_mock_qdrant_client()

        with patch("dytopo.memory.writer.AsyncQdrantClient", return_value=mock_client):
            writer = SwarmMemoryWriter()
            point_id = await writer.write(**_sample_kwargs())

        assert isinstance(point_id, str)
        assert len(point_id) > 0, "Should return a non-empty point ID"
        # Validate it looks like a UUID
        uuid.UUID(point_id)

    @pytest.mark.asyncio
    async def test_write_calls_upsert(self, mock_embed):
        mock_client = _make_mock_qdrant_client()

        with patch("dytopo.memory.writer.AsyncQdrantClient", return_value=mock_client):
            writer = SwarmMemoryWriter()
            await writer.write(**_sample_kwargs())

        mock_client.upsert.assert_awaited_once()
        call_kwargs = mock_client.upsert.call_args
        assert call_kwargs.kwargs["collection_name"] == "swarm_memory"

        points = call_kwargs.kwargs["points"]
        assert len(points) == 1

        payload = points[0].payload
        assert payload["task_description"] == _sample_kwargs()["task_description"]
        assert payload["domain"] == "code"
        assert payload["convergence_achieved"] is True

    @pytest.mark.asyncio
    async def test_write_creates_collection_when_missing(self, mock_embed):
        mock_client = _make_mock_qdrant_client(collections=[])

        with patch("dytopo.memory.writer.AsyncQdrantClient", return_value=mock_client):
            writer = SwarmMemoryWriter()
            await writer.write(**_sample_kwargs())

        mock_client.get_collections.assert_awaited_once()
        mock_client.create_collection.assert_awaited_once()
        create_kwargs = mock_client.create_collection.call_args.kwargs
        assert create_kwargs["collection_name"] == "swarm_memory"

    @pytest.mark.asyncio
    async def test_write_skips_create_when_exists(self, mock_embed):
        mock_client = _make_mock_qdrant_client(collections=["swarm_memory"])

        with patch("dytopo.memory.writer.AsyncQdrantClient", return_value=mock_client):
            writer = SwarmMemoryWriter()
            await writer.write(**_sample_kwargs())

        mock_client.create_collection.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_write_embeds_task_and_findings(self, mock_embed):
        mock_client = _make_mock_qdrant_client()

        with patch("dytopo.memory.writer.AsyncQdrantClient", return_value=mock_client):
            writer = SwarmMemoryWriter()
            kw = _sample_kwargs()
            await writer.write(**kw)

        expected_text = f"{kw['task_description']} {' '.join(kw['key_findings'])}"
        mock_embed.encode.assert_called_once()
        actual_text = mock_embed.encode.call_args.args[0]
        assert actual_text == expected_text

    @pytest.mark.asyncio
    async def test_write_with_no_metadata(self, mock_embed):
        mock_client = _make_mock_qdrant_client()

        with patch("dytopo.memory.writer.AsyncQdrantClient", return_value=mock_client):
            writer = SwarmMemoryWriter()
            kw = _sample_kwargs()
            kw["metadata"] = None
            point_id = await writer.write(**kw)

        assert isinstance(point_id, str)
        assert len(point_id) > 0

        payload = mock_client.upsert.call_args.kwargs["points"][0].payload
        assert payload["metadata"] == {}


# ---------------------------------------------------------------------------
# SwarmMemoryWriter.query_similar() -- mocked Qdrant
# ---------------------------------------------------------------------------

class TestSwarmMemoryQuery:
    """Test query_similar() with mocked search results."""

    @staticmethod
    def _build_hits():
        """Build two fake scored-point objects."""
        hit1 = MagicMock()
        hit1.payload = {
            "id": "aaa-111",
            "task_description": "Solve TSP for 8 cities",
            "domain": "code",
            "agent_roles": ["developer", "tester"],
            "round_count": 2,
            "key_findings": ["Used branch and bound"],
            "final_answer_summary": "def tsp(): ...",
            "convergence_achieved": True,
            "total_tokens": 9000,
            "wall_time_ms": 5000,
        }
        hit1.score = 0.92

        hit2 = MagicMock()
        hit2.payload = {
            "id": "bbb-222",
            "task_description": "Shortest path in weighted graph",
            "domain": "code",
            "agent_roles": ["developer"],
            "round_count": 1,
            "key_findings": ["Dijkstra is optimal for non-negative weights"],
            "final_answer_summary": "def dijkstra(): ...",
            "convergence_achieved": False,
            "total_tokens": 4000,
            "wall_time_ms": 2000,
        }
        hit2.score = 0.78

        return [hit1, hit2]

    @pytest.mark.asyncio
    async def test_query_returns_payloads(self, mock_embed):
        hits = self._build_hits()
        mock_client = _make_mock_qdrant_client(
            collections=["swarm_memory"], query_results=hits,
        )

        with patch("dytopo.memory.writer.AsyncQdrantClient", return_value=mock_client):
            writer = SwarmMemoryWriter()
            results = await writer.query_similar("traveling salesman problem")

        assert len(results) == 2
        assert results[0]["task_description"] == "Solve TSP for 8 cities"
        assert results[1]["task_description"] == "Shortest path in weighted graph"

    @pytest.mark.asyncio
    async def test_query_respects_limit(self, mock_embed):
        hits = self._build_hits()
        mock_client = _make_mock_qdrant_client(
            collections=["swarm_memory"], query_results=hits,
        )

        with patch("dytopo.memory.writer.AsyncQdrantClient", return_value=mock_client):
            writer = SwarmMemoryWriter()
            await writer.query_similar("graph algorithm", limit=3)

        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs["limit"] == 3

    @pytest.mark.asyncio
    async def test_query_passes_vector(self, mock_embed):
        hits = self._build_hits()
        mock_client = _make_mock_qdrant_client(
            collections=["swarm_memory"], query_results=hits,
        )

        with patch("dytopo.memory.writer.AsyncQdrantClient", return_value=mock_client):
            writer = SwarmMemoryWriter()
            await writer.query_similar("optimize sorting")

        mock_embed.encode.assert_called_once()
        actual_text = mock_embed.encode.call_args.args[0]
        assert actual_text == "optimize sorting"

    @pytest.mark.asyncio
    async def test_query_empty_collection(self, mock_embed):
        """query_similar on a collection with no results returns empty list."""
        mock_client = _make_mock_qdrant_client(
            collections=["swarm_memory"], query_results=[],
        )

        with patch("dytopo.memory.writer.AsyncQdrantClient", return_value=mock_client):
            writer = SwarmMemoryWriter()
            results = await writer.query_similar("anything")

        assert results == []


# ---------------------------------------------------------------------------
# Graceful failure when Qdrant is down
# ---------------------------------------------------------------------------

class TestGracefulFailure:
    """Verify that Qdrant unavailability never crashes the caller."""

    @pytest.mark.asyncio
    async def test_write_returns_empty_on_qdrant_error(self, mock_embed):
        mock_client = _make_mock_qdrant_client()
        mock_client.get_collections = AsyncMock(
            side_effect=ConnectionError("Qdrant is down"),
        )
        mock_client.upsert = AsyncMock(
            side_effect=ConnectionError("Qdrant is down"),
        )

        with patch("dytopo.memory.writer.AsyncQdrantClient", return_value=mock_client):
            writer = SwarmMemoryWriter()
            result = await writer.write(**_sample_kwargs())

        assert result == "", "Should return empty string on failure"

    @pytest.mark.asyncio
    async def test_query_returns_empty_on_qdrant_error(self, mock_embed):
        mock_client = _make_mock_qdrant_client()
        mock_client.get_collections = AsyncMock(
            side_effect=ConnectionError("Qdrant is down"),
        )

        with patch("dytopo.memory.writer.AsyncQdrantClient", return_value=mock_client):
            writer = SwarmMemoryWriter()
            results = await writer.query_similar("test query")

        assert results == [], "Should return empty list on failure"

    @pytest.mark.asyncio
    async def test_write_survives_upsert_failure(self, mock_embed):
        mock_client = _make_mock_qdrant_client(collections=["swarm_memory"])
        mock_client.upsert = AsyncMock(
            side_effect=TimeoutError("Connection timed out"),
        )

        with patch("dytopo.memory.writer.AsyncQdrantClient", return_value=mock_client):
            writer = SwarmMemoryWriter()
            result = await writer.write(**_sample_kwargs())

        assert result == "", "Should return empty string when upsert fails"

    @pytest.mark.asyncio
    async def test_query_survives_search_failure(self, mock_embed):
        mock_client = _make_mock_qdrant_client(collections=["swarm_memory"])
        mock_client.query_points = AsyncMock(
            side_effect=RuntimeError("Unexpected Qdrant error"),
        )

        with patch("dytopo.memory.writer.AsyncQdrantClient", return_value=mock_client):
            writer = SwarmMemoryWriter()
            results = await writer.query_similar("test")

        assert results == [], "Should return empty list when search fails"

    @pytest.mark.asyncio
    async def test_ensure_collection_survives_error(self, mock_embed):
        mock_client = _make_mock_qdrant_client()
        mock_client.get_collections = AsyncMock(
            side_effect=OSError("Network unreachable"),
        )

        with patch("dytopo.memory.writer.AsyncQdrantClient", return_value=mock_client):
            writer = SwarmMemoryWriter()

            # Should not raise
            await writer.ensure_collection()

            assert writer._collection_ready is False, (
                "Should NOT mark collection as ready after failure"
            )
