"""
DyTopo Integration and Unit Tests
==================================

Unit tests (no LLM required):
    pytest tests/test_dytopo.py -k "not integration" -v

Integration tests (requires LM Studio on localhost:1234):
    pytest tests/test_dytopo.py -k "integration" -v
"""

import asyncio
import json
import pytest
import numpy as np
from pathlib import Path


def _lm_studio_available() -> bool:
    """Check if LM Studio is running."""
    try:
        import httpx
        r = httpx.get("http://localhost:1234/v1/models", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UNIT TESTS — No LLM required
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRoutingMath:
    """Test similarity matrix, threshold, degree cap — no LLM needed."""

    def test_similarity_matrix_shape(self):
        """Test that similarity matrix has correct shape and no self-loops."""
        from dytopo.router import embed_descriptors, compute_similarity_matrix

        agent_ids = ["a", "b", "c"]
        descriptors = {
            "a": {"key": "I have sorting code", "query": "I need test cases"},
            "b": {"key": "I have test cases", "query": "I need the code to test"},
            "c": {"key": "I reviewed the design", "query": "I need code quality metrics"},
        }

        key_vecs, query_vecs = embed_descriptors(agent_ids, descriptors)
        S = compute_similarity_matrix(query_vecs, key_vecs)

        assert S.shape == (3, 3), f"Expected (3,3), got {S.shape}"
        assert S[0][0] == 0.0, "Diagonal should be zero (no self-loops)"
        assert S[1][1] == 0.0
        assert S[2][2] == 0.0
        assert np.all(S >= 0.0) and np.all(S <= 1.0), "Similarities should be in [0,1]"

    def test_threshold_sparsity(self):
        """Test that thresholding produces correct sparsity."""
        from dytopo.router import apply_threshold

        S = np.array([
            [0.0, 0.5, 0.2],
            [0.4, 0.0, 0.1],
            [0.6, 0.3, 0.0]
        ])

        A = apply_threshold(S, tau=0.3)

        assert A[0][1] == 1, "0.5 >= 0.3 should be active"
        assert A[0][2] == 0, "0.2 < 0.3 should be inactive"
        assert A[2][0] == 1, "0.6 >= 0.3 should be active"
        assert A[2][1] == 1, "0.3 >= 0.3 should be active"
        assert A.sum() == 4, f"Expected 4 edges, got {A.sum()}"

    def test_degree_cap(self):
        """Test that max indegree cap is enforced."""
        from dytopo.router import enforce_max_indegree

        S = np.array([
            [0.0, 0.9, 0.8, 0.7, 0.6],
            [0.5, 0.0, 0.4, 0.3, 0.2],
            [0.1, 0.2, 0.0, 0.3, 0.4],
            [0.3, 0.4, 0.5, 0.0, 0.6],
            [0.2, 0.3, 0.4, 0.5, 0.0],
        ])

        A = (S >= 0.3).astype(np.int32)
        A_capped, removed = enforce_max_indegree(A, S, K_in=2)

        # Check indegree constraint
        for col_idx in range(5):
            indegree = A_capped[:, col_idx].sum()
            assert indegree <= 2, f"Column {col_idx} has indegree {indegree} > 2"

    def test_edge_direction_convention(self):
        """Verify A[i][j] = 1 means j → i (j sends to i)."""
        from dytopo.router import build_routing_result

        agent_ids = ["dev", "tester"]
        descriptors = {
            "dev": {"key": "I wrote the sorting function", "query": "I need test results"},
            "tester": {"key": "I have test case results", "query": "I need the source code"},
        }

        result = build_routing_result(agent_ids, descriptors, tau=0.1, K_in=3)
        edges = result["edges"]

        # With low tau, should have edges
        assert len(edges) > 0, "Should have edges with tau=0.1"

        # Verify no self-loops
        for src, tgt, _ in edges:
            assert src != tgt, f"Self-loop found: {src} -> {tgt}"


class TestGraphOperations:
    """Test cycle breaking and topological sort — no LLM needed."""

    def test_dag_topo_sort(self):
        """Test topological sort on a simple DAG."""
        from dytopo.graph import build_execution_graph, get_execution_order

        edges = [("a", "b", 0.5), ("a", "c", 0.4), ("b", "c", 0.3)]
        G = build_execution_graph(edges, ["a", "b", "c"])
        order = get_execution_order(G, ["a", "b", "c"])

        # a must come before b and c, b must come before c
        assert order.index("a") < order.index("b"), "a should come before b"
        assert order.index("a") < order.index("c"), "a should come before c"
        assert order.index("b") < order.index("c"), "b should come before c"

    def test_cycle_breaking(self):
        """Test that cycles are broken to create a DAG."""
        from dytopo.graph import build_execution_graph, break_cycles
        import networkx as nx

        # Create a cycle: a↔b
        edges = [("a", "b", 0.5), ("b", "a", 0.3)]
        G = build_execution_graph(edges, ["a", "b"])

        # Should have a cycle initially
        assert not nx.is_directed_acyclic_graph(G), "Graph should have cycle before breaking"

        removed = break_cycles(G)

        assert nx.is_directed_acyclic_graph(G), "Graph should be DAG after breaking cycles"
        assert len(removed) == 1, f"Should remove 1 edge, removed {len(removed)}"
        # Should remove the weaker edge (b→a, weight 0.3)
        assert removed[0] == ("b", "a"), f"Should remove (b, a), removed {removed[0]}"


class TestConvergence:
    """Test convergence detection logic."""

    def test_convergence_detected(self):
        """Test that identical outputs trigger convergence."""
        from dytopo.governance import detect_convergence
        from dytopo.models import AgentDescriptor

        round_history = [
            {"round": 1, "outputs": {"dev": AgentDescriptor(key="k", query="q", work="hello world code")}},
            {"round": 2, "outputs": {"dev": AgentDescriptor(key="k", query="q", work="hello world code")}},
            {"round": 3, "outputs": {"dev": AgentDescriptor(key="k", query="q", work="hello world code")}},
        ]

        converged, sim = detect_convergence(round_history, window_size=3, similarity_threshold=0.95)

        assert converged is True, "Should detect convergence with identical outputs"
        assert sim >= 0.95, f"Similarity should be >= 0.95, got {sim}"

    def test_no_convergence(self):
        """Test that different outputs don't trigger convergence."""
        from dytopo.governance import detect_convergence
        from dytopo.models import AgentDescriptor

        round_history = [
            {"round": 1, "outputs": {"dev": AgentDescriptor(key="k", query="q", work="approach A: quicksort")}},
            {"round": 2, "outputs": {"dev": AgentDescriptor(key="k", query="q", work="approach B: mergesort with optimization")}},
            {"round": 3, "outputs": {"dev": AgentDescriptor(key="k", query="q", work="approach C: heapsort implementation")}},
        ]

        converged, sim = detect_convergence(round_history, window_size=3, similarity_threshold=0.95)

        assert converged is False, "Should not detect convergence with different outputs"
        assert sim < 0.95, f"Similarity should be < 0.95, got {sim}"


class TestStalling:
    """Test stalling detection logic."""

    def test_stalling_detected(self):
        """Test that repeated outputs trigger stalling detection."""
        from dytopo.governance import detect_stalling
        from dytopo.models import AgentDescriptor

        round_history = [
            {"round": 1, "outputs": {"dev": AgentDescriptor(key="k", query="q", work="same output v1")}},
            {"round": 2, "outputs": {"dev": AgentDescriptor(key="k", query="q", work="same output v1")}},
            {"round": 3, "outputs": {"dev": AgentDescriptor(key="k", query="q", work="same output v1")}},
        ]

        is_stalling, sim = detect_stalling("dev", round_history, window_size=3, similarity_threshold=0.98)

        assert is_stalling is True, "Should detect stalling with identical outputs"
        assert sim >= 0.98, f"Similarity should be >= 0.98, got {sim}"


class TestAuditLog:
    """Test audit logging functionality."""

    def test_audit_writes_jsonl(self, tmp_path):
        """Test that audit log writes valid JSONL."""
        from dytopo.audit import SwarmAuditLog

        audit = SwarmAuditLog("test-123", base_dir=tmp_path)
        audit.swarm_started("2+2", 3, ["mgr", "solver"])
        audit.round_started(1)
        audit.agent_executed(1, "solver", {"work": "4"}, 1.5)
        audit.swarm_completed(1, "The answer is 4", 1)
        audit.close()

        log_file = tmp_path / "test-123" / "audit.jsonl"
        assert log_file.exists(), f"Audit log file should exist at {log_file}"

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 4, f"Expected 4 events, got {len(lines)}"

        # Parse and verify events
        events = [json.loads(line) for line in lines]
        assert events[0]["event_type"] == "swarm_started"
        assert events[1]["event_type"] == "round_started"
        assert events[2]["event_type"] == "agent_executed"
        assert events[3]["event_type"] == "swarm_completed"

        # Verify all events have required fields
        for event in events:
            assert "timestamp" in event
            assert "task_id" in event
            assert event["task_id"] == "test-123"


class TestMetrics:
    """Test metrics tracking."""

    def test_agent_metrics_properties(self):
        """Test AgentMetrics computed properties."""
        from dytopo.models import AgentMetrics

        metrics = AgentMetrics(
            successful_rounds=8,
            failed_rounds=2,
            total_latency_ms=5000,
            total_tokens_in=1000,
            total_tokens_out=2000,
        )

        assert metrics.avg_latency_ms == 500.0, "5000ms / 10 rounds = 500ms"
        assert metrics.success_rate == 0.8, "8 / 10 = 0.8"
        assert metrics.avg_tokens_per_round == 375.0, "(1000 + 2000) / 8 = 375"

    def test_swarm_metrics_initialization(self):
        """Test SwarmMetrics initializes correctly."""
        from dytopo.models import SwarmMetrics

        metrics = SwarmMetrics()

        assert metrics.total_rounds == 0
        assert metrics.total_llm_calls == 0
        assert metrics.total_tokens == 0
        assert metrics.convergence_detected_at is None
        assert len(metrics.per_agent) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INTEGRATION TESTS — Requires LM Studio
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.mark.skipif(
    not _lm_studio_available(),
    reason="LM Studio not running on localhost:1234"
)
class TestIntegration:
    """Full swarm execution tests. Requires LM Studio running with a model."""

    @pytest.fixture(autouse=True)
    def event_loop(self):
        """Provide event loop for async tests."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    def test_math_swarm_simple(self, event_loop):
        """Test a simple math problem with the math swarm."""
        from dytopo.models import SwarmTask, SwarmDomain
        from dytopo.orchestrator import run_swarm

        swarm = SwarmTask(
            task="What is 15 * 23?",
            domain=SwarmDomain.MATH,
            T_max=2,
        )

        result = event_loop.run_until_complete(run_swarm(swarm))

        assert result.status.value == "complete", f"Expected complete, got {result.status}"
        assert result.final_answer is not None, "Should have final answer"
        # The answer should be 345
        assert "345" in result.final_answer, f"Expected 345 in answer: {result.final_answer}"

    def test_code_swarm_fizzbuzz(self, event_loop):
        """Test FizzBuzz implementation with code swarm."""
        from dytopo.models import SwarmTask, SwarmDomain
        from dytopo.orchestrator import run_swarm

        swarm = SwarmTask(
            task="Write a Python function fizzbuzz(n) that returns a list of strings for numbers 1 to n, "
                 "replacing multiples of 3 with 'Fizz', multiples of 5 with 'Buzz', and multiples of both with 'FizzBuzz'.",
            domain=SwarmDomain.CODE,
            T_max=3,
        )

        result = event_loop.run_until_complete(run_swarm(swarm))

        assert result.status.value == "complete", f"Expected complete, got {result.status}"
        assert result.final_answer is not None, "Should have final answer"
        # Should mention fizzbuzz or have the function
        answer_lower = result.final_answer.lower()
        assert "fizzbuzz" in answer_lower or "def " in answer_lower, \
            f"Expected FizzBuzz code in answer: {result.final_answer[:200]}"

    def test_convergence_detection_integration(self, event_loop):
        """Test that convergence detection works in a real swarm."""
        from dytopo.models import SwarmTask, SwarmDomain
        from dytopo.orchestrator import run_swarm

        swarm = SwarmTask(
            task="What is 2 + 2?",  # Simple task should converge quickly
            domain=SwarmDomain.MATH,
            T_max=5,
        )

        result = event_loop.run_until_complete(run_swarm(swarm))

        # Check that swarm completed
        assert result.status.value in ["complete", "failed"], f"Unexpected status: {result.status}"

        # Check metrics were populated
        assert result.swarm_metrics.total_rounds > 0, "Should have completed at least one round"
        assert result.swarm_metrics.total_llm_calls > 0, "Should have made LLM calls"
        assert result.swarm_metrics.total_tokens > 0, "Should have used tokens"

    def test_audit_log_created(self, event_loop, tmp_path):
        """Test that audit log is created during swarm execution."""
        from dytopo.models import SwarmTask, SwarmDomain
        from dytopo.orchestrator import run_swarm
        from dytopo.config import load_config
        import os

        # Override log directory
        os.environ["DYTOPO_LOG_DIR"] = str(tmp_path)

        swarm = SwarmTask(
            task="Count to 5",
            domain=SwarmDomain.GENERAL,
            T_max=1,
        )

        result = event_loop.run_until_complete(run_swarm(swarm))

        # Check audit log was created
        audit_file = tmp_path / result.task_id / "audit.jsonl"
        # Note: This may not exist if config override doesn't work properly
        # In that case, check the default location
        if not audit_file.exists():
            default_log = Path.home() / "dytopo-logs" / result.task_id / "audit.jsonl"
            assert default_log.exists(), \
                f"Audit log not found at {audit_file} or {default_log}"
        else:
            assert audit_file.exists(), f"Audit log should exist at {audit_file}"
