"""Tests for DyTopo async routing engine.

NOTE: Tests that call build_routing_graph() require sentence-transformers
to be installed (loads MiniLM-L6-v2, ~80 MB). Tier computation tests
are pure graph logic and require only networkx.
"""

import asyncio

import networkx as nx
import numpy as np
import pytest

from dytopo.routing import AsyncRoutingEngine, RoutingResult


# ── Tier Computation (no embedding needed) ───────────────────────────────────


class TestGetExecutionTiers:
    def test_empty_graph(self):
        G = nx.DiGraph()
        tiers = AsyncRoutingEngine.get_execution_tiers(G, [])
        assert tiers == []

    def test_single_node(self):
        G = nx.DiGraph()
        G.add_node("a")
        tiers = AsyncRoutingEngine.get_execution_tiers(G, ["a"])
        assert tiers == [["a"]]

    def test_no_edges_all_tier_zero(self):
        """Disconnected agents are all in tier 0 (can run in parallel)."""
        G = nx.DiGraph()
        G.add_nodes_from(["a", "b", "c"])
        tiers = AsyncRoutingEngine.get_execution_tiers(G, ["a", "b", "c"])
        assert len(tiers) == 1
        assert set(tiers[0]) == {"a", "b", "c"}

    def test_linear_chain(self):
        """a→b→c produces 3 tiers."""
        G = nx.DiGraph()
        G.add_edge("a", "b")
        G.add_edge("b", "c")
        tiers = AsyncRoutingEngine.get_execution_tiers(G, ["a", "b", "c"])
        assert len(tiers) == 3
        assert tiers[0] == ["a"]
        assert tiers[1] == ["b"]
        assert tiers[2] == ["c"]

    def test_fork_same_tier(self):
        """a→b and a→c: b and c are in same tier."""
        G = nx.DiGraph()
        G.add_edge("a", "b")
        G.add_edge("a", "c")
        tiers = AsyncRoutingEngine.get_execution_tiers(G, ["a", "b", "c"])
        assert len(tiers) == 2
        assert tiers[0] == ["a"]
        assert set(tiers[1]) == {"b", "c"}

    def test_diamond_dag(self):
        """a→b, a→c, b→d, c→d: d must wait for both b and c."""
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
        tiers = AsyncRoutingEngine.get_execution_tiers(G, ["a", "b", "c", "d"])
        assert len(tiers) == 3
        assert tiers[0] == ["a"]
        assert set(tiers[1]) == {"b", "c"}
        assert tiers[2] == ["d"]

    def test_mixed_independent_and_dependent(self):
        """a→c, b independent: a and b in tier 0, c in tier 1."""
        G = nx.DiGraph()
        G.add_nodes_from(["a", "b", "c"])
        G.add_edge("a", "c")
        tiers = AsyncRoutingEngine.get_execution_tiers(G, ["a", "b", "c"])
        assert len(tiers) == 2
        assert set(tiers[0]) == {"a", "b"}
        assert tiers[1] == ["c"]


# ── Full Routing Pipeline (requires sentence-transformers) ───────────────────


def _has_sentence_transformers():
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _has_sentence_transformers(),
    reason="sentence-transformers not installed",
)
class TestAsyncRoutingEngine:
    def _run(self, coro):
        """Helper to run async code in a fresh event loop."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_empty_descriptors(self):
        engine = AsyncRoutingEngine(tau=0.3, K_in=3)
        result = self._run(engine.build_routing_graph([], {}))
        assert isinstance(result, RoutingResult)
        assert result.execution_order == []
        assert result.tiers == []
        assert result.edges == []

    def test_single_agent(self):
        engine = AsyncRoutingEngine(tau=0.3, K_in=3)
        descriptors = {"agent_a": {"key": "I provide analysis", "query": "I need data"}}
        result = self._run(engine.build_routing_graph(["agent_a"], descriptors))
        assert result.execution_order == ["agent_a"]
        assert result.tiers == [["agent_a"]]
        assert result.edges == []
        assert result.isolated == ["agent_a"]

    def test_two_agents_routing(self):
        engine = AsyncRoutingEngine(tau=0.1, K_in=3)  # Low tau to ensure edges
        descriptors = {
            "coder": {"key": "I write Python code", "query": "I need test results"},
            "tester": {"key": "I provide test results", "query": "I need Python code to test"},
        }
        result = self._run(engine.build_routing_graph(["coder", "tester"], descriptors))
        assert isinstance(result.graph, nx.DiGraph)
        assert len(result.execution_order) == 2
        assert result.similarity_matrix.shape == (2, 2)
        # Self-loops should be zero
        assert result.similarity_matrix[0, 0] == 0.0
        assert result.similarity_matrix[1, 1] == 0.0

    def test_four_agent_swarm(self):
        """Simulate a realistic 4-agent swarm."""
        engine = AsyncRoutingEngine(tau=0.2, K_in=3)
        descriptors = {
            "analyst": {
                "key": "I provide deep analysis of the problem",
                "query": "I need critical feedback on my analysis",
            },
            "critic": {
                "key": "I provide critical evaluation and identify weaknesses",
                "query": "I need analysis and proposed solutions to evaluate",
            },
            "synthesizer": {
                "key": "I synthesize insights into a coherent final answer",
                "query": "I need analysis and criticism to integrate",
            },
            "researcher": {
                "key": "I research background information and context",
                "query": "I need to know what information gaps exist",
            },
        }
        agent_ids = ["analyst", "critic", "synthesizer", "researcher"]
        result = self._run(engine.build_routing_graph(agent_ids, descriptors))

        # Basic structure checks
        assert len(result.execution_order) == 4
        assert set(result.execution_order) == set(agent_ids)
        assert result.similarity_matrix.shape == (4, 4)
        assert len(result.tiers) >= 1
        assert result.stats["edge_count"] >= 0
        # Graph should be a DAG
        assert nx.is_directed_acyclic_graph(result.graph)

    def test_graph_is_dag(self):
        """Even with high similarity, result should always be a DAG."""
        engine = AsyncRoutingEngine(tau=0.05, K_in=3)  # Very low tau = many edges
        descriptors = {
            "a": {"key": "I do task A", "query": "I need help with task A"},
            "b": {"key": "I do task B", "query": "I need help with task B"},
            "c": {"key": "I do task C", "query": "I need help with task C"},
        }
        result = self._run(engine.build_routing_graph(["a", "b", "c"], descriptors))
        assert nx.is_directed_acyclic_graph(result.graph)

    def test_routing_result_completeness(self):
        """All RoutingResult fields should be populated."""
        engine = AsyncRoutingEngine(tau=0.2, K_in=3)
        descriptors = {
            "a": {"key": "code writing", "query": "test results needed"},
            "b": {"key": "test execution", "query": "code to test needed"},
        }
        result = self._run(engine.build_routing_graph(["a", "b"], descriptors))

        assert isinstance(result.graph, nx.DiGraph)
        assert isinstance(result.execution_order, list)
        assert isinstance(result.tiers, list)
        assert isinstance(result.edges, list)
        assert isinstance(result.isolated, list)
        assert isinstance(result.removed_edges, list)
        assert isinstance(result.stats, dict)
        assert isinstance(result.similarity_matrix, np.ndarray)
        assert "edge_count" in result.stats
        assert "density" in result.stats
        assert "tier_count" in result.stats

    def test_max_indegree_enforced(self):
        """K_in=1 should limit each agent to at most 1 incoming edge."""
        engine = AsyncRoutingEngine(tau=0.05, K_in=1)  # Very low tau, strict K_in
        descriptors = {
            "a": {"key": "analysis and planning", "query": "need feedback"},
            "b": {"key": "criticism and review", "query": "need analysis"},
            "c": {"key": "synthesis and summary", "query": "need analysis and criticism"},
        }
        result = self._run(engine.build_routing_graph(["a", "b", "c"], descriptors))
        # Each node should have at most 1 incoming edge
        for node in result.graph.nodes():
            assert result.graph.in_degree(node) <= 1
