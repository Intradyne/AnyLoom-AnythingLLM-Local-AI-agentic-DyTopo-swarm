"""End-to-end routing efficiency tests.

Proves DyTopo's sparse routing reduces redundant messages vs broadcast.
No LLM needed — uses synthetic descriptors and real routing math.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np

# Skip if sentence-transformers not installed
try:
    from dytopo.router import build_routing_result
    HAS_ST = True
except ImportError:
    HAS_ST = False

pytestmark = pytest.mark.skipif(not HAS_ST, reason="sentence-transformers required")


# Diverse descriptors that should produce sparse (not fully-connected) graph
DIVERSE_DESCRIPTORS = {
    "developer": {"key": "Python implementation of REST API endpoints", "query": "What database schema should I use?"},
    "tester": {"key": "Unit test coverage and edge case validation", "query": "What functions need testing?"},
    "researcher": {"key": "Security best practices and OWASP guidelines", "query": "What authentication patterns are recommended?"},
    "designer": {"key": "System architecture and component interfaces", "query": "How should modules communicate?"},
    "manager": {"key": "Project timeline and task decomposition", "query": "What are the dependencies between tasks?"},
}

# Similar descriptors that should produce dense (nearly fully-connected) graph
SIMILAR_DESCRIPTORS = {
    f"agent_{i}": {"key": "Python code implementation", "query": "How to write Python code?"}
    for i in range(5)
}


class TestRoutingEfficiency:
    def test_sparse_routing_reduces_messages(self):
        """Routed messages should be fewer than broadcast (N*(N-1))."""
        agent_ids = list(DIVERSE_DESCRIPTORS.keys())
        n = len(agent_ids)
        broadcast_messages = n * (n - 1)  # every agent sends to every other

        # Build routing result with default tau=0.3
        result = build_routing_result(agent_ids, DIVERSE_DESCRIPTORS, tau=0.3, K_in=3)
        routed_messages = len(result["edges"])

        assert routed_messages < broadcast_messages, (
            f"Routing did not reduce messages: {routed_messages} >= {broadcast_messages}"
        )

    def test_no_isolated_agents_with_diverse_descriptors(self):
        """All agents should participate in at least one edge (send or receive)."""
        agent_ids = list(DIVERSE_DESCRIPTORS.keys())
        result = build_routing_result(agent_ids, DIVERSE_DESCRIPTORS, tau=0.2, K_in=3)
        agents_in_edges = set()
        for edge in result["edges"]:
            agents_in_edges.add(edge[0])
            agents_in_edges.add(edge[1])

        all_agents = set(DIVERSE_DESCRIPTORS.keys())
        isolated = all_agents - agents_in_edges
        # With tau=0.2 (permissive), most agents should connect
        assert len(isolated) <= 1, f"Too many isolated agents: {isolated}"

    def test_tier_ordering_is_topological(self):
        """Agents in earlier tiers should not depend on agents in later tiers."""
        agent_ids = list(DIVERSE_DESCRIPTORS.keys())
        result = build_routing_result(agent_ids, DIVERSE_DESCRIPTORS, tau=0.3, K_in=3)

        # Build graph and get tiers
        from dytopo.graph import build_execution_graph, get_execution_tiers
        graph = build_execution_graph(result["edges"], agent_ids)
        tiers = get_execution_tiers(graph, agent_ids)

        # Build tier index: agent -> tier_number
        tier_index = {}
        for tier_num, agents in enumerate(tiers):
            for agent in agents:
                tier_index[agent] = tier_num

        # For each edge (A -> B), A's tier should be <= B's tier
        for edge in result["edges"]:
            src, dst = edge[0], edge[1]
            if src in tier_index and dst in tier_index:
                assert tier_index[src] <= tier_index[dst], (
                    f"Topological violation: {src} (tier {tier_index[src]}) -> {dst} (tier {tier_index[dst]})"
                )

    def test_routing_determinism_across_runs(self):
        """Same descriptors should produce same routing graph."""
        agent_ids = list(DIVERSE_DESCRIPTORS.keys())
        r1 = build_routing_result(agent_ids, DIVERSE_DESCRIPTORS, tau=0.3, K_in=3)
        r2 = build_routing_result(agent_ids, DIVERSE_DESCRIPTORS, tau=0.3, K_in=3)
        assert r1["edges"] == r2["edges"], "Routing is not deterministic"

    def test_high_tau_produces_sparser_graph(self):
        """Higher threshold should produce fewer edges."""
        agent_ids = list(DIVERSE_DESCRIPTORS.keys())
        r_low = build_routing_result(agent_ids, DIVERSE_DESCRIPTORS, tau=0.1, K_in=5)
        r_high = build_routing_result(agent_ids, DIVERSE_DESCRIPTORS, tau=0.6, K_in=5)
        assert len(r_high["edges"]) <= len(r_low["edges"]), (
            f"Higher tau should produce fewer edges: {len(r_high['edges'])} > {len(r_low['edges'])}"
        )

    def test_low_tau_approaches_broadcast(self):
        """Very low threshold with similar descriptors should produce near-complete graph."""
        agent_ids = list(SIMILAR_DESCRIPTORS.keys())
        n = len(agent_ids)
        max_edges = n * (n - 1)  # directed complete graph

        result = build_routing_result(agent_ids, SIMILAR_DESCRIPTORS, tau=0.01, K_in=n)
        actual_edges = len(result["edges"])

        # Should be at least 50% of complete graph with very similar descriptors
        assert actual_edges >= max_edges * 0.5, (
            f"Low tau with similar descriptors should approach broadcast: {actual_edges}/{max_edges}"
        )
