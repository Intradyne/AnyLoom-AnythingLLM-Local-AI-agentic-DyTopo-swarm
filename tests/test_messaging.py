"""Tests for DyTopo agent message protocol."""

import asyncio

import networkx as nx
import pytest

from dytopo.messaging import AgentMessage, MessageHistory, MessageRouter


# ── MessageHistory ───────────────────────────────────────────────────────────


class TestMessageHistory:
    def test_add_and_get(self):
        history = MessageHistory(max_per_agent=5)
        msg = AgentMessage(
            from_agent="a", to_agent="b",
            content="hello", similarity=0.8, round_number=1,
        )
        history.add("b", msg)
        assert len(history.get("b")) == 1
        assert history.get("b")[0].content == "hello"

    def test_prune_on_overflow(self):
        history = MessageHistory(max_per_agent=3)
        for i in range(5):
            msg = AgentMessage(
                from_agent="a", to_agent="b",
                content=f"msg_{i}", similarity=0.5, round_number=i,
            )
            history.add("b", msg)
        # Should keep only last 3
        msgs = history.get("b")
        assert len(msgs) == 3
        assert msgs[0].content == "msg_2"
        assert msgs[2].content == "msg_4"

    def test_filter_by_round(self):
        history = MessageHistory()
        for r in [1, 1, 2, 2, 3]:
            history.add("b", AgentMessage(
                from_agent="a", to_agent="b",
                content=f"round_{r}", similarity=0.5, round_number=r,
            ))
        assert len(history.get("b", round_number=2)) == 2
        assert len(history.get("b", round_number=3)) == 1
        assert len(history.get("b", round_number=99)) == 0

    def test_get_empty_agent(self):
        history = MessageHistory()
        assert history.get("nonexistent") == []

    def test_clear(self):
        history = MessageHistory()
        history.add("a", AgentMessage(
            from_agent="x", to_agent="a",
            content="test", similarity=0.5, round_number=1,
        ))
        history.clear()
        assert history.get("a") == []


# ── MessageRouter ────────────────────────────────────────────────────────────


def _make_graph(edges, nodes=None):
    """Helper: build a DiGraph from (src, tgt, weight) tuples."""
    G = nx.DiGraph()
    if nodes:
        G.add_nodes_from(nodes)
    for src, tgt, w in edges:
        G.add_edge(src, tgt, weight=w)
    return G


class TestMessageRouter:
    def test_no_edges_no_messages(self):
        """Graph with no edges should produce no messages."""
        G = _make_graph([], nodes=["a", "b", "c"])
        router = MessageRouter()
        loop = asyncio.new_event_loop()
        try:
            routed = loop.run_until_complete(
                router.route_round(G, {"a": "out_a", "b": "out_b", "c": "out_c"}, 1)
            )
            for agent_id in ["a", "b", "c"]:
                assert routed[agent_id] == []
        finally:
            loop.close()

    def test_linear_chain(self):
        """a→b→c: a sends to b, b sends to c."""
        G = _make_graph([("a", "b", 0.8), ("b", "c", 0.6)], nodes=["a", "b", "c"])
        router = MessageRouter()
        loop = asyncio.new_event_loop()
        try:
            routed = loop.run_until_complete(
                router.route_round(G, {"a": "out_a", "b": "out_b"}, 1)
            )
            # b receives from a
            assert len(routed["b"]) == 1
            assert routed["b"][0].from_agent == "a"
            assert routed["b"][0].content == "out_a"
            assert routed["b"][0].similarity == 0.8

            # c receives from b
            assert len(routed["c"]) == 1
            assert routed["c"][0].from_agent == "b"

            # a receives nothing
            assert routed["a"] == []
        finally:
            loop.close()

    def test_fork_broadcast(self):
        """a→b and a→c: a sends to both b and c."""
        G = _make_graph([("a", "b", 0.9), ("a", "c", 0.7)], nodes=["a", "b", "c"])
        router = MessageRouter()
        loop = asyncio.new_event_loop()
        try:
            routed = loop.run_until_complete(
                router.route_round(G, {"a": "out_a"}, 1)
            )
            assert len(routed["b"]) == 1
            assert len(routed["c"]) == 1
            assert routed["b"][0].from_agent == "a"
            assert routed["c"][0].from_agent == "a"
        finally:
            loop.close()

    def test_join_multiple_senders(self):
        """a→c and b→c: c receives from both a and b, sorted by similarity."""
        G = _make_graph(
            [("a", "c", 0.6), ("b", "c", 0.9)],
            nodes=["a", "b", "c"],
        )
        router = MessageRouter()
        loop = asyncio.new_event_loop()
        try:
            routed = loop.run_until_complete(
                router.route_round(G, {"a": "out_a", "b": "out_b"}, 1)
            )
            assert len(routed["c"]) == 2
            # Sorted by similarity descending
            assert routed["c"][0].from_agent == "b"  # 0.9
            assert routed["c"][1].from_agent == "a"  # 0.6
        finally:
            loop.close()

    def test_history_accumulates(self):
        """Messages across rounds accumulate in history."""
        G = _make_graph([("a", "b", 0.8)], nodes=["a", "b"])
        router = MessageRouter()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(router.route_round(G, {"a": "r1"}, 1))
            loop.run_until_complete(router.route_round(G, {"a": "r2"}, 2))
            all_msgs = router.history.get("b")
            assert len(all_msgs) == 2
            round1 = router.history.get("b", round_number=1)
            assert len(round1) == 1
            assert round1[0].content == "r1"
        finally:
            loop.close()

    def test_format_context_empty(self):
        router = MessageRouter()
        assert router.format_context("nonexistent") == "No incoming messages."

    def test_format_context_with_messages(self):
        G = _make_graph([("a", "b", 0.85)], nodes=["a", "b"])
        router = MessageRouter()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(router.route_round(G, {"a": "hello world"}, 1))
            ctx = router.format_context("b")
            assert "From a" in ctx
            assert "0.850" in ctx
            assert "hello world" in ctx
        finally:
            loop.close()

    def test_format_context_current_round_only(self):
        G = _make_graph([("a", "b", 0.8)], nodes=["a", "b"])
        router = MessageRouter()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(router.route_round(G, {"a": "r1"}, 1))
            loop.run_until_complete(router.route_round(G, {"a": "r2"}, 2))
            ctx = router.format_context("b", include_history=False)
            assert "r2" in ctx
            assert "r1" not in ctx
        finally:
            loop.close()

    def test_to_legacy_format(self):
        """Legacy format matches orchestrator's incoming_messages structure."""
        messages = [
            AgentMessage(
                from_agent="developer",
                to_agent="tester",
                content="code here",
                similarity=0.85,
                round_number=1,
            ),
        ]
        legacy = MessageRouter.to_legacy_format(messages)
        assert len(legacy) == 1
        assert legacy[0] == {
            "role": "developer",
            "sim": 0.85,
            "content": "code here",
        }

    def test_output_agent_not_in_graph(self):
        """Agent with output but not in graph should be skipped gracefully."""
        G = _make_graph([("a", "b", 0.8)], nodes=["a", "b"])
        router = MessageRouter()
        loop = asyncio.new_event_loop()
        try:
            # "c" has output but is not in graph
            routed = loop.run_until_complete(
                router.route_round(G, {"a": "out_a", "c": "out_c"}, 1)
            )
            assert len(routed["b"]) == 1
            assert "c" not in routed  # c is not a graph node
        finally:
            loop.close()
