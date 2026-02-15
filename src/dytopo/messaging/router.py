"""
Agent Message Protocol for DyTopo Swarms
=========================================

Typed message routing between agents based on graph edges,
with bounded history and context formatting.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import networkx as nx
from pydantic import BaseModel, Field

logger = logging.getLogger("dytopo.messaging")


class AgentMessage(BaseModel):
    """A typed message passed between agents via the routing graph."""

    from_agent: str
    to_agent: str
    content: str
    similarity: float
    round_number: int
    timestamp: float = Field(default_factory=time.time)


class MessageHistory:
    """Bounded per-agent message store with automatic pruning."""

    def __init__(self, max_per_agent: int = 10):
        self.max_per_agent = max_per_agent
        self._store: Dict[str, List[AgentMessage]] = {}

    def add(self, agent_id: str, message: AgentMessage) -> None:
        """Add a message to an agent's history, pruning if needed."""
        if agent_id not in self._store:
            self._store[agent_id] = []
        self._store[agent_id].append(message)
        if len(self._store[agent_id]) > self.max_per_agent:
            self._store[agent_id] = self._store[agent_id][-self.max_per_agent :]

    def get(
        self,
        agent_id: str,
        round_number: Optional[int] = None,
    ) -> List[AgentMessage]:
        """Get messages for an agent, optionally filtered by round."""
        messages = self._store.get(agent_id, [])
        if round_number is not None:
            messages = [m for m in messages if m.round_number == round_number]
        return messages

    def clear(self) -> None:
        """Clear all message history."""
        self._store.clear()


class MessageRouter:
    """Routes messages between agents according to graph structure.

    Ensures:
    - Messages only flow along edges
    - Messages sorted by similarity (highest first)
    - Bounded message history per agent
    - Provides context formatting for agent prompts
    - Legacy format adapter for existing orchestrator

    Usage:
        router = MessageRouter()
        routed = await router.route_round(graph, outputs, round_number=2)
        context = router.format_context("developer")
        legacy = router.to_legacy_format(routed["developer"])
    """

    def __init__(self, max_messages_per_agent: int = 10):
        self.history = MessageHistory(max_per_agent=max_messages_per_agent)

    async def route_round(
        self,
        graph: nx.DiGraph,
        outputs: Dict[str, str],
        round_number: int,
    ) -> Dict[str, List[AgentMessage]]:
        """Route messages based on graph edges for one round.

        For each edge src→tgt in the graph, creates an AgentMessage from
        src to tgt carrying src's output content.

        Args:
            graph: Directed graph from routing engine.
            outputs: Dict mapping agent_id to output text for this round.
            round_number: Current round number.

        Returns:
            Dict mapping each agent_id to its list of incoming messages,
            sorted by similarity descending.
        """
        routed: Dict[str, List[AgentMessage]] = {
            node: [] for node in graph.nodes()
        }

        now = time.time()

        for src in outputs:
            if not graph.has_node(src):
                continue
            for tgt in graph.successors(src):
                weight = graph[src][tgt].get("weight", 0.0)
                msg = AgentMessage(
                    from_agent=src,
                    to_agent=tgt,
                    content=outputs[src],
                    similarity=weight,
                    round_number=round_number,
                    timestamp=now,
                )
                routed[tgt].append(msg)
                self.history.add(tgt, msg)

        # Sort each agent's messages by similarity descending
        for agent_id in routed:
            routed[agent_id].sort(key=lambda m: m.similarity, reverse=True)

        return routed

    def format_context(
        self,
        agent_id: str,
        include_history: bool = True,
    ) -> str:
        """Build formatted message context for an agent's prompt.

        Args:
            agent_id: The agent to build context for.
            include_history: If True, include all history. If False, only latest round.

        Returns:
            Formatted string with all incoming messages.
        """
        messages = self.history.get(agent_id)
        if not messages:
            return "No incoming messages."

        if not include_history:
            current_round = max(m.round_number for m in messages)
            messages = [m for m in messages if m.round_number == current_round]

        lines = []
        for msg in messages:
            lines.append(
                f"--- From {msg.from_agent} "
                f"(similarity: {msg.similarity:.3f}, round: {msg.round_number}) ---"
            )
            lines.append(msg.content)
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def to_legacy_format(messages: List[AgentMessage]) -> List[Dict]:
        """Convert AgentMessages to legacy orchestrator format.

        Returns list of {"role": str, "sim": float, "content": str}
        compatible with _call_worker()'s incoming_messages parameter.
        """
        return [
            {
                "role": msg.from_agent,
                "sim": msg.similarity,
                "content": msg.content,
            }
            for msg in messages
        ]
