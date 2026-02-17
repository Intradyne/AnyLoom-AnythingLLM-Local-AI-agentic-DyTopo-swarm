"""
Async Routing Engine for DyTopo Swarms
=======================================

Wraps the synchronous router.py and graph.py functions with an async
interface, adding lock-based embedding serialization and tier computation
for parallel-within-tier agent execution.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

from dytopo.router import (
    apply_threshold,
    compute_similarity_matrix,
    embed_descriptors,
    enforce_max_indegree,
)
from dytopo.graph import (
    build_execution_graph,
    get_execution_order,
)

logger = logging.getLogger("dytopo.routing.async_engine")


async def export_routing_trace(
    task_id: str,
    round_num: int,
    routing_result: RoutingResult,
    log_dir: str = "logs/dytopo",
) -> Path:
    """Write structured routing trace to JSON for debugging and visualization.

    Output: logs/dytopo/{YYYY-MM-DD}/{task_id}/round_{NN}.json

    Args:
        task_id: Unique task identifier
        round_num: Round number (1-indexed)
        routing_result: RoutingResult from build_routing_graph
        log_dir: Base directory for logs

    Returns:
        Path to written trace file
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    trace_dir = Path(log_dir) / date_str / task_id
    trace_dir.mkdir(parents=True, exist_ok=True)

    trace = {
        "task_id": task_id,
        "round": round_num,
        "timestamp": datetime.now().isoformat(),
        "agents": [str(a) for a in routing_result.execution_order],
        "edges": [
            {"from": str(e[0]), "to": str(e[1]), "weight": round(e[2], 4) if len(e) > 2 else None}
            for e in routing_result.edges
        ],
        "tiers": [[str(a) for a in tier] for tier in routing_result.tiers],
        "isolated": [str(a) for a in routing_result.isolated],
        "stats": routing_result.stats,
    }

    out_path = trace_dir / f"round_{round_num:02d}.json"
    out_path.write_text(json.dumps(trace, indent=2))
    logger.debug(f"Routing trace written: {out_path}")
    return out_path


@dataclass
class RoutingResult:
    """Result of async routing graph construction.

    Attributes:
        graph: NetworkX DiGraph with agents as nodes.
        execution_order: Topological agent ordering.
        tiers: Groups of agents that can execute in parallel.
        edges: List of (source, target, weight) tuples.
        isolated: Agent IDs with no incoming edges.
        removed_edges: Edges removed during cycle breaking.
        stats: Routing statistics dict.
        similarity_matrix: NxN cosine similarity matrix.
    """

    graph: nx.DiGraph
    execution_order: List[str]
    tiers: List[List[str]]
    edges: List[Tuple[str, str, float]]
    isolated: List[str]
    removed_edges: List[Tuple[str, str]]
    stats: Dict[str, Any]
    similarity_matrix: np.ndarray


class AsyncRoutingEngine:
    """Async wrapper around DyTopo's synchronous routing pipeline.

    Thread Safety:
    - sentence-transformers (MiniLM) is NOT thread-safe.
      All embedding calls are serialized via asyncio.Lock.
    - numpy similarity computation is reentrant (read-only on model).
    - networkx graph construction is reentrant.

    Usage:
        engine = AsyncRoutingEngine(tau=0.5, K_in=3)
        result = await engine.build_routing_graph(agent_ids, descriptors)
        # result.tiers tells you which agents can run concurrently
    """

    def __init__(self, tau: float = 0.5, K_in: int = 3):
        self.tau = tau
        self.K_in = K_in
        self._embed_lock = asyncio.Lock()

    async def build_routing_graph(
        self,
        agent_ids: List[str],
        descriptors: Dict[str, dict],
    ) -> RoutingResult:
        """Build routing graph from agent descriptors.

        Steps:
        1. Embed descriptors (serialized via lock, offloaded to thread)
        2. Compute similarity matrix (pure numpy, fast)
        3. Apply threshold and enforce max indegree
        4. Build execution graph and determine order + tiers

        Args:
            agent_ids: Ordered list of agent IDs.
            descriptors: Dict mapping agent_id to {"key": str, "query": str}.

        Returns:
            RoutingResult with graph, execution order, tiers, and statistics.
        """
        if not agent_ids:
            return RoutingResult(
                graph=nx.DiGraph(),
                execution_order=[],
                tiers=[],
                edges=[],
                isolated=[],
                removed_edges=[],
                stats={"edge_count": 0, "density": 0.0},
                similarity_matrix=np.array([]),
            )

        if len(agent_ids) == 1:
            G = nx.DiGraph()
            G.add_node(agent_ids[0])
            return RoutingResult(
                graph=G,
                execution_order=agent_ids,
                tiers=[agent_ids],
                edges=[],
                isolated=agent_ids,
                removed_edges=[],
                stats={"edge_count": 0, "density": 0.0, "isolated_count": 1},
                similarity_matrix=np.zeros((1, 1), dtype=np.float32),
            )

        # Step 1: Embed (serialized — sentence-transformers not thread-safe)
        async with self._embed_lock:
            key_vecs, query_vecs = await asyncio.to_thread(
                embed_descriptors, agent_ids, descriptors
            )

        # Step 2: Similarity matrix (numpy, CPU-bound but fast)
        S = compute_similarity_matrix(query_vecs, key_vecs)

        # Step 3: Threshold + degree cap
        A = apply_threshold(S, self.tau)
        A, removed_idx = enforce_max_indegree(A, S, self.K_in)

        # Step 4: Build edges list
        N = len(agent_ids)
        edges: List[Tuple[str, str, float]] = []
        for i in range(N):
            for j in range(N):
                if A[i][j] == 1:
                    # A[i][j] == 1 means j→i (j sends to i)
                    edges.append((agent_ids[j], agent_ids[i], float(S[i][j])))

        isolated = [agent_ids[i] for i in range(N) if A[i].sum() == 0]
        removed_named = [(agent_ids[i], agent_ids[j]) for i, j in removed_idx]

        # Step 5: Build graph and execution order
        G = build_execution_graph(edges, agent_ids)
        execution_order = get_execution_order(G, agent_ids)

        # Step 6: Compute execution tiers
        tiers = self.get_execution_tiers(G, execution_order)

        # Statistics
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
            "tier_count": len(tiers),
        }

        return RoutingResult(
            graph=G,
            execution_order=execution_order,
            tiers=tiers,
            edges=edges,
            isolated=isolated,
            removed_edges=removed_named,
            stats=stats,
            similarity_matrix=S,
        )

    @staticmethod
    def get_execution_tiers(
        graph: nx.DiGraph,
        execution_order: List[str],
    ) -> List[List[str]]:
        """Group agents into tiers for parallel-within-tier execution.

        Tier 0: agents with in-degree 0 (no dependencies).
        Tier N: agents whose predecessors are all in tiers < N.

        Agents within the same tier have no edges between them and
        can safely execute concurrently.

        Args:
            graph: Directed acyclic graph of agent dependencies.
            execution_order: Topological ordering of agents.

        Returns:
            List of tiers, where each tier is a list of agent IDs.
        """
        if not execution_order:
            return []

        # Assign tier to each agent
        tier_of: Dict[str, int] = {}
        for agent_id in execution_order:
            predecessors = list(graph.predecessors(agent_id))
            if not predecessors:
                tier_of[agent_id] = 0
            else:
                # Tier is 1 + max tier of all predecessors
                max_pred_tier = max(
                    tier_of.get(p, 0) for p in predecessors
                )
                tier_of[agent_id] = max_pred_tier + 1

        # Group agents by tier
        max_tier = max(tier_of.values()) if tier_of else 0
        tiers: List[List[str]] = [[] for _ in range(max_tier + 1)]
        for agent_id in execution_order:
            tiers[tier_of[agent_id]].append(agent_id)

        return tiers
