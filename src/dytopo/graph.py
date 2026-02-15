"""
DyTopo Graph Construction
=========================

Directed graph building, cycle breaking, and topological sort for execution order.
"""

import bisect
import logging
import networkx as nx

logger = logging.getLogger("dytopo.graph")


def build_execution_graph(
    edges: list[tuple[str, str, float]],
    agent_ids: list[str],
) -> nx.DiGraph:
    """Build a directed graph from routing edges.

    Edge (src, tgt, weight) means src sends output to tgt.
    Execution order: sources before targets (topological sort).

    Args:
        edges: List of (source_id, target_id, weight) tuples
        agent_ids: All agent IDs (includes isolated agents with no edges)

    Returns:
        NetworkX DiGraph with all agents as nodes
    """
    G = nx.DiGraph()
    G.add_nodes_from(agent_ids)
    for src, tgt, weight in edges:
        G.add_edge(src, tgt, weight=weight)
    return G


def break_cycles(G: nx.DiGraph) -> list[tuple[str, str]]:
    """Remove minimum edges to make G a DAG. Returns list of removed (src, tgt) pairs.

    Strategy: Find all simple cycles, remove the edge with lowest weight in each cycle.
    With 4-5 agents and sparse connectivity, cycles are rare.

    Args:
        G: Directed graph (modified in-place)

    Returns:
        List of removed (source, target) edge pairs
    """
    removed = []
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G)
        except nx.NetworkXNoCycle:
            break

        # Find weakest edge in the cycle
        min_weight = float("inf")
        min_edge = None
        for u, v in cycle:
            w = G[u][v].get("weight", 0.0)
            if w < min_weight:
                min_weight = w
                min_edge = (u, v)

        if min_edge:
            G.remove_edge(*min_edge)
            removed.append(min_edge)
            logger.info(f"Broke cycle: removed edge {min_edge[0]} → {min_edge[1]} (weight={min_weight:.3f})")

    return removed


def get_execution_order(G: nx.DiGraph, agent_ids: list[str]) -> list[str]:
    """Return topological execution order. Isolated nodes appended at end.

    Uses Kahn's algorithm with alphabetical tiebreaking for determinism.

    If the graph has no edges, returns agent_ids as-is (broadcast order).

    Args:
        G: Directed graph (must be a DAG; call break_cycles first if needed)
        agent_ids: All agent IDs for fallback order

    Returns:
        List of agent IDs in execution order
    """
    # First, break any cycles
    removed = break_cycles(G)
    if removed:
        logger.info(f"Removed {len(removed)} edges to break cycles")

    # Kahn's algorithm with alphabetical tiebreaking
    try:
        in_deg = dict(G.in_degree())
        queue = sorted([n for n in G.nodes() if in_deg[n] == 0])
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for succ in sorted(G.successors(node)):
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    bisect.insort(queue, succ)

        # Ensure all agents appear (isolated agents may not have edges but are still nodes)
        for aid in agent_ids:
            if aid not in result:
                result.append(aid)

        return result

    except Exception as e:
        logger.warning(f"Topological sort failed: {e}. Using original order.")
        return agent_ids


def get_incoming_agents(G: nx.DiGraph, target_id: str) -> list[str]:
    """Return list of agent IDs that have edges pointing TO target_id.

    Args:
        G: Directed graph
        target_id: Target agent ID

    Returns:
        List of source agent IDs that send messages to target_id
    """
    if not G.has_node(target_id):
        return []
    return [src for src, _ in G.in_edges(target_id)]
