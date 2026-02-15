"""
DyTopo Package
==============

Dynamic Topology Routing for Multi-Agent Swarms with governance capabilities.

This package provides:
- Governance and failure-handling utilities
- Pydantic models for type-safe swarm state management
- Audit utilities for monitoring swarm execution
"""

from dytopo.governance import (
    execute_agent_safe,
    detect_convergence,
    detect_stalling,
    recommend_redelegation,
    update_agent_metrics,
)
from dytopo.models import (
    AgentState,
    AgentDescriptor,
    AgentMetrics,
    RoundRecord,
    SwarmTask,
    SwarmMetrics,
    AgentRole,
    SwarmDomain,
    SwarmStatus,
)

__all__ = [
    # Governance functions
    "execute_agent_safe",
    "detect_convergence",
    "detect_stalling",
    "recommend_redelegation",
    "update_agent_metrics",
    # Pydantic models
    "AgentState",
    "AgentDescriptor",
    "AgentMetrics",
    "RoundRecord",
    "SwarmTask",
    "SwarmMetrics",
    "AgentRole",
    "SwarmDomain",
    "SwarmStatus",
]
