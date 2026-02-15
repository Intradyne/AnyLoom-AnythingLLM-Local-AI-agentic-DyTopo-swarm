"""
DyTopo: Dynamic Topology Routing for Multi-Agent Swarms
========================================================

Implements semantic routing with sparse directed graphs reconstructed each round.

Based on arXiv 2602.06039, "Dynamic Topology Routing for Multi-Agent
Reasoning via Semantic Matching" (Lu et al., Feb 2026)

Core modules:
- models: Pydantic v2 data structures (SwarmTask, AgentState, etc.)
- config: YAML configuration loader with defaults
- agents: System prompts, domain configs, roster builder
- router: Embedding, similarity matrix, threshold, degree cap
- graph: DAG construction, cycle breaking, topological sort
- orchestrator: Main swarm execution loop with singleton LLM client
"""

from dytopo.models import (
    AgentDescriptor,
    AgentMetrics,
    AgentRole,
    AgentState,
    ManagerDecision,
    RoundRecord,
    SwarmDomain,
    SwarmMetrics,
    SwarmStatus,
    SwarmTask,
)
from dytopo.config import load_config
from dytopo.agents import (
    AGENT_OUTPUT_SCHEMA,
    DESCRIPTOR_SCHEMA,
    MANAGER_OUTPUT_SCHEMA,
    build_agent_roster,
    get_role_name,
    get_system_prompt,
    get_worker_names,
)
from dytopo.orchestrator import run_swarm
from dytopo.governance import (
    execute_agent_safe,
    detect_convergence,
    detect_stalling,
    recommend_redelegation,
    update_agent_metrics,
)
from dytopo.audit import SwarmAuditLog

__all__ = [
    # Models
    "AgentDescriptor",
    "AgentMetrics",
    "AgentRole",
    "AgentState",
    "ManagerDecision",
    "RoundRecord",
    "SwarmDomain",
    "SwarmMetrics",
    "SwarmStatus",
    "SwarmTask",
    # Config
    "load_config",
    # Agents
    "AGENT_OUTPUT_SCHEMA",
    "DESCRIPTOR_SCHEMA",
    "MANAGER_OUTPUT_SCHEMA",
    "build_agent_roster",
    "get_role_name",
    "get_system_prompt",
    "get_worker_names",
    # Orchestrator
    "run_swarm",
    # Governance
    "execute_agent_safe",
    "detect_convergence",
    "detect_stalling",
    "recommend_redelegation",
    "update_agent_metrics",
    # Audit
    "SwarmAuditLog",
]
