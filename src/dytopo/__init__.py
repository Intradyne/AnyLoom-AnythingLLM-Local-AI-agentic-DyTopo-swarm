"""
DyTopo: Dynamic Topology Routing for Multi-Agent Swarms
========================================================

Implements semantic routing with sparse directed graphs reconstructed each round.

Based on arXiv 2602.06039, "Dynamic Topology Routing for Multi-Agent
Reasoning via Semantic Matching" (Lu et al., Feb 2026)

Core modules:
- models: Pydantic v2 data structures (SwarmTask, AgentState, etc.)
- config: YAML configuration loader with defaults + concurrency backend
- agents: System prompts, domain configs, roster builder
- router: Embedding, similarity matrix, threshold, degree cap
- graph: DAG construction, cycle breaking, topological tiers
- orchestrator: Async parallel swarm loop with backend-agnostic LLM client
- governance: Convergence detection, stalling, re-delegation
- audit: JSONL event logging

Sub-packages:
- observability: Distributed tracing, metrics, profiling, failure analysis
- safeguards: Rate limiter, token budget, circuit breaker
- messaging: Typed agent message passing
- routing: AsyncRoutingEngine with embed lock
- delegation: Subtask delegation with depth/timeout control
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
from dytopo.safeguards import (
    CircuitBreaker,
    CircuitBreakerOpen,
    PerformanceSafeguards,
    RateLimiter,
    TokenBudget,
    TokenBudgetExceeded,
)
from dytopo.messaging import AgentMessage, MessageHistory, MessageRouter
from dytopo.routing import AsyncRoutingEngine, RoutingResult
from dytopo.delegation import DelegationError, DelegationManager, DelegationRecord

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
    # Safeguards
    "CircuitBreaker",
    "CircuitBreakerOpen",
    "PerformanceSafeguards",
    "RateLimiter",
    "TokenBudget",
    "TokenBudgetExceeded",
    # Messaging
    "AgentMessage",
    "MessageHistory",
    "MessageRouter",
    # Routing (async)
    "AsyncRoutingEngine",
    "RoutingResult",
    # Delegation
    "DelegationError",
    "DelegationManager",
    "DelegationRecord",
]
