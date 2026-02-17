"""
DyTopo Data Models
==================

Pydantic v2 data structures for multi-agent swarm orchestration.
Based on arXiv 2602.06039 (Lu et al., Feb 2026).
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AgentMetrics(BaseModel):
    """Per-agent performance tracking across a swarm run."""
    successful_rounds: int = 0
    failed_rounds: int = 0
    total_latency_ms: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    times_cited: int = 0         # how many other agents received this agent's output
    times_isolated: int = 0       # rounds where this agent had 0 incoming edges
    descriptor_quality_scores: list[float] = Field(default_factory=list)  # similarity to best match
    last_status: str = "pending"

    @property
    def avg_latency_ms(self) -> float:
        total = self.successful_rounds + self.failed_rounds
        return self.total_latency_ms / max(1, total)

    @property
    def success_rate(self) -> float:
        total = self.successful_rounds + self.failed_rounds
        return self.successful_rounds / max(1, total)

    @property
    def avg_tokens_per_round(self) -> float:
        return (self.total_tokens_in + self.total_tokens_out) / max(1, self.successful_rounds)


class SwarmMetrics(BaseModel):
    """Aggregate metrics for the entire swarm run."""
    total_rounds: int = 0
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_wall_time_ms: int = 0
    routing_density_per_round: list[float] = Field(default_factory=list)
    convergence_detected_at: Optional[int] = None  # round number, or None
    cycle_breaks: int = 0
    isolation_fallbacks: int = 0
    agent_failures: int = 0
    redelegations: int = 0
    per_agent: dict[str, AgentMetrics] = Field(default_factory=dict)


class AgentRole(str, Enum):
    MANAGER = "manager"
    DEVELOPER = "developer"
    RESEARCHER = "researcher"
    TESTER = "tester"
    DESIGNER = "designer"
    PROBLEM_PARSER = "problem_parser"
    SOLVER = "solver"
    VERIFIER = "verifier"
    ANALYST = "analyst"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"


class SwarmDomain(str, Enum):
    CODE = "code"
    MATH = "math"
    GENERAL = "general"


class AgentDescriptor(BaseModel):
    key: str = ""        # what this agent offers
    query: str = ""      # what this agent needs
    work: str = ""       # main work product


class AgentState(BaseModel):
    agent_id: str
    role: AgentRole
    history: list[dict] = Field(default_factory=list)  # per-round outputs
    descriptor: AgentDescriptor = Field(default_factory=AgentDescriptor)
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    failure_count: int = 0
    metrics: AgentMetrics = Field(default_factory=AgentMetrics)


class RoundRecord(BaseModel):
    round_num: int
    goal: str
    agent_outputs: dict[str, AgentDescriptor] = Field(default_factory=dict)
    edges: list[tuple[str, str, float]] = Field(default_factory=list)  # (src, tgt, weight)
    isolated_agents: list[str] = Field(default_factory=list)
    execution_order: list[str] = Field(default_factory=list)
    removed_edges: list[tuple[str, str]] = Field(default_factory=list)
    routing_stats: dict = Field(default_factory=dict)
    duration_sec: float = 0.0
    final_answer: str = ""
    timestamp: float = Field(default_factory=time.monotonic)


class ManagerDecision(BaseModel):
    goal: str
    terminate: bool = False
    reasoning: str = ""
    final_answer: Optional[str] = None


class SwarmStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class SwarmTask(BaseModel):
    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    task: str
    domain: SwarmDomain
    context: str = Field(default="", description="Pre-fetched RAG context injected into Manager's task framing")
    status: SwarmStatus = SwarmStatus.PENDING
    tau: float = 0.5
    K_in: int = 3
    T_max: int = 5
    rounds: list[RoundRecord] = Field(default_factory=list)
    final_answer: Optional[str] = None
    error: Optional[str] = None
    start_time: float = Field(default_factory=time.monotonic)
    end_time: Optional[float] = None
    total_llm_calls: int = 0
    total_tokens: int = 0
    termination_reason: str = "max_rounds"
    progress_message: str = ""
    swarm_metrics: SwarmMetrics = Field(default_factory=SwarmMetrics)
