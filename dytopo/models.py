"""
Pydantic models for the DyTopo swarm system.
"""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class AgentMetrics(BaseModel):
    """Per-agent performance tracking."""

    total_latency_ms: float = 0.0
    total_rounds: int = 0
    successful_rounds: int = 0
    total_tokens: int = 0

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per round in milliseconds."""
        if self.total_rounds == 0:
            return 0.0
        return self.total_latency_ms / self.total_rounds

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction (0.0 to 1.0)."""
        if self.total_rounds == 0:
            return 0.0
        return self.successful_rounds / self.total_rounds

    @property
    def avg_tokens_per_round(self) -> float:
        """Average tokens used per round."""
        if self.total_rounds == 0:
            return 0.0
        return self.total_tokens / self.total_rounds


class SwarmMetrics(BaseModel):
    """Aggregate metrics for entire swarm run."""

    total_rounds: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    per_agent: dict[str, AgentMetrics] = Field(default_factory=dict)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per round across all agents."""
        if self.total_rounds == 0:
            return 0.0
        return self.total_latency_ms / self.total_rounds

    @property
    def avg_tokens_per_round(self) -> float:
        """Average tokens per round across all agents."""
        if self.total_rounds == 0:
            return 0.0
        return self.total_tokens / self.total_rounds


class AgentDescriptor(BaseModel):
    """Basic descriptor model for an agent."""

    key: str
    query: str
    work: str


class AgentRole(str, Enum):
    """Enumeration of possible agent roles."""

    manager = "manager"
    developer = "developer"
    researcher = "researcher"
    tester = "tester"
    designer = "designer"
    parser = "parser"
    solver = "solver"
    verifier = "verifier"
    analyst = "analyst"
    critic = "critic"
    synthesizer = "synthesizer"


class AgentState(BaseModel):
    """State for a single agent."""

    history: list[dict[str, Any]] = Field(default_factory=list)
    descriptor: AgentDescriptor
    metrics: AgentMetrics = Field(default_factory=AgentMetrics)
    failure_count: int = 0


class SwarmDomain(str, Enum):
    """Enumeration of swarm domains."""

    code = "code"
    math = "math"
    general = "general"


class SwarmStatus(str, Enum):
    """Enumeration of swarm task statuses."""

    pending = "pending"
    running = "running"
    complete = "complete"
    error = "error"


class RoundRecord(BaseModel):
    """Record of a single round's data."""

    round_num: int
    agent_key: str
    query: str
    response: str
    latency_ms: float
    tokens_used: int
    timestamp: float


class SwarmTask(BaseModel):
    """Complete swarm task with rounds, agents, metrics."""

    task_id: str
    domain: SwarmDomain
    status: SwarmStatus = SwarmStatus.pending
    rounds: list[RoundRecord] = Field(default_factory=list)
    agents: dict[str, AgentState] = Field(default_factory=dict)
    swarm_metrics: SwarmMetrics = Field(default_factory=SwarmMetrics)
    error_message: str | None = None
    created_at: float
    completed_at: float | None = None
