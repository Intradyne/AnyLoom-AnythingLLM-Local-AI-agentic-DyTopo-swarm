"""
DyTopo Governance Module
========================

Provides failure-resilient agent execution, convergence detection, and
adaptive re-delegation strategies for multi-agent swarms.

Key features:
- Graceful degradation: never crash the MCP server on agent failures
- Timeout handling: prevent hung agents from blocking the swarm
- JSON repair: handle malformed LLM outputs without raising exceptions
- Convergence detection: identify when agents reach stable outputs (>80% similarity)
- Stalling detection: catch agents producing identical outputs across rounds
- Re-delegation recommendations: suggest modified prompts for underperforming agents

Based on the DyTopo architecture described in arXiv 2602.06039.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import random
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("dytopo.governance")

_metrics_lock = asyncio.Lock()


@dataclass
class BackendRetryPolicy:
    """Backend-aware retry configuration."""
    max_retries: int = 3
    base_delay: float = 1.0   # seconds
    max_delay: float = 8.0    # seconds
    jitter: bool = True

    @classmethod
    def for_backend(cls, backend: str = "llama-cpp") -> "BackendRetryPolicy":
        """Get retry policy for the inference backend.

        Short backoff (0.5-2s) optimized for parallel-safe operation.
        """
        return cls(max_retries=3, base_delay=0.5, max_delay=2.0, jitter=True)

    def calculate_delay(self, attempt: int) -> float:
        """Calculate backoff delay for given attempt number."""
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        if self.jitter:
            delay *= (0.5 + random.random())
        return delay


def _get_work(output: Any) -> str:
    """Extract 'work' from a dict or Pydantic AgentDescriptor."""
    if hasattr(output, "work"):
        return output.work or ""
    if isinstance(output, dict):
        return output.get("work", "")
    return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FAILURE-SAFE AGENT EXECUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def execute_agent_safe(
    agent_id: str,
    agent_call_coro: Any,
    timeout_sec: float = 120.0,
    max_retries: int = 2,
    backend: str = "llama-cpp",
) -> dict[str, Any]:
    """
    Execute an agent with retry logic and graceful failure handling.

    Returns a failure stub instead of raising exceptions, allowing the swarm
    to continue even when individual agents fail.

    Args:
        agent_id: Unique identifier for the agent
        agent_call_coro: Coroutine OR callable that returns a coroutine
                        If callable, will be called fresh for each retry attempt
        timeout_sec: Maximum time to wait for agent response (default: 120s)
        max_retries: Number of retry attempts on failure (default: 2)

    Returns:
        dict with fields:
            - success: bool indicating if agent executed successfully
            - key: agent's offering (or failure stub)
            - query: agent's information needs (or empty)
            - work: agent's work product (or failure message)
            - error: optional error message if failed
            - retries: number of retries attempted

    Example:
        >>> # With callable (recommended for retries)
        >>> async def make_call():
        ...     return await llm_call("You are a developer", "Write hello world")
        >>> result = await execute_agent_safe("developer", make_call, timeout_sec=60.0)
        >>>
        >>> # With coroutine (single attempt only)
        >>> result = await execute_agent_safe(
        ...     "developer",
        ...     llm_call("You are a developer", "Write hello world"),
        ...     timeout_sec=60.0,
        ...     max_retries=0  # Must be 0 for coroutine objects
        ... )
    """
    retries = 0
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            # Get the coroutine (call if callable, otherwise use as-is)
            if callable(agent_call_coro):
                coro = agent_call_coro()
            else:
                coro = agent_call_coro

            # Wrap the agent call with a timeout
            result = await asyncio.wait_for(coro, timeout=timeout_sec)

            # Try to parse the result if it's a string
            if isinstance(result, str):
                result = _safe_json_parse(result)

            # Validate the result has expected fields
            if not isinstance(result, dict):
                raise ValueError(f"Agent returned non-dict: {type(result)}")

            # Ensure required fields exist
            result.setdefault("key", f"Agent {agent_id} produced output")
            result.setdefault("query", "No additional information needed")
            result.setdefault("work", str(result))

            # Mark as successful
            result["success"] = True
            result["retries"] = retries
            return result

        except asyncio.TimeoutError:
            last_error = f"Timeout after {timeout_sec}s"
            logger.warning(f"Agent {agent_id} timed out (attempt {attempt + 1}/{max_retries + 1})")
            retries = attempt + 1

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            logger.warning(f"Agent {agent_id} returned invalid JSON (attempt {attempt + 1}/{max_retries + 1}): {e}")
            retries = attempt + 1

        except Exception as e:
            last_error = f"Unexpected error: {type(e).__name__}: {e}"
            logger.warning(f"Agent {agent_id} failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
            retries = attempt + 1

        # Don't retry on last attempt
        if attempt < max_retries:
            policy = BackendRetryPolicy.for_backend(backend)
            backoff_delay = policy.calculate_delay(attempt)
            logger.debug(f"Agent {agent_id} retry {attempt + 1} after {backoff_delay:.2f}s")
            await asyncio.sleep(backoff_delay)

    # All retries exhausted - return failure stub
    logger.error(f"Agent {agent_id} failed after {max_retries + 1} attempts: {last_error}")
    return {
        "success": False,
        "key": f"[FAILURE] Agent {agent_id} encountered errors and could not complete",
        "query": "Recovery assistance needed",
        "work": f"Agent {agent_id} failed to execute. Error: {last_error}",
        "error": last_error,
        "retries": retries,
    }


def _safe_json_parse(text: str) -> dict[str, Any]:
    """
    Parse JSON with fallback to json-repair for malformed outputs.

    Args:
        text: Raw text that should contain JSON

    Returns:
        Parsed dictionary

    Raises:
        json.JSONDecodeError: If parsing fails even after repair attempts
    """
    # First try standard JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

    # Try json-repair as last resort
    try:
        from json_repair import repair_json
        repaired = repair_json(text)
        return json.loads(repaired)
    except ImportError:
        logger.warning("json_repair not available, cannot repair malformed JSON")
        raise json.JSONDecodeError("Could not parse JSON and json_repair not available", text, 0)
    except Exception:
        raise json.JSONDecodeError("JSON repair failed", text, 0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONVERGENCE DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def detect_convergence(
    round_history: list[dict[str, Any]],
    window_size: int = 3,
    similarity_threshold: float = 0.80,
) -> tuple[bool, float]:
    """
    Detect if swarm outputs have converged (stopped changing).

    Compares the last `window_size` rounds and checks if the outputs are
    highly similar (>80% by default). Convergence suggests the swarm has
    reached a stable solution or is stuck.

    Args:
        round_history: List of round records, each with 'outputs' dict
        window_size: Number of recent rounds to compare (default: 3)
        similarity_threshold: Minimum similarity to consider converged (default: 0.80)

    Returns:
        Tuple of (is_converged: bool, avg_similarity: float)

    Example:
        >>> round_history = [
        ...     {"round": 1, "outputs": {"dev": {"work": "def foo(): pass"}}},
        ...     {"round": 2, "outputs": {"dev": {"work": "def foo(): pass"}}},
        ...     {"round": 3, "outputs": {"dev": {"work": "def foo(): pass"}}},
        ... ]
        >>> converged, similarity = detect_convergence(round_history, window_size=3)
        >>> print(f"Converged: {converged}, Similarity: {similarity:.2%}")
        Converged: True, Similarity: 100.00%
    """
    if len(round_history) < window_size:
        return False, 0.0

    # Extract the last `window_size` rounds
    recent_rounds = round_history[-window_size:]

    # Collect all agent IDs across these rounds
    agent_ids = set()
    for rh in recent_rounds:
        agent_ids.update(rh.get("outputs", {}).keys())

    if not agent_ids:
        return False, 0.0

    # Compare consecutive rounds for each agent
    similarities = []
    for i in range(len(recent_rounds) - 1):
        round_a = recent_rounds[i].get("outputs", {})
        round_b = recent_rounds[i + 1].get("outputs", {})

        for agent_id in agent_ids:
            # Get work products from both rounds (handles dict or Pydantic)
            work_a = _get_work(round_a.get(agent_id, {}))
            work_b = _get_work(round_b.get(agent_id, {}))

            # Skip if either is empty
            if not work_a or not work_b:
                continue

            # Compute similarity
            sim = _text_similarity(work_a, work_b)
            similarities.append(sim)

    if not similarities:
        return False, 0.0

    avg_similarity = sum(similarities) / len(similarities)
    is_converged = avg_similarity >= similarity_threshold

    if is_converged:
        logger.info(
            f"Convergence detected: {avg_similarity:.1%} similarity over "
            f"last {window_size} rounds (threshold: {similarity_threshold:.1%})"
        )

    return is_converged, avg_similarity


def _text_similarity(text_a: str, text_b: str) -> float:
    """
    Compute text similarity using difflib.SequenceMatcher.

    Args:
        text_a: First text
        text_b: Second text

    Returns:
        Similarity ratio in [0.0, 1.0]
    """
    matcher = difflib.SequenceMatcher(None, text_a, text_b)
    return matcher.ratio()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STALLING DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def detect_stalling(
    agent_id: str,
    round_history: list[dict[str, Any]],
    window_size: int = 3,
    similarity_threshold: float = 0.98,
) -> tuple[bool, float]:
    """
    Detect if a specific agent is stalling (producing near-identical outputs).

    Stalling indicates an agent is stuck in a loop or unable to progress.
    This is more targeted than convergence detection, which looks at all agents.

    Args:
        agent_id: The agent to check for stalling
        round_history: List of round records with 'outputs' dicts
        window_size: Number of recent rounds to check (default: 3)
        similarity_threshold: Minimum similarity to consider stalling (default: 0.98)

    Returns:
        Tuple of (is_stalling: bool, avg_similarity: float)

    Example:
        >>> round_history = [
        ...     {"round": 1, "outputs": {"dev": {"work": "impl v1"}}},
        ...     {"round": 2, "outputs": {"dev": {"work": "impl v1"}}},
        ...     {"round": 3, "outputs": {"dev": {"work": "impl v1"}}},
        ... ]
        >>> stalling, similarity = detect_stalling("dev", round_history)
        >>> if stalling:
        ...     print(f"Agent is stalling (similarity: {similarity:.1%})")
    """
    if len(round_history) < window_size:
        return False, 0.0

    # Extract recent rounds
    recent_rounds = round_history[-window_size:]

    # Get this agent's work products
    work_products = []
    for rh in recent_rounds:
        outputs = rh.get("outputs", {})
        if agent_id in outputs:
            work = _get_work(outputs[agent_id])
            if work:
                work_products.append(work)

    if len(work_products) < 2:
        return False, 0.0

    # Compare consecutive work products
    similarities = []
    for i in range(len(work_products) - 1):
        sim = _text_similarity(work_products[i], work_products[i + 1])
        similarities.append(sim)

    avg_similarity = sum(similarities) / len(similarities)
    is_stalling = avg_similarity >= similarity_threshold

    if is_stalling:
        logger.warning(
            f"Agent {agent_id} is stalling: {avg_similarity:.1%} similarity "
            f"over last {window_size} rounds (threshold: {similarity_threshold:.1%})"
        )

    return is_stalling, avg_similarity


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RE-DELEGATION RECOMMENDATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def recommend_redelegation(
    round_history: list[dict[str, Any]],
    agent_roles: dict[str, Any],
    stall_threshold: float = 0.98,
    failure_threshold: int = 2,
) -> list[dict[str, Any]]:
    """
    Identify agents that should be re-prompted with modified instructions.

    Analyzes agent performance across rounds to find:
    1. Stalling agents (producing identical outputs)
    2. Frequently failing agents (timeouts, errors)
    3. Agents with low routing connectivity (isolated)

    Args:
        round_history: List of round records with outputs and metadata
        agent_roles: Dict mapping agent_id -> agent configuration
        stall_threshold: Similarity threshold for stalling detection (default: 0.98)
        failure_threshold: Number of failures to trigger re-delegation (default: 2)

    Returns:
        List of dicts with:
            - agent_id: ID of the agent needing re-delegation
            - reason: Why re-delegation is recommended
            - recommendation: Specific suggestion for modification
            - severity: "high" | "medium" | "low"

    Example:
        >>> recommendations = recommend_redelegation(round_history, agent_roles)
        >>> for rec in recommendations:
        ...     print(f"{rec['agent_id']}: {rec['reason']} -> {rec['recommendation']}")
    """
    recommendations = []

    # Get all agent IDs
    agent_ids = set(agent_roles.keys())

    # Track failures per agent
    failure_counts = {aid: 0 for aid in agent_ids}
    for rh in round_history:
        outputs = rh.get("outputs", {})
        for aid, output in outputs.items():
            success = output.get("success", True) if isinstance(output, dict) else getattr(output, "success", True)
            if not success:
                failure_counts[aid] += 1

    # Check each agent
    for agent_id in agent_ids:
        # Check for stalling
        is_stalling, similarity = detect_stalling(
            agent_id, round_history, window_size=3, similarity_threshold=stall_threshold
        )

        if is_stalling:
            recommendations.append({
                "agent_id": agent_id,
                "reason": f"Stalling detected ({similarity:.1%} similarity across recent rounds)",
                "recommendation": (
                    "Modify prompt to encourage exploration of alternative approaches. "
                    "Add explicit instruction to try a different method than previously used."
                ),
                "severity": "high",
            })

        # Check for repeated failures
        fail_count = failure_counts.get(agent_id, 0)
        if fail_count >= failure_threshold:
            recommendations.append({
                "agent_id": agent_id,
                "reason": f"Multiple failures ({fail_count} failures in recent rounds)",
                "recommendation": (
                    "Simplify the agent's task scope. Consider splitting responsibilities "
                    "or reducing output complexity to improve reliability."
                ),
                "severity": "high",
            })

        # Check for low connectivity (isolation)
        connectivity = _check_agent_connectivity(agent_id, round_history)
        if connectivity < 0.2 and len(round_history) >= 3:
            recommendations.append({
                "agent_id": agent_id,
                "reason": f"Low routing connectivity ({connectivity:.1%} average edge participation)",
                "recommendation": (
                    "Revise agent's descriptor prompts to better advertise capabilities "
                    "and express information needs. Current descriptors may be too generic."
                ),
                "severity": "medium",
            })

    # Log recommendations
    if recommendations:
        logger.info(f"Generated {len(recommendations)} re-delegation recommendations")
        for rec in recommendations:
            logger.info(f"  {rec['agent_id']} [{rec['severity']}]: {rec['reason']}")

    return recommendations


def _check_agent_connectivity(agent_id: str, round_history: list[dict[str, Any]]) -> float:
    """
    Calculate average connectivity ratio for an agent across rounds.

    Args:
        agent_id: Agent to check
        round_history: Round records with edge lists

    Returns:
        Average ratio of rounds where agent had incoming/outgoing edges
    """
    if not round_history:
        return 0.0

    connected_rounds = 0
    total_rounds = 0

    for rh in round_history:
        edges = rh.get("edges", [])
        if not edges:
            continue

        total_rounds += 1

        # Check if agent appears in any edge
        for edge in edges:
            if isinstance(edge, dict):
                if edge.get("source") == agent_id or edge.get("target") == agent_id:
                    connected_rounds += 1
                    break
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                if edge[0] == agent_id or edge[1] == agent_id:
                    connected_rounds += 1
                    break

    if total_rounds == 0:
        return 0.0

    return connected_rounds / total_rounds


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  METRICS TRACKING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def update_agent_metrics(
    agent_state: dict[str, Any],
    round_result: dict[str, Any],
) -> None:
    """Update agent state with metrics from a round execution (thread-safe).

    NOTE: Changed to async to support locking for concurrent updates.

    Modifies agent_state in-place to track:
    - Total rounds participated
    - Failure count
    - Average execution time
    - Connectivity statistics

    Args:
        agent_state: Dict representing agent state (modified in-place)
        round_result: Result dict from agent execution

    Example:
        >>> agent_state = {"id": "developer", "failure_count": 0, "metrics": {}}
        >>> round_result = {"success": False, "retries": 2, "error": "Timeout"}
        >>> await update_agent_metrics(agent_state, round_result)
        >>> print(agent_state["failure_count"])
        1
    """
    async with _metrics_lock:
        # Initialize metrics if not present
        if "metrics" not in agent_state:
            agent_state["metrics"] = {}

        metrics = agent_state["metrics"]

        # Initialize counters
        metrics.setdefault("total_rounds", 0)
        metrics.setdefault("total_failures", 0)
        metrics.setdefault("total_retries", 0)

        # Update counters
        metrics["total_rounds"] += 1

        if not round_result.get("success", True):
            metrics["total_failures"] += 1
            agent_state["failure_count"] = agent_state.get("failure_count", 0) + 1

        if "retries" in round_result:
            metrics["total_retries"] += round_result["retries"]

        # Compute failure rate
        metrics["failure_rate"] = metrics["total_failures"] / max(1, metrics["total_rounds"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AEGEAN-PROTOCOL CONSENSUS TERMINATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


_embedder = None


def _get_embedder():
    """Lazy-load MiniLM-L6-v2 for consensus embedding. CPU — tiny model."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _embedder


def compute_consensus_matrix(embeddings: list[list[float]]) -> list[list[float]]:
    """Compute NxN cosine similarity matrix for consensus detection.

    Returns a nested list (not numpy array) for JSON serialization.

    Args:
        embeddings: List of embedding vectors, each a list of floats.

    Returns:
        NxN nested list of cosine similarities in [−1, 1].
    """
    import numpy as np

    arr = np.array(embeddings, dtype=np.float64)
    # Normalize each row to unit length
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # Avoid division by zero for zero-length vectors
    norms = np.where(norms == 0, 1.0, norms)
    arr_normed = arr / norms
    # Cosine similarity via dot product of normalized vectors
    sim_matrix = arr_normed @ arr_normed.T
    return sim_matrix.tolist()


async def check_aegean_termination(
    agent_outputs: list[str],
    agent_names: list[str],
    round_number: int,
    min_rounds: int = 2,
    consensus_threshold: float = 0.85,
    min_agreement_ratio: float = 0.75,
) -> tuple[bool, list]:
    """
    Aegean-protocol-inspired consensus check.

    Algorithm:
    1. If round_number < min_rounds, return (False, [])
    2. Embed all agent outputs using _get_embedder()
    3. Compute pairwise cosine similarities via compute_consensus_matrix()
    4. For each agent, compute its average similarity to all other agents
    5. An agent "votes to terminate" if its avg similarity > consensus_threshold
    6. If the fraction of termination votes >= min_agreement_ratio, terminate
    7. Return (should_terminate, list_of_AegeanVote_objects)

    Args:
        agent_outputs: List of textual outputs from each agent.
        agent_names: Corresponding agent names (same length as agent_outputs).
        round_number: Current round number (0-indexed or 1-indexed).
        min_rounds: Minimum rounds before termination can trigger.
        consensus_threshold: Per-agent avg similarity above which it votes to terminate.
        min_agreement_ratio: Fraction of agents that must vote to terminate.

    Returns:
        Tuple of (should_terminate, list_of_AegeanVote).
    """
    from dytopo.models import AegeanVote

    # Guard: not enough rounds yet
    if round_number < min_rounds:
        return False, []

    # Guard: empty or mismatched inputs
    if not agent_outputs or not agent_names:
        return False, []

    n = len(agent_outputs)

    # Single agent trivially agrees with itself — but no *other* agent to compare
    if n < 2:
        vote = AegeanVote(
            agent_name=agent_names[0],
            round_number=round_number,
            supports_termination=True,
            confidence=1.0,
        )
        return True, [vote]

    # Step 2: embed all outputs
    model = _get_embedder()
    embeddings = model.encode(agent_outputs, convert_to_numpy=True).tolist()

    # Step 3: compute pairwise cosine similarities
    sim_matrix = compute_consensus_matrix(embeddings)

    # Steps 4-5: per-agent average similarity to *other* agents, then vote
    votes: list[AegeanVote] = []
    terminate_count = 0

    for i in range(n):
        # Average similarity to all other agents (exclude self)
        other_sims = [sim_matrix[i][j] for j in range(n) if j != i]
        avg_sim = sum(other_sims) / len(other_sims)

        supports = avg_sim > consensus_threshold
        if supports:
            terminate_count += 1

        votes.append(AegeanVote(
            agent_name=agent_names[i],
            round_number=round_number,
            supports_termination=supports,
            confidence=round(avg_sim, 6),
        ))

    # Step 6: check agreement ratio
    agreement_ratio = terminate_count / n
    should_terminate = agreement_ratio >= min_agreement_ratio

    if should_terminate:
        logger.info(
            f"Aegean termination triggered at round {round_number}: "
            f"{terminate_count}/{n} agents voted ({agreement_ratio:.0%} >= {min_agreement_ratio:.0%})"
        )

    return should_terminate, votes
