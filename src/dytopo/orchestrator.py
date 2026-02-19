"""
DyTopo Orchestrator
===================

Main swarm execution loop with dynamic semantic routing and parallel execution.

Architecture:
- Lazy singleton AsyncOpenAI client via llama.cpp inference backend
- Semaphore-controlled concurrency (max_concurrent from config)
- Round 1: Broadcast (all agents execute in parallel via asyncio.gather)
- Rounds 2+: Phase A (parallel descriptor generation) -> Phase B (routing graph)
             -> Phase C (tiered parallel execution via topological_generations)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Callable, Optional

from dytopo.models import (
    AgentDescriptor,
    AgentRole,
    AgentState,
    ManagerDecision,
    RoundRecord,
    SwarmStatus,
    SwarmTask,
)
from dytopo.config import load_config
from dytopo.agents import (
    AGENT_OUTPUT_SCHEMA,
    DESCRIPTOR_ONLY_INSTRUCTIONS,
    DESCRIPTOR_SCHEMA,
    INCOMING_MSG_TEMPLATE,
    MANAGER_OUTPUT_SCHEMA,
    build_agent_roster,
    get_role_name,
    get_system_prompt,
)
from dytopo.router import build_routing_result, log_routing_round
from dytopo.stigmergic_router import StigmergicRouter, build_trace_edges
from dytopo.graph import build_execution_graph, get_execution_order, get_execution_tiers, get_incoming_agents
from dytopo.governance import (
    check_aegean_termination,
    detect_convergence,
    recommend_redelegation,
    update_agent_metrics,
)
from dytopo.audit import SwarmAuditLog
from dytopo.health.checker import preflight_check
from dytopo.memory.writer import SwarmMemoryWriter
from inference.llm_client import get_client, reset_client

logger = logging.getLogger("dytopo.orchestrator")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Inference client (singleton in inference.llm_client)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Note: The InferenceClient singleton is now managed by get_client() from
# inference.llm_client. It provides connection pooling, health checks,
# retry logic, and per-agent token tracking.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Utility Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _truncate_at_sentence(text: str, max_chars: int = 1000) -> str:
    """Truncate at the last sentence boundary before max_chars."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_boundary = -1
    for pattern in [". ", ".\n", "! ", "!\n", "? ", "?\n"]:
        pos = truncated.rfind(pattern)
        if pos > last_boundary:
            last_boundary = pos + 1
    if last_boundary > max_chars * 0.5:
        return truncated[:last_boundary].rstrip()
    return truncated.rstrip() + "..."


def _strip_thinking_tags(raw: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 responses."""
    stripped = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    return stripped.strip()


def _extract_json(raw: str) -> dict:
    """Full pipeline: strip thinking -> find JSON -> parse -> repair -> fallback."""
    import json

    content = _strip_thinking_tags(raw)

    # Direct parse
    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        pass

    # Markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # Brace-depth extraction for nested objects
    brace_start = content.find("{")
    if brace_start != -1:
        depth = 0
        for i, char in enumerate(content[brace_start:], start=brace_start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(content[brace_start : i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break

    # json-repair fallback
    try:
        from json_repair import repair_json
        repaired = repair_json(content, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass

    # Final fallback
    logger.warning(f"Failed to parse JSON from LLM response. Raw: {raw[:200]}")
    return {
        "key": "Agent produced unstructured output",
        "query": "Unable to determine information needs",
        "work": content if content else raw,
        "_parse_failed": True,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LLM Interface
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _llm_call(
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 4096,
    json_mode: bool = False,
    response_format: dict | None = None,
    config: dict | None = None,
    agent_id: str = "",
) -> dict:
    """Single LLM call via InferenceClient with retry and timeout.

    Returns:
        {"content": str, "parsed": dict|None, "tokens_in": int, "tokens_out": int}
    """
    cfg = config or load_config()
    # InferenceClient expects a flat dict; merge llm + concurrency sections
    client_cfg = {**cfg.get("concurrency", {}), **cfg.get("llm", {})}
    client = await get_client(client_cfg)
    timeout_sec = cfg["llm"]["timeout_seconds"]

    # Build kwargs for the underlying OpenAI call
    # Note: InferenceClient handles semaphore, retry, and token tracking internally
    kwargs: dict[str, Any] = {
        "model": cfg["llm"]["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.85,
        "extra_body": {"top_k": 20, "min_p": 0.05},
    }
    if response_format:
        kwargs["response_format"] = response_format

    # Use the new inference client (handles retry, semaphore, token tracking)
    result = await client.chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        agent_id=agent_id,
        timeout=timeout_sec,
    )

    # Check for truncation
    content = result.content
    if result.content and "finish_reason" in dir(result):
        # Note: CompletionResult doesn't currently expose finish_reason
        # This is handled by the llm_client internally
        pass

    # Parse JSON if needed
    parsed = _extract_json(content) if json_mode or response_format else None

    return {
        "content": content,
        "parsed": parsed,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
    }


async def _call_manager(
    task: str,
    round_history: list[RoundRecord],
    round_num: int,
    domain,
    agent_map: dict[str, AgentState],
    config: dict,
    pre_context: str = "",
) -> tuple[ManagerDecision, int]:
    """Send task + round history summary to Manager, parse ManagerDecision.

    Args:
        pre_context: Pre-fetched RAG context injected into task framing (round 1 only).

    Returns:
        (decision, tokens_used)
    """
    manager_prompt = get_system_prompt(domain, AgentRole.MANAGER)

    manager_context = f"Task: {task}\n\n"
    if pre_context and round_num == 1:
        manager_context += f"Reference context (from workspace RAG):\n{pre_context}\n\n"
    if round_history:
        manager_context += _summarize_rounds(round_history[-3:], agent_map)
    manager_context += (
        f"Generate the goal for round {round_num}. "
        "If the task is solved, set terminate=true and provide the final_answer."
    )

    result = await _llm_call(
        messages=[
            {"role": "system", "content": manager_prompt},
            {"role": "user", "content": manager_context},
        ],
        response_format=MANAGER_OUTPUT_SCHEMA,
        temperature=config["llm"]["temperature_manager"],
        max_tokens=config["llm"]["max_tokens_manager"],
        json_mode=True,
        config=config,
        agent_id="manager",
    )

    parsed = result["parsed"] or {}
    tokens = result["tokens_in"] + result["tokens_out"]

    decision = ManagerDecision(
        goal=parsed.get("goal", f"Round {round_num}: continue working on the task"),
        terminate=parsed.get("terminate", False),
        reasoning=parsed.get("reasoning", ""),
        final_answer=parsed.get("final_answer"),
    )

    return decision, tokens


async def _call_worker(
    agent: AgentState,
    round_goal: str,
    task: str,
    incoming_messages: list[dict],
    domain,
    config: dict,
    descriptor_only: bool = False,
) -> tuple[AgentDescriptor, int]:
    """Call a worker agent with system prompt, round goal, and routed messages.

    Args:
        agent: The agent state
        round_goal: Current round goal from Manager
        task: The original task
        incoming_messages: List of {"role": str, "sim": float, "content": str}
        domain: SwarmDomain
        config: Config dict
        descriptor_only: If True, only request key/query (Phase A)

    Returns:
        (descriptor, tokens_used)
    """
    try:
        system_prompt = get_system_prompt(domain, agent.role)
    except KeyError:
        system_prompt = f"You are the {get_role_name(agent.role)} agent."

    # Build history summary from agent's history
    history_summary = "\n".join(
        f"Round {i+1}: {_truncate_at_sentence(h.get('work', ''), 500)}"
        for i, h in enumerate(agent.history[-2:])
    )

    if descriptor_only:
        user_msg = (
            f"/no_think\nRound goal: {round_goal}\n\n"
            f"Task: {task}\n\n"
            f"Your previous work:\n{history_summary}\n\n"
            f"{DESCRIPTOR_ONLY_INSTRUCTIONS}"
        )
        response_format = DESCRIPTOR_SCHEMA
        temperature = config["llm"]["temperature_descriptor"]
        max_tokens = config["llm"]["max_tokens_descriptor"]
    else:
        # Build incoming message block
        incoming_parts = []
        for msg in incoming_messages:
            incoming_parts.append(
                INCOMING_MSG_TEMPLATE.format(
                    role=msg["role"],
                    sim=msg.get("sim", 0.0),
                    content=msg["content"],
                )
            )
        incoming_block = (
            "\n\n".join(incoming_parts) if incoming_parts
            else "(no routed messages this round)"
        )

        prefix = "/no_think\n" if not agent.history else ""
        user_msg = (
            f"{prefix}Round goal: {round_goal}\n\n"
            f"Task: {task}\n\n"
            f"Your previous work:\n{history_summary}\n\n"
            f"Messages from collaborators:\n{incoming_block}\n\n"
            "Produce your work for this round."
        )
        response_format = AGENT_OUTPUT_SCHEMA
        temperature = config["llm"]["temperature_work"]
        max_tokens = config["llm"]["max_tokens_work"]

    try:
        result = await _llm_call(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            config=config,
            agent_id=agent.agent_id,
        )
        parsed = result["parsed"] or {}
        tokens = result["tokens_in"] + result["tokens_out"]

        descriptor = AgentDescriptor(
            key=parsed.get("key", ""),
            query=parsed.get("query", ""),
            work=parsed.get("work", "") if not descriptor_only else "",
        )
        return descriptor, tokens

    except Exception as e:
        error_msg = str(e)[:200]
        logger.error(f"Worker {agent.agent_id} failed: {error_msg}")
        agent.failure_count += 1
        stub = AgentDescriptor(
            key=f"[{get_role_name(agent.role)} failed: {error_msg}]",
            query="",
            work=f"[Agent {get_role_name(agent.role)} encountered an error: {error_msg}]",
        )
        return stub, 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _gather_broadcast(rounds: list[RoundRecord], exclude_agent_id: str) -> list[dict]:
    """Return all outputs from the previous round except the agent's own."""
    if not rounds:
        return []
    last = rounds[-1]
    messages = []
    for agent_id, descriptor in last.agent_outputs.items():
        if agent_id != exclude_agent_id and descriptor.work:
            messages.append({
                "role": get_role_name(AgentRole(agent_id)) if agent_id in AgentRole._value2member_map_ else agent_id,
                "sim": 0.0,
                "content": _truncate_at_sentence(descriptor.work, 500),
            })
    return messages


def _gather_routed(
    graph,
    round_outputs: dict[str, AgentDescriptor],
    target_agent_id: str,
    agent_map: dict[str, AgentState],
) -> list[dict]:
    """Return only outputs from agents with an edge pointing to target."""
    if graph is None:
        return []
    incoming_ids = get_incoming_agents(graph, target_agent_id)
    messages = []
    for src_id in incoming_ids:
        if src_id in round_outputs:
            src_agent = agent_map.get(src_id)
            role_name = get_role_name(src_agent.role) if src_agent else src_id
            sim = graph[src_id][target_agent_id].get("weight", 0.0) if graph.has_edge(src_id, target_agent_id) else 0.0
            messages.append({
                "role": role_name,
                "sim": sim,
                "content": round_outputs[src_id].work,
            })
    return messages


def _extract_best_answer(rounds: list[RoundRecord]) -> str:
    """Extract the last Developer/Solver work product as best answer."""
    preferred_roles = {AgentRole.DEVELOPER.value, AgentRole.SOLVER.value, AgentRole.SYNTHESIZER.value}
    for rnd in reversed(rounds):
        for agent_id, descriptor in rnd.agent_outputs.items():
            if agent_id in preferred_roles and descriptor.work:
                return descriptor.work
    # Fallback: last round, any agent with work
    if rounds and rounds[-1].agent_outputs:
        for descriptor in reversed(list(rounds[-1].agent_outputs.values())):
            if descriptor.work:
                return descriptor.work
    return "[No answer produced]"


def _summarize_rounds(rounds: list[RoundRecord], agent_map: dict[str, AgentState]) -> str:
    """Compact text summary of round history for the Manager's context window."""
    parts = []
    for rh in rounds:
        parts.append(f"--- Round {rh.round_num} (goal: {rh.goal}) ---")
        for aid, descriptor in rh.agent_outputs.items():
            agent = agent_map.get(aid)
            role = get_role_name(agent.role) if agent else aid
            work = _truncate_at_sentence(descriptor.work, 800)
            parts.append(f"[{role}]: {work}")
        parts.append("")
    return "\n".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main Orchestration Loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def run_swarm(
    swarm: SwarmTask,
    on_progress: Optional[Callable] = None,
) -> SwarmTask:
    """Execute the full DyTopo orchestration loop.

    Args:
        swarm: Initialized SwarmTask with task, domain, tau, T_max set
        on_progress: Optional async callback(event_type: str, data: dict)

    Returns:
        Updated SwarmTask with results
    """
    config = load_config()

    # Pre-run health check
    try:
        llm_url = config["llm"]["base_url"].rstrip("/v1").rstrip("/")
        health = await preflight_check(llm_url=llm_url)
        for comp in health.components:
            if comp.healthy:
                logger.info(f"Health OK: {comp.component} ({comp.latency_ms:.0f}ms)")
            else:
                logger.warning(f"Health FAIL: {comp.component} — {comp.error}")
        llm_status = next((c for c in health.components if c.component == "llm"), None)
        if llm_status and not llm_status.healthy:
            raise RuntimeError(f"LLM health check failed: {llm_status.error}")
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f"Pre-run health check failed (non-fatal): {e}")

    # Initialize stigmergic router if traces are enabled
    traces_cfg = config.get("traces", {})
    stigmergic_router = None
    all_trace_edges: list[dict] = []
    if traces_cfg.get("enabled", False):
        stigmergic_router = StigmergicRouter(
            qdrant_url=traces_cfg.get("qdrant_url", "http://localhost:6333"),
            enable_traces=True,
            trace_boost_weight=traces_cfg.get("boost_weight", 0.15),
            trace_half_life_hours=traces_cfg.get("half_life_hours", 168.0),
            top_k=traces_cfg.get("top_k", 5),
            min_quality=traces_cfg.get("min_quality", 0.5),
            prune_max_age_hours=traces_cfg.get("prune_max_age_hours", 720.0),
        )
        logger.info("Stigmergic router enabled (boost=%.2f, half_life=%.0fh)",
                     traces_cfg.get("boost_weight", 0.15),
                     traces_cfg.get("half_life_hours", 168.0))

    agents = build_agent_roster(swarm.domain)
    agent_map = {a.agent_id: a for a in agents if a.role != AgentRole.MANAGER}
    manager_agent = next(a for a in agents if a.role == AgentRole.MANAGER)
    agent_role_map = {a.agent_id: get_role_name(a.role) for a in agents}

    # Initialize audit log
    audit = SwarmAuditLog(swarm.task_id, base_dir=config["logging"]["log_dir"])
    agent_names = [a.agent_id for a in agents]
    audit.swarm_started(swarm.task, swarm.T_max, agent_names)

    swarm.status = SwarmStatus.RUNNING

    async def progress(event: str, data: dict):
        if on_progress:
            await on_progress(event, data)

    try:
        for t in range(1, swarm.T_max + 1):
            round_start = time.monotonic()
            await progress("round_start", {"round": t, "total": swarm.T_max})
            audit.round_started(t)

            # ── Phase 1: Manager sets goal or terminates ─────────────────────
            decision, mgr_tokens = await _call_manager(
                swarm.task, swarm.rounds, t, swarm.domain, agent_map, config,
                pre_context=swarm.context,
            )
            swarm.total_llm_calls += 1
            swarm.total_tokens += mgr_tokens

            if decision.terminate:
                swarm.final_answer = decision.final_answer or decision.goal
                swarm.status = SwarmStatus.COMPLETE
                swarm.termination_reason = "manager_halt"
                swarm.end_time = time.monotonic()
                swarm.rounds.append(RoundRecord(
                    round_num=t,
                    goal=decision.goal,
                    final_answer=decision.final_answer or "",
                ))
                await progress("manager_terminated", {"round": t, "reasoning": decision.reasoning})
                break

            round_goal = decision.goal
            round_record = RoundRecord(round_num=t, goal=round_goal)
            swarm.progress_message = f"Round {t}/{swarm.T_max}: {round_goal[:60]}..."
            await progress("manager_goal", {"round": t, "goal": round_goal})

            # ── Round 1: BROADCAST (no routing, parallel calls) ───────────────
            if t == 1 and config["routing"]["broadcast_round_1"]:
                await progress("phase_broadcast", {"round": t, "agents": list(agent_map.keys())})

                async def _broadcast_agent(aid: str, astate: AgentState):
                    desc, tok = await _call_worker(
                        astate, round_goal, swarm.task,
                        incoming_messages=[],
                        domain=swarm.domain,
                        config=config,
                    )
                    return aid, astate, desc, tok

                try:
                    broadcast_results = await asyncio.wait_for(
                        asyncio.gather(
                            *[_broadcast_agent(aid, ast) for aid, ast in agent_map.items()],
                            return_exceptions=True,
                        ),
                        timeout=120,
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Round {t} broadcast execution timed out")
                    broadcast_results = [TimeoutError(f"Round {t} timed out")] * len(agent_map)

                for result in broadcast_results:
                    if isinstance(result, Exception):
                        logger.error(f"Round 1 broadcast agent failed: {result}")
                        continue
                    agent_id, agent_state, descriptor, tokens = result
                    agent_state.descriptor = descriptor
                    agent_state.history.append(descriptor.model_dump())
                    round_record.agent_outputs[agent_id] = descriptor
                    swarm.total_llm_calls += 1
                    swarm.total_tokens += tokens

                round_record.execution_order = list(agent_map.keys())
                round_record.duration_sec = time.monotonic() - round_start
                swarm.rounds.append(round_record)

                # Check for convergence after round 1
                if len(swarm.rounds) >= 2:
                    round_history = [{"round": r.round_num, "outputs": r.agent_outputs} for r in swarm.rounds]
                    converged, similarity = detect_convergence(round_history, window_size=2, similarity_threshold=config["orchestration"]["convergence_threshold"])
                    if converged:
                        audit.convergence_detected(t, f"Similarity: {similarity:.2%}", similarity)
                        swarm.swarm_metrics.convergence_detected_at = t

                continue

            # ── Rounds 2+: TWO-PHASE SPLIT ──────────────────────────────────

            # Phase A: Descriptor-only calls (parallel, /no_think, temp 0.1)
            await progress("phase_descriptors", {"round": t})

            async def _gen_descriptor(aid: str, astate: AgentState):
                desc, tok = await _call_worker(
                    astate, round_goal, swarm.task,
                    incoming_messages=[],
                    domain=swarm.domain,
                    config=config,
                    descriptor_only=True,
                )
                return aid, astate, desc, tok

            try:
                desc_results = await asyncio.wait_for(
                    asyncio.gather(
                        *[_gen_descriptor(aid, ast) for aid, ast in agent_map.items()],
                        return_exceptions=True,
                    ),
                    timeout=120,
                )
            except asyncio.TimeoutError:
                logger.error(f"Round {t} descriptor generation timed out")
                desc_results = [TimeoutError(f"Round {t} timed out")] * len(agent_map)

            descriptors: dict[str, dict] = {}
            for result in desc_results:
                if isinstance(result, Exception):
                    logger.error(f"Descriptor generation failed: {result}")
                    continue
                agent_id, agent_state, desc, tokens = result
                swarm.total_llm_calls += 1
                swarm.total_tokens += tokens
                descriptors[agent_id] = {
                    "key": desc.key or f"{get_role_name(agent_state.role)} has general output available",
                    "query": desc.query or f"{get_role_name(agent_state.role)} can proceed independently",
                }

            # Phase B: Build routing graph (CPU-only, ~10-50ms)
            await progress("phase_routing", {"round": t})
            agent_ids = list(agent_map.keys())
            if stigmergic_router is not None:
                routing = await stigmergic_router.build_topology(
                    agent_ids, descriptors,
                    task_summary=swarm.task,
                    threshold=swarm.tau,
                    max_in_degree=swarm.K_in,
                )
                # Accumulate trace edges for post-run deposit
                all_trace_edges.extend(
                    build_trace_edges(routing["edges"], agent_role_map, t)
                )
                trace_ctx = routing.get("trace_context", {})
                if trace_ctx.get("boost_applied"):
                    logger.info(f"Round {t}: trace boost applied from {trace_ctx.get('traces_used', 0)} traces")
            else:
                routing = build_routing_result(agent_ids, descriptors, swarm.tau, swarm.K_in)
            round_record.edges = routing["edges"]
            round_record.isolated_agents = routing["isolated"]
            round_record.removed_edges = routing["removed_edges"]
            round_record.routing_stats = routing["stats"]

            # Log routing to disk
            if config["logging"]["save_similarity_matrices"]:
                log_routing_round(swarm.task_id, t, routing, config["logging"]["log_dir"])

            # Build execution graph and order
            G = build_execution_graph(routing["edges"], agent_ids)
            execution_order = get_execution_order(G, agent_ids)
            round_record.execution_order = execution_order

            # Handle full isolation fallback
            all_isolated = routing["stats"]["isolated_count"] == len(agent_ids)
            if all_isolated and config["orchestration"]["fallback_on_isolation"]:
                logger.warning(f"Round {t}: ALL agents isolated. Falling back to broadcast.")
                routing_graph = None
            else:
                routing_graph = G

            n_edges = routing["stats"]["edge_count"]
            density = routing["stats"]["density"]
            await progress("routing_done", {
                "round": t,
                "edges": n_edges,
                "density": density,
                "execution_order": execution_order,
            })

            # Phase C: Execute agents in topological tiers (parallel within tier)
            round_outputs: dict[str, AgentDescriptor] = {}
            tiers = get_execution_tiers(
                routing_graph if routing_graph is not None else G,
                list(agent_map.keys()),
            )

            for tier_idx, tier in enumerate(tiers):
                await progress("tier_start", {"round": t, "tier": tier_idx, "agents": tier})

                async def _exec_agent(aid: str):
                    astate = agent_map[aid]
                    # Collect incoming messages from prior tiers' outputs
                    if routing_graph is not None:
                        incoming = _gather_routed(routing_graph, round_outputs, aid, agent_map)
                    else:
                        incoming = []
                    # Cold start mitigation: inject round 1 outputs if round 2 and no edges
                    if not incoming and t == 2 and swarm.rounds:
                        incoming = _gather_broadcast(swarm.rounds, aid)

                    descriptor, tokens = await _call_worker(
                        astate, round_goal, swarm.task,
                        incoming_messages=incoming,
                        domain=swarm.domain,
                        config=config,
                    )
                    return aid, astate, descriptor, tokens

                try:
                    tier_results = await asyncio.wait_for(
                        asyncio.gather(
                            *[_exec_agent(aid) for aid in tier],
                            return_exceptions=True,
                        ),
                        timeout=120,
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Round {t} agent execution timed out")
                    tier_results = [TimeoutError(f"Round {t} timed out")] * len(tier)

                # Accumulate outputs — all agents in this tier complete before next tier
                for result in tier_results:
                    if isinstance(result, Exception):
                        logger.error(f"Agent execution failed in tier {tier_idx}: {result}")
                        continue
                    agent_id, agent_state, descriptor, tokens = result
                    agent_state.descriptor = descriptor
                    agent_state.history.append(descriptor.model_dump())
                    round_outputs[agent_id] = descriptor
                    round_record.agent_outputs[agent_id] = descriptor
                    swarm.total_llm_calls += 1
                    swarm.total_tokens += tokens

                await progress("tier_done", {"round": t, "tier": tier_idx, "agents": tier})

            round_record.duration_sec = time.monotonic() - round_start
            swarm.rounds.append(round_record)

            # Update swarm metrics
            swarm.swarm_metrics.total_rounds = len(swarm.rounds)
            if round_record.routing_stats:
                swarm.swarm_metrics.routing_density_per_round.append(round_record.routing_stats.get("density", 0.0))

            # Convergence check using governance module
            if len(swarm.rounds) >= 3:
                round_history = [{"round": r.round_num, "outputs": r.agent_outputs} for r in swarm.rounds]
                converged, similarity = detect_convergence(
                    round_history,
                    window_size=3,
                    similarity_threshold=config["orchestration"]["convergence_threshold"]
                )
                if converged:
                    audit.convergence_detected(t, f"Similarity: {similarity:.2%}", similarity)
                    swarm.swarm_metrics.convergence_detected_at = t
                    swarm.termination_reason = "convergence"
                    await progress("convergence", {"round": t, "similarity": similarity})
                    logger.info(f"Convergence detected at round {t} (similarity: {similarity:.2%})")
                    break

            # Aegean consensus-based early termination
            if len(swarm.rounds) >= 2 and t >= 2:
                try:
                    last_round = swarm.rounds[-1]
                    agent_outputs_text = []
                    agent_names_list = []
                    for aid, descriptor in last_round.agent_outputs.items():
                        agent_names_list.append(aid)
                        agent_outputs_text.append(descriptor.work or descriptor.key or "")

                    if agent_outputs_text:
                        should_terminate, votes = await check_aegean_termination(
                            agent_outputs=agent_outputs_text,
                            agent_names=agent_names_list,
                            round_number=t,
                            min_rounds=2,
                            consensus_threshold=config["orchestration"].get("convergence_threshold", 0.80),
                            min_agreement_ratio=0.75,
                        )
                        if should_terminate:
                            vote_count = sum(1 for v in votes if v.supports_termination)
                            logger.info(
                                f"Aegean consensus reached at round {t}: "
                                f"{vote_count}/{len(votes)} agents agree"
                            )
                            swarm.termination_reason = "aegean_consensus"
                            await progress("aegean_termination", {
                                "round": t,
                                "votes": vote_count,
                                "total_agents": len(votes),
                            })
                            break
                except Exception as e:
                    logger.debug(f"Aegean check skipped (non-fatal): {e}")

            # Check for re-delegation needs
            if len(swarm.rounds) >= 2:
                round_history = [{"round": r.round_num, "outputs": r.agent_outputs} for r in swarm.rounds]
                recommendations = recommend_redelegation(round_history, agent_map, stall_threshold=0.98)
                for rec in recommendations:
                    audit.redelegation(t, rec["agent_id"], rec["reason"], rec["recommendation"])
                    swarm.swarm_metrics.redelegations += 1
                    logger.warning(f"Re-delegation for {rec['agent_id']}: {rec['reason']}")

        # ── Final answer extraction ──────────────────────────────────────────
        if swarm.status != SwarmStatus.COMPLETE:
            # Check if any round has a final answer from manager termination
            for rh in reversed(swarm.rounds):
                if rh.final_answer:
                    swarm.final_answer = rh.final_answer
                    break

            if not swarm.final_answer:
                # Ask manager for final extraction
                await progress("final_extraction", {})
                extract_context = f"Task: {swarm.task}\n\nAll rounds complete. "
                if swarm.rounds:
                    last = swarm.rounds[-1]
                    for aid, descriptor in last.agent_outputs.items():
                        agent = agent_map.get(aid)
                        role = get_role_name(agent.role) if agent else aid
                        work = _truncate_at_sentence(descriptor.work, 1500)
                        extract_context += f"\n[{role}]: {work}\n"
                extract_context += "\nExtract the best final answer from the team's work. Set terminate=true."

                result = await _llm_call(
                    messages=[
                        {"role": "system", "content": get_system_prompt(swarm.domain, AgentRole.MANAGER)},
                        {"role": "user", "content": extract_context},
                    ],
                    response_format=MANAGER_OUTPUT_SCHEMA,
                    temperature=config["llm"]["temperature_manager"],
                    max_tokens=config["llm"]["max_tokens_work"],
                    json_mode=True,
                    config=config,
                )
                swarm.total_llm_calls += 1
                swarm.total_tokens += result["tokens_in"] + result["tokens_out"]

                final_parsed = result["parsed"] or {}
                swarm.final_answer = final_parsed.get(
                    "final_answer",
                    final_parsed.get("work", _extract_best_answer(swarm.rounds)),
                )

            swarm.status = SwarmStatus.COMPLETE
            swarm.end_time = time.monotonic()

            await progress("swarm_complete", {
                "rounds": len(swarm.rounds),
                "tokens": swarm.total_tokens,
                "wall_clock_sec": (swarm.end_time or time.monotonic()) - swarm.start_time,
            })

    except Exception as e:
        logger.error(f"Swarm {swarm.task_id} failed: {e}", exc_info=True)
        swarm.status = SwarmStatus.FAILED
        swarm.error = str(e)
        swarm.end_time = time.monotonic()
        audit.swarm_failed(len(swarm.rounds), str(e), str(e))
        raise

    finally:
        # Update final swarm metrics
        elapsed_ms = int((swarm.end_time or time.monotonic() - swarm.start_time) * 1000)
        swarm.swarm_metrics.total_rounds = len(swarm.rounds)
        swarm.swarm_metrics.total_llm_calls = swarm.total_llm_calls
        swarm.swarm_metrics.total_tokens = swarm.total_tokens
        swarm.swarm_metrics.total_wall_time_ms = elapsed_ms

        # Log final completion
        if swarm.status == SwarmStatus.COMPLETE:
            audit.swarm_completed(
                len(swarm.rounds),
                swarm.final_answer or "",
                len(swarm.rounds)
            )
        audit.close()

        # Post-run memory write (non-fatal)
        if swarm.status == SwarmStatus.COMPLETE:
            try:
                memory_writer = SwarmMemoryWriter()
                key_findings = []
                for r in swarm.rounds:
                    if r.final_answer:
                        key_findings.append(r.final_answer[:200])
                    for aid, desc in r.agent_outputs.items():
                        if desc.work:
                            key_findings.append(desc.work[:200])
                key_findings = key_findings[:10]  # cap at 10

                agent_roles = [
                    get_role_name(a.role) for a in agent_map.values()
                ] if agent_map else []

                wall_ms = int(swarm.swarm_metrics.total_wall_time_ms)
                await memory_writer.write(
                    task_description=swarm.task,
                    domain=swarm.domain.value,
                    agent_roles=agent_roles,
                    round_count=len(swarm.rounds),
                    key_findings=key_findings,
                    final_answer=swarm.final_answer or "",
                    convergence_achieved=swarm.termination_reason in ("convergence", "aegean_consensus"),
                    total_tokens=swarm.total_tokens,
                    wall_time_ms=wall_ms,
                    metadata={"task_id": swarm.task_id, "termination_reason": swarm.termination_reason},
                )
                logger.info(f"Swarm result persisted to memory for task {swarm.task_id}")
            except Exception as e:
                logger.warning(f"Post-run memory write failed (non-fatal): {e}")

        # Post-run stigmergic trace deposit (non-fatal)
        if swarm.status == SwarmStatus.COMPLETE and stigmergic_router is not None and all_trace_edges:
            try:
                agent_roles_list = [
                    get_role_name(a.role) for a in agent_map.values()
                ] if agent_map else []
                # Use convergence similarity as quality proxy (0.0-1.0)
                quality = 0.7  # default for max_rounds termination
                if swarm.termination_reason in ("convergence", "aegean_consensus"):
                    quality = 0.9
                elif swarm.termination_reason == "manager_halt":
                    quality = 0.8

                trace_id = await stigmergic_router.deposit_trace(
                    task_summary=swarm.task,
                    active_edges=all_trace_edges,
                    agent_roles=agent_roles_list,
                    rounds_to_converge=len(swarm.rounds),
                    final_answer_quality=quality,
                    convergence_method=swarm.termination_reason,
                    task_domain=swarm.domain.value,
                )
                if trace_id:
                    logger.info(f"Stigmergic trace deposited: {trace_id}")
            except Exception as e:
                logger.warning(f"Stigmergic trace deposit failed (non-fatal): {e}")

    return swarm
