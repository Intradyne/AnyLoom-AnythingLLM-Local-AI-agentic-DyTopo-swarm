"""
Trace Visualizer CLI for DyTopo Swarm Audit Logs.

Reads audit.jsonl files produced by SwarmAuditLog and generates either
Mermaid flowchart diagrams or self-contained HTML timeline visualizations.

Usage:
    python scripts/visualize_trace.py <log_dir> [--format mermaid|html] [--output FILE] [--detect-loops]

Arguments:
    log_dir          Path to directory containing audit.jsonl, or direct path to a .jsonl file.

Options:
    --format         Output format: "mermaid" or "html" (default: "html")
    --output         Output file path (default: stdout for mermaid, trace.html for html)
    --detect-loops   Enable loop/stall detection and report findings

Examples:
    python scripts/visualize_trace.py ~/dytopo-logs/task_abc123 --format mermaid
    python scripts/visualize_trace.py ~/dytopo-logs/task_abc123 --format html --output report.html
    python scripts/visualize_trace.py ~/dytopo-logs/task_abc123 --detect-loops
    python scripts/visualize_trace.py ~/dytopo-logs/task_abc123/audit.jsonl --format mermaid --output flow.md
"""

import argparse
import html as html_module
import json
import os
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _resolve_audit_path(log_dir_arg):
    """
    Resolve the audit.jsonl file path from a directory or direct file path.

    Args:
        log_dir_arg: Path string from the CLI positional argument.

    Returns:
        pathlib.Path to the audit.jsonl file.

    Raises:
        SystemExit: If the path cannot be resolved to a valid audit file.
    """
    p = Path(log_dir_arg).expanduser().resolve()

    if p.is_file():
        return p

    if p.is_dir():
        candidate = p / "audit.jsonl"
        if candidate.is_file():
            return candidate
        print(f"Error: No audit.jsonl found in directory: {p}", file=sys.stderr)
        sys.exit(1)

    print(f"Error: Path does not exist: {p}", file=sys.stderr)
    sys.exit(1)


def parse_audit_file(path):
    """
    Parse an audit.jsonl file into a list of event dicts.

    Malformed lines are skipped with a warning on stderr.

    Args:
        path: pathlib.Path to the audit.jsonl file.

    Returns:
        List of parsed event dictionaries.
    """
    events = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    print(
                        f"Warning: line {line_no} is not a JSON object, skipping.",
                        file=sys.stderr,
                    )
                    continue
                events.append(event)
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: line {line_no} is malformed JSON ({exc}), skipping.",
                    file=sys.stderr,
                )
    return events


# ---------------------------------------------------------------------------
# Organize events by round
# ---------------------------------------------------------------------------

def organize_by_round(events):
    """
    Organize flat event list into per-round data structures.

    Returns:
        dict mapping round number to a dict with keys:
            agents_executed: list of agent_executed events
            agents_failed:   list of agent_failed events
            routing:         routing_computed event or None
            convergence:     convergence_detected event or None
            redelegations:   list of redelegation events
            round_started:   round_started event or None
        Also returns metadata dict with swarm_started, swarm_completed,
        swarm_failed events.
    """
    rounds = {}
    metadata = {
        "swarm_started": None,
        "swarm_completed": None,
        "swarm_failed": None,
    }

    for event in events:
        etype = event.get("event_type", "")
        rnd = event.get("round", 0)

        # Handle metadata-level events
        if etype == "swarm_started":
            metadata["swarm_started"] = event
            continue
        if etype == "swarm_completed":
            metadata["swarm_completed"] = event
            continue
        if etype == "swarm_failed":
            metadata["swarm_failed"] = event
            continue

        # Ensure round bucket exists
        if rnd not in rounds:
            rounds[rnd] = {
                "agents_executed": [],
                "agents_failed": [],
                "routing": None,
                "convergence": None,
                "redelegations": [],
                "round_started": None,
            }

        bucket = rounds[rnd]

        if etype == "round_started":
            bucket["round_started"] = event
        elif etype == "routing_computed":
            bucket["routing"] = event
        elif etype == "agent_executed":
            bucket["agents_executed"].append(event)
        elif etype == "agent_failed":
            bucket["agents_failed"].append(event)
        elif etype == "convergence_detected":
            bucket["convergence"] = event
        elif etype == "redelegation":
            bucket["redelegations"].append(event)
        # Unknown event types are silently ignored

    return rounds, metadata


# ---------------------------------------------------------------------------
# Loop / stall detection
# ---------------------------------------------------------------------------

def detect_loops(rounds, metadata):
    """
    Analyze trace for problematic patterns.

    Detects:
        1. Ping-pong pairs: agents A->B and B->A routing for 3+ consecutive rounds.
        2. Empty routing rounds: rounds with no routing edges.
        3. Convergence regression: convergence score decreasing over 2+ rounds.

    Args:
        rounds: dict from organize_by_round
        metadata: dict from organize_by_round

    Returns:
        List of finding dicts with keys: type, description, rounds.
    """
    findings = []
    sorted_rnds = sorted(rounds.keys())

    # --- 1. Ping-pong detection ---
    # Build per-round edge sets (as frozensets of (from, to) pairs)
    round_edges = {}
    for rnd in sorted_rnds:
        routing = rounds[rnd].get("routing")
        edges = set()
        if routing:
            router_output = routing.get("router_output")
            if isinstance(router_output, dict):
                # router_output may have agent->targets mappings or an edges list
                for key, val in router_output.items():
                    if isinstance(val, list):
                        for target in val:
                            if isinstance(target, str):
                                edges.add((key, target))
                            elif isinstance(target, dict) and "agent" in target:
                                edges.add((key, target["agent"]))
                    elif isinstance(val, str):
                        edges.add((key, val))
            elif isinstance(router_output, list):
                for item in router_output:
                    if isinstance(item, dict):
                        src = item.get("from") or item.get("source", "")
                        tgt = item.get("to") or item.get("target", "")
                        if src and tgt:
                            edges.add((src, tgt))
            # Also derive edges from redelegations
            for rd in rounds[rnd].get("redelegations", []):
                from_a = rd.get("from_agent", "")
                for to_a in rd.get("to_agents", []):
                    if from_a and to_a:
                        edges.add((from_a, to_a))
        round_edges[rnd] = edges

    # Check for ping-pong: pair (A,B) appears as (A->B) in one round and (B->A) in next
    pair_streak = {}  # (A,B) canonical pair -> list of consecutive round numbers
    for idx, rnd in enumerate(sorted_rnds):
        edges = round_edges.get(rnd, set())
        current_pairs = set()
        for src, tgt in edges:
            canonical = tuple(sorted([src, tgt]))
            current_pairs.add(canonical)
        # Check reverse in next round
        if idx + 1 < len(sorted_rnds):
            next_rnd = sorted_rnds[idx + 1]
            next_edges = round_edges.get(next_rnd, set())
            for src, tgt in edges:
                if (tgt, src) in next_edges:
                    canonical = tuple(sorted([src, tgt]))
                    if canonical not in pair_streak:
                        pair_streak[canonical] = []
                    if rnd not in pair_streak[canonical]:
                        pair_streak[canonical].append(rnd)
                    if next_rnd not in pair_streak[canonical]:
                        pair_streak[canonical].append(next_rnd)

    for pair, rnds in pair_streak.items():
        # Check for 3+ consecutive rounds
        consecutive = _longest_consecutive_run(rnds, sorted_rnds)
        if consecutive >= 3:
            findings.append({
                "type": "ping_pong",
                "description": (
                    f"Ping-pong detected between {pair[0]} and {pair[1]} "
                    f"for {consecutive} consecutive rounds."
                ),
                "rounds": sorted(rnds),
            })

    # --- 2. Empty routing rounds ---
    empty_rounds = []
    for rnd in sorted_rnds:
        routing = rounds[rnd].get("routing")
        agents_exec = rounds[rnd].get("agents_executed", [])
        agents_fail = rounds[rnd].get("agents_failed", [])
        selected = []
        if routing:
            selected = routing.get("selected_agents", [])
        if not selected and not agents_exec and not agents_fail:
            empty_rounds.append(rnd)

    if empty_rounds:
        findings.append({
            "type": "empty_routing",
            "description": (
                f"Empty routing rounds (no agents selected or executed): "
                f"rounds {empty_rounds}."
            ),
            "rounds": empty_rounds,
        })

    # --- 3. Convergence regression ---
    conv_scores = []
    for rnd in sorted_rnds:
        conv = rounds[rnd].get("convergence")
        if conv and conv.get("confidence") is not None:
            try:
                score = float(conv["confidence"])
                conv_scores.append((rnd, score))
            except (ValueError, TypeError):
                pass

    if len(conv_scores) >= 2:
        regression_runs = []
        current_run = []
        for i in range(1, len(conv_scores)):
            prev_rnd, prev_score = conv_scores[i - 1]
            curr_rnd, curr_score = conv_scores[i]
            if curr_score < prev_score:
                if not current_run:
                    current_run = [prev_rnd, curr_rnd]
                else:
                    current_run.append(curr_rnd)
            else:
                if len(current_run) >= 2:
                    regression_runs.append(list(current_run))
                current_run = []
        if len(current_run) >= 2:
            regression_runs.append(list(current_run))

        for run in regression_runs:
            findings.append({
                "type": "convergence_regression",
                "description": (
                    f"Convergence score decreased over rounds {run}."
                ),
                "rounds": run,
            })

    return findings


def _longest_consecutive_run(round_list, all_rounds_sorted):
    """
    Find the length of the longest consecutive run of rounds from round_list
    within the global round ordering.
    """
    if not round_list:
        return 0
    round_set = set(round_list)
    best = 0
    current = 0
    for rnd in all_rounds_sorted:
        if rnd in round_set:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


# ---------------------------------------------------------------------------
# Mermaid generation
# ---------------------------------------------------------------------------

def _sanitize_mermaid_id(name, rnd):
    """Create a valid Mermaid node ID from agent name and round."""
    safe = name.replace(" ", "_").replace("-", "_").replace(".", "_")
    # Remove any characters that are not alphanumeric or underscore
    safe = "".join(c for c in safe if c.isalnum() or c == "_")
    return f"{safe}_round_{rnd}"


def generate_mermaid(rounds, metadata, loop_findings=None):
    """
    Generate a Mermaid flowchart TD string from parsed trace data.

    Args:
        rounds: dict from organize_by_round
        metadata: dict from organize_by_round
        loop_findings: optional list from detect_loops

    Returns:
        String containing the full Mermaid diagram.
    """
    lines = ["flowchart TD"]
    sorted_rnds = sorted(rounds.keys())

    # Track stalled round numbers for red edge styling
    stalled_rounds = set()
    if loop_findings:
        for finding in loop_findings:
            if finding["type"] == "ping_pong":
                stalled_rounds.update(finding["rounds"])

    failed_nodes = []
    stalled_edges = []
    edge_counter = 0

    for rnd in sorted_rnds:
        bucket = rounds[rnd]
        lines.append(f"    subgraph Round_{rnd}[\"Round {rnd}\"]")

        # Collect all agent names mentioned in this round
        agent_names = set()
        for ae in bucket["agents_executed"]:
            name = ae.get("agent_name", "unknown")
            agent_names.add(name)
        for af in bucket["agents_failed"]:
            name = af.get("agent_name", "unknown")
            agent_names.add(name)

        # Also add selected agents from routing
        routing = bucket.get("routing")
        if routing:
            for sa in routing.get("selected_agents", []):
                if isinstance(sa, str):
                    agent_names.add(sa)

        # Create nodes for each agent
        for name in sorted(agent_names):
            node_id = _sanitize_mermaid_id(name, rnd)
            lines.append(f"        {node_id}[\"{name}\"]")

        # Convergence note
        conv = bucket.get("convergence")
        if conv:
            reason = conv.get("reason", "converged")
            confidence = conv.get("confidence")
            label = f"Converged: {reason}"
            if confidence is not None:
                label += f" ({confidence})"
            note_id = f"conv_note_{rnd}"
            lines.append(f"        {note_id}([\"" + label.replace('"', "'") + "\"])")

        lines.append("    end")

    # Add routing edges between rounds (source_round_N --> target_round_M)
    for rnd in sorted_rnds:
        bucket = rounds[rnd]
        routing = bucket.get("routing")
        if not routing:
            continue

        router_output = routing.get("router_output")
        selected_agents = routing.get("selected_agents", [])

        # Try to extract edges with similarity scores from router_output
        edges_to_draw = []

        if isinstance(router_output, dict):
            for src, targets in router_output.items():
                if isinstance(targets, list):
                    for t in targets:
                        if isinstance(t, dict):
                            tgt = t.get("agent") or t.get("name", "")
                            sim = t.get("similarity") or t.get("score", "")
                            if tgt:
                                edges_to_draw.append((src, tgt, sim))
                        elif isinstance(t, str):
                            edges_to_draw.append((src, t, ""))
                elif isinstance(targets, (int, float)):
                    # src is agent name, targets is a score — pair with selected_agents
                    pass
        elif isinstance(router_output, list):
            for item in router_output:
                if isinstance(item, dict):
                    src = item.get("from") or item.get("source", "")
                    tgt = item.get("to") or item.get("target", "")
                    sim = item.get("similarity") or item.get("score", "")
                    if src and tgt:
                        edges_to_draw.append((src, tgt, sim))

        # Also draw redelegation edges
        for rd in bucket.get("redelegations", []):
            from_a = rd.get("from_agent", "")
            for to_a in rd.get("to_agents", []):
                if from_a and to_a:
                    edges_to_draw.append((from_a, to_a, "redelegate"))

        # If no structured edges but we have selected_agents, create simple
        # edges from previous-round agents to this round's selected agents
        if not edges_to_draw and selected_agents and rnd > 0:
            prev_rnd = None
            for candidate in sorted(rounds.keys()):
                if candidate < rnd:
                    prev_rnd = candidate
            if prev_rnd is not None:
                prev_agents = set()
                for ae in rounds[prev_rnd]["agents_executed"]:
                    prev_agents.add(ae.get("agent_name", "unknown"))
                for sa in selected_agents:
                    if isinstance(sa, str):
                        for pa in prev_agents:
                            src_id = _sanitize_mermaid_id(pa, prev_rnd)
                            tgt_id = _sanitize_mermaid_id(sa, rnd)
                            lines.append(f"    {src_id} --> {tgt_id}")

        for src, tgt, sim in edges_to_draw:
            # Determine which rounds these agents belong to
            src_rnd = None
            tgt_rnd = rnd
            # Source agent is likely in a previous round or same round
            for candidate in sorted(rounds.keys()):
                if candidate <= rnd:
                    bucket_c = rounds[candidate]
                    names_c = set()
                    for ae in bucket_c["agents_executed"]:
                        names_c.add(ae.get("agent_name"))
                    for af in bucket_c["agents_failed"]:
                        names_c.add(af.get("agent_name"))
                    if src in names_c:
                        src_rnd = candidate

            if src_rnd is None:
                src_rnd = rnd  # fallback: same round

            src_id = _sanitize_mermaid_id(src, src_rnd)
            tgt_id = _sanitize_mermaid_id(tgt, tgt_rnd)

            if sim != "" and sim is not None:
                sim_label = sim
                if isinstance(sim, float):
                    sim_label = f"{sim:.2f}"
                edge_line = f"    {src_id} -->|\"{sim_label}\"| {tgt_id}"
            else:
                edge_line = f"    {src_id} --> {tgt_id}"

            lines.append(edge_line)

            # Track stalled edges for styling
            if rnd in stalled_rounds:
                stalled_edges.append(edge_counter)
            edge_counter += 1

    # Style failed agents in red
    for rnd in sorted_rnds:
        for af in rounds[rnd]["agents_failed"]:
            name = af.get("agent_name", "unknown")
            node_id = _sanitize_mermaid_id(name, rnd)
            failed_nodes.append(node_id)

    for node_id in failed_nodes:
        lines.append(f"    style {node_id} fill:#ff6b6b")

    # Style stalled edges in red
    for edge_idx in stalled_edges:
        lines.append(f"    linkStyle {edge_idx} stroke:#ff0000,stroke-width:2px")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _escape(text):
    """HTML-escape a string."""
    if text is None:
        return ""
    return html_module.escape(str(text))


def _truncate(text, max_len=500):
    """Truncate text to max_len characters, appending ellipsis if needed."""
    text = str(text) if text is not None else ""
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _round_status(bucket):
    """Determine the status of a round: converged, failed, or active."""
    if bucket.get("convergence"):
        return "converged"
    if bucket["agents_failed"]:
        return "failed"
    return "active"


def generate_html(rounds, metadata, loop_findings=None):
    """
    Generate a self-contained HTML timeline visualization.

    Args:
        rounds: dict from organize_by_round
        metadata: dict from organize_by_round
        loop_findings: optional list from detect_loops

    Returns:
        String containing the full HTML document.
    """
    sorted_rnds = sorted(rounds.keys())
    swarm_started = metadata.get("swarm_started")
    swarm_completed = metadata.get("swarm_completed")
    swarm_failed = metadata.get("swarm_failed")

    # Determine stalled rounds from loop findings
    stalled_rounds = set()
    if loop_findings:
        for finding in loop_findings:
            stalled_rounds.update(finding.get("rounds", []))

    parts = []

    # --- Header and CSS ---
    parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DyTopo Swarm Trace</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #f5f5f5;
    color: #333;
    padding: 20px;
    line-height: 1.5;
}
h1 { margin-bottom: 10px; font-size: 1.6em; }
.meta { color: #666; font-size: 0.9em; margin-bottom: 20px; padding: 12px; background: #fff; border-radius: 6px; border-left: 4px solid #4a90d9; }
.meta p { margin: 2px 0; }
.timeline { position: relative; padding-left: 30px; }
.timeline::before {
    content: "";
    position: absolute;
    left: 14px;
    top: 0;
    bottom: 0;
    width: 3px;
    background: #ccc;
}
.round {
    position: relative;
    margin-bottom: 24px;
    background: #fff;
    border-radius: 8px;
    padding: 16px 20px;
    border-left: 4px solid #999;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.round::before {
    content: "";
    position: absolute;
    left: -24px;
    top: 20px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #999;
    border: 2px solid #fff;
}
.round.converged { border-left-color: #4caf50; }
.round.converged::before { background: #4caf50; }
.round.active { border-left-color: #ff9800; }
.round.active::before { background: #ff9800; }
.round.failed { border-left-color: #f44336; }
.round.failed::before { background: #f44336; }
.round.stalled { border-left-color: #f44336; border-left-style: dashed; }
.round.stalled::before { background: #f44336; }
.round-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.round-header h2 { font-size: 1.1em; }
.round-header .badge {
    font-size: 0.75em;
    padding: 2px 8px;
    border-radius: 10px;
    color: #fff;
    font-weight: 600;
    text-transform: uppercase;
}
.badge.converged { background: #4caf50; }
.badge.active { background: #ff9800; }
.badge.failed { background: #f44336; }
.badge.stalled { background: #f44336; }
.goal { font-size: 0.85em; color: #666; margin-bottom: 10px; font-style: italic; }
.agents { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }
.agent-card {
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 10px 14px;
    min-width: 220px;
    max-width: 400px;
    flex: 1;
}
.agent-card.success { border-left: 3px solid #4caf50; }
.agent-card.error { border-left: 3px solid #f44336; background: #fff5f5; }
.agent-card .agent-name { font-weight: 600; font-size: 0.95em; margin-bottom: 4px; }
.agent-card .agent-time { font-size: 0.8em; color: #888; }
.agent-card .agent-output {
    font-size: 0.82em;
    color: #555;
    margin-top: 6px;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 120px;
    overflow-y: auto;
    background: #fafafa;
    padding: 6px;
    border-radius: 4px;
}
.routing-edges { font-size: 0.85em; color: #555; margin-top: 8px; }
.routing-edges ul { list-style: none; padding: 0; }
.routing-edges li { padding: 2px 0; }
.routing-edges li::before { content: "\\2192 "; color: #999; }
.convergence-info {
    margin-top: 8px;
    padding: 6px 10px;
    background: #e8f5e9;
    border-radius: 4px;
    font-size: 0.85em;
    color: #2e7d32;
}
.redelegation-info {
    margin-top: 8px;
    padding: 6px 10px;
    background: #fff3e0;
    border-radius: 4px;
    font-size: 0.85em;
    color: #e65100;
}
.result-section {
    margin-top: 20px;
    padding: 16px;
    border-radius: 8px;
    background: #fff;
}
.result-section.success { border-left: 4px solid #4caf50; }
.result-section.failure { border-left: 4px solid #f44336; }
.result-section h2 { font-size: 1.1em; margin-bottom: 8px; }
.result-section .output { font-size: 0.85em; white-space: pre-wrap; word-break: break-word; }
.loop-findings {
    margin-top: 20px;
    padding: 16px;
    background: #fff;
    border-left: 4px solid #ff9800;
    border-radius: 8px;
}
.loop-findings h2 { font-size: 1.1em; margin-bottom: 8px; color: #e65100; }
.loop-findings .finding {
    padding: 6px 0;
    border-bottom: 1px solid #f0f0f0;
    font-size: 0.9em;
}
.loop-findings .finding:last-child { border-bottom: none; }
.finding-type {
    display: inline-block;
    font-size: 0.75em;
    padding: 1px 6px;
    border-radius: 8px;
    background: #ff9800;
    color: #fff;
    font-weight: 600;
    margin-right: 6px;
    text-transform: uppercase;
}
</style>
</head>
<body>
""")

    # --- Title and metadata ---
    task_id = ""
    query = ""
    if swarm_started:
        task_id = swarm_started.get("task_id", "")
        query = swarm_started.get("query", "")
        max_rounds = swarm_started.get("max_rounds", "")
        agents = swarm_started.get("agents", [])
        ts = swarm_started.get("timestamp", "")
    else:
        max_rounds = ""
        agents = []
        ts = ""

    parts.append(f"<h1>DyTopo Swarm Trace</h1>")
    parts.append('<div class="meta">')
    if task_id:
        parts.append(f"<p><strong>Task ID:</strong> {_escape(task_id)}</p>")
    if query:
        parts.append(f"<p><strong>Query:</strong> {_escape(_truncate(query, 200))}</p>")
    if ts:
        parts.append(f"<p><strong>Started:</strong> {_escape(ts)}</p>")
    if max_rounds:
        parts.append(f"<p><strong>Max Rounds:</strong> {_escape(str(max_rounds))}</p>")
    if agents:
        agent_names = ", ".join(_escape(str(a)) for a in agents)
        parts.append(f"<p><strong>Agents:</strong> {agent_names}</p>")
    parts.append("</div>")

    # --- Timeline ---
    parts.append('<div class="timeline">')

    for rnd in sorted_rnds:
        bucket = rounds[rnd]
        status = _round_status(bucket)
        if rnd in stalled_rounds:
            status = "stalled"

        parts.append(f'<div class="round {status}">')
        parts.append('<div class="round-header">')
        parts.append(f"<h2>Round {rnd}</h2>")
        parts.append(f'<span class="badge {status}">{status}</span>')
        parts.append("</div>")

        # Goal from routing
        routing = bucket.get("routing")
        if routing:
            selected = routing.get("selected_agents", [])
            if selected:
                goal_text = "Selected agents: " + ", ".join(
                    _escape(str(a)) for a in selected
                )
                parts.append(f'<div class="goal">{goal_text}</div>')

        # Agent cards
        all_agents = bucket["agents_executed"] + bucket["agents_failed"]
        if all_agents:
            parts.append('<div class="agents">')
            for ae in bucket["agents_executed"]:
                name = ae.get("agent_name", "unknown")
                exec_time = ae.get("execution_time")
                output = ae.get("output", "")
                parts.append('<div class="agent-card success">')
                parts.append(f'<div class="agent-name">{_escape(name)}</div>')
                if exec_time is not None:
                    parts.append(
                        f'<div class="agent-time">{float(exec_time):.2f}s</div>'
                    )
                if output:
                    parts.append(
                        f'<div class="agent-output">{_escape(_truncate(str(output)))}</div>'
                    )
                parts.append("</div>")

            for af in bucket["agents_failed"]:
                name = af.get("agent_name", "unknown")
                error = af.get("error", "")
                parts.append('<div class="agent-card error">')
                parts.append(f'<div class="agent-name">{_escape(name)}</div>')
                parts.append('<div class="agent-time">FAILED</div>')
                if error:
                    parts.append(
                        f'<div class="agent-output">{_escape(_truncate(str(error)))}</div>'
                    )
                parts.append("</div>")
            parts.append("</div>")

        # Routing edges as text list
        if routing:
            edges_text = _extract_routing_edges_text(routing)
            if edges_text:
                parts.append('<div class="routing-edges"><strong>Routing:</strong><ul>')
                for edge in edges_text:
                    parts.append(f"<li>{_escape(edge)}</li>")
                parts.append("</ul></div>")

        # Redelegation info
        for rd in bucket.get("redelegations", []):
            from_a = rd.get("from_agent", "?")
            to_a = rd.get("to_agents", [])
            reason = rd.get("reason", "")
            to_str = ", ".join(str(a) for a in to_a)
            parts.append(
                f'<div class="redelegation-info">'
                f"<strong>Redelegation:</strong> {_escape(from_a)} "
                f"delegated to [{_escape(to_str)}]"
            )
            if reason:
                parts.append(f" &mdash; {_escape(reason)}")
            parts.append("</div>")

        # Convergence info
        conv = bucket.get("convergence")
        if conv:
            reason = conv.get("reason", "")
            confidence = conv.get("confidence")
            conv_text = f"Convergence: {_escape(reason)}"
            if confidence is not None:
                conv_text += f" (confidence: {confidence})"
            parts.append(f'<div class="convergence-info">{conv_text}</div>')

        parts.append("</div>")  # close .round

    parts.append("</div>")  # close .timeline

    # --- Swarm result ---
    if swarm_completed:
        parts.append('<div class="result-section success">')
        parts.append("<h2>Swarm Completed</h2>")
        total = swarm_completed.get("total_rounds", "")
        if total:
            parts.append(f"<p>Total rounds: {_escape(str(total))}</p>")
        final = swarm_completed.get("final_output", "")
        if final:
            parts.append(
                f'<div class="output">{_escape(_truncate(str(final), 2000))}</div>'
            )
        parts.append("</div>")

    if swarm_failed:
        parts.append('<div class="result-section failure">')
        parts.append("<h2>Swarm Failed</h2>")
        reason = swarm_failed.get("reason", "")
        error = swarm_failed.get("error", "")
        if reason:
            parts.append(f"<p><strong>Reason:</strong> {_escape(reason)}</p>")
        if error:
            parts.append(f'<div class="output">{_escape(_truncate(str(error), 2000))}</div>')
        parts.append("</div>")

    # --- Loop findings ---
    if loop_findings:
        parts.append('<div class="loop-findings">')
        parts.append("<h2>Loop / Stall Detection Results</h2>")
        for finding in loop_findings:
            ftype = finding.get("type", "unknown")
            desc = finding.get("description", "")
            parts.append('<div class="finding">')
            parts.append(f'<span class="finding-type">{_escape(ftype)}</span>')
            parts.append(f"{_escape(desc)}")
            parts.append("</div>")
        parts.append("</div>")

    # --- Footer ---
    now = datetime.now().isoformat(timespec="seconds")
    parts.append(f'<p style="margin-top:20px;font-size:0.75em;color:#aaa;">Generated {_escape(now)}</p>')
    parts.append("</body>\n</html>\n")

    return "\n".join(parts)


def _extract_routing_edges_text(routing_event):
    """
    Extract human-readable edge descriptions from a routing_computed event.

    Returns:
        List of strings like "AgentA -> AgentB (similarity: 0.73)"
    """
    edges = []
    router_output = routing_event.get("router_output")
    selected_agents = routing_event.get("selected_agents", [])

    if isinstance(router_output, dict):
        for src, targets in router_output.items():
            if isinstance(targets, list):
                for t in targets:
                    if isinstance(t, dict):
                        tgt = t.get("agent") or t.get("name", "?")
                        sim = t.get("similarity") or t.get("score")
                        edge_str = f"{src} -> {tgt}"
                        if sim is not None:
                            edge_str += f" (similarity: {sim})"
                        edges.append(edge_str)
                    elif isinstance(t, str):
                        edges.append(f"{src} -> {t}")
            elif isinstance(targets, str):
                edges.append(f"{src} -> {targets}")
    elif isinstance(router_output, list):
        for item in router_output:
            if isinstance(item, dict):
                src = item.get("from") or item.get("source", "?")
                tgt = item.get("to") or item.get("target", "?")
                sim = item.get("similarity") or item.get("score")
                edge_str = f"{src} -> {tgt}"
                if sim is not None:
                    edge_str += f" (similarity: {sim})"
                edges.append(edge_str)

    # If we couldn't extract structured edges, just list selected agents
    if not edges and selected_agents:
        for sa in selected_agents:
            edges.append(f"-> {sa}")

    return edges


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_output(content, output_path, default_name=None):
    """
    Write content to output_path, or stdout if output_path is None and
    default_name is None.

    Args:
        content: string to write
        output_path: explicit --output path, or None
        default_name: fallback filename if output_path is None (None means stdout)
    """
    if output_path:
        target = Path(output_path)
    elif default_name:
        target = Path(default_name)
    else:
        sys.stdout.write(content)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"Output written to: {target}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize DyTopo swarm audit traces as Mermaid diagrams or HTML timelines.",
        epilog="Examples:\n"
               "  python scripts/visualize_trace.py ~/dytopo-logs/task_abc --format mermaid\n"
               "  python scripts/visualize_trace.py ~/dytopo-logs/task_abc --format html --output report.html\n"
               "  python scripts/visualize_trace.py ~/dytopo-logs/task_abc --detect-loops\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "log_dir",
        help="Path to directory containing audit.jsonl, or direct path to a .jsonl file.",
    )
    parser.add_argument(
        "--format",
        choices=["mermaid", "html"],
        default="html",
        dest="fmt",
        help='Output format: "mermaid" or "html" (default: html).',
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: stdout for mermaid, trace.html for html).",
    )
    parser.add_argument(
        "--detect-loops",
        action="store_true",
        default=False,
        help="Enable loop/stall detection and include findings in output.",
    )

    args = parser.parse_args()

    # Resolve audit file path
    audit_path = _resolve_audit_path(args.log_dir)

    # Parse events
    events = parse_audit_file(audit_path)
    if not events:
        print("Warning: No events found in audit file.", file=sys.stderr)

    # Organize by round
    rounds, metadata = organize_by_round(events)

    # Loop detection
    loop_findings = None
    if args.detect_loops:
        loop_findings = detect_loops(rounds, metadata)
        if loop_findings:
            print(f"Loop detection: {len(loop_findings)} finding(s):", file=sys.stderr)
            for f in loop_findings:
                print(f"  [{f['type']}] {f['description']}", file=sys.stderr)
        else:
            print("Loop detection: No issues found.", file=sys.stderr)

    # Generate output
    if args.fmt == "mermaid":
        content = generate_mermaid(rounds, metadata, loop_findings)
        default_out = None  # stdout
    else:
        content = generate_html(rounds, metadata, loop_findings)
        default_out = "trace.html"

    write_output(content, args.output, default_out)


if __name__ == "__main__":
    main()
