"""
AnyLoom System Status MCP Server
=================================
FastMCP server that exposes stack diagnostics as agent-callable tools.
Lets the LLM agent query the health and status of all AnyLoom
infrastructure components: LLM server, Qdrant, AnythingLLM, GPU,
Docker containers, and configuration.

MCP Tools (6 total):
  service_health()      - Probe all stack components in parallel
  qdrant_collections()  - List Qdrant collections with point counts
  gpu_status()          - GPU temperature, utilization, VRAM via nvidia-smi
  llm_slots()           - Active inference slots on llama.cpp
  docker_status()       - Running anyloom-* Docker containers
  stack_config()        - Current port/model/embedding configuration

Transport: stdio (stdout is JSON-RPC, all logging goes to stderr)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path manipulation — ensure the dytopo package is importable regardless
# of the CWD the server is launched from.
# ---------------------------------------------------------------------------
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Logging — stderr only (stdout is JSON-RPC for MCP)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("system-status")

# ---------------------------------------------------------------------------
# MCP Server Instance
# ---------------------------------------------------------------------------
mcp = FastMCP("system-status")

# ---------------------------------------------------------------------------
# Default endpoints (overridable via env vars)
# ---------------------------------------------------------------------------
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8008")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
ANYTHINGLLM_URL = os.environ.get("ANYTHINGLLM_URL", "http://localhost:3001")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Tool 1: service_health
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@mcp.tool()
async def service_health() -> str:
    """Check health of all AnyLoom stack components.

    Probes LLM (port 8008), Qdrant (port 6333),
    AnythingLLM (port 3001), and GPU in parallel.
    Returns structured status for each component.
    """
    try:
        from dytopo.health.checker import HealthChecker

        checker = HealthChecker(
            llm_url=LLM_URL,
            qdrant_url=QDRANT_URL,
            anythingllm_url=ANYTHINGLLM_URL,
        )
        stack = await checker.check_all()

        lines: list[str] = []
        lines.append(f"Stack healthy: {'YES' if stack.all_healthy else 'NO'}")
        lines.append("")

        for comp in stack.components:
            status = "OK" if comp.healthy else "FAIL"
            line = f"  [{status}] {comp.component:<14} latency={comp.latency_ms:.0f}ms"
            if comp.error:
                line += f"  error: {comp.error}"
            if comp.details:
                # Include selected details without overwhelming the output
                detail_str = _format_details(comp.component, comp.details)
                if detail_str:
                    line += f"  {detail_str}"
            lines.append(line)

        return "\n".join(lines)
    except Exception as exc:
        logger.error("service_health failed: %s", exc, exc_info=True)
        return f"Error checking service health: {exc}"


def _format_details(component: str, details: dict) -> str:
    """Extract the most useful detail fields for a component."""
    try:
        if component == "gpu" and "gpus" in details:
            parts = []
            for gpu in details["gpus"]:
                name = gpu.get("name", "?")
                temp = gpu.get("temperature_c", "?")
                util = gpu.get("utilization_pct", "?")
                mem_used = gpu.get("memory_used_mb", 0)
                mem_total = gpu.get("memory_total_mb", 0)
                parts.append(
                    f"{name} {temp}C {util}% {mem_used:.0f}/{mem_total:.0f}MB"
                )
            return " | ".join(parts)
        if component == "qdrant" and "result" in details:
            collections = details["result"].get("collections", [])
            return f"collections={len(collections)}"
        if component == "llm" and "status" in details:
            return f"status={details['status']}"
    except Exception:
        pass
    return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Tool 2: qdrant_collections
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@mcp.tool()
async def qdrant_collections() -> str:
    """List all Qdrant collections with point counts and status."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{QDRANT_URL}/collections")
            resp.raise_for_status()
            data = resp.json()

        collections = data.get("result", {}).get("collections", [])
        if not collections:
            return "No collections found in Qdrant."

        lines: list[str] = [f"Qdrant collections ({len(collections)}):"]
        lines.append("")

        # Fetch details for each collection
        for coll in collections:
            name = coll.get("name", "unknown")
            detail_line = await _get_collection_detail(name)
            lines.append(detail_line)

        return "\n".join(lines)
    except Exception as exc:
        logger.error("qdrant_collections failed: %s", exc, exc_info=True)
        return f"Error listing Qdrant collections: {exc}"


async def _get_collection_detail(name: str) -> str:
    """Fetch point count and status for a single Qdrant collection."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{QDRANT_URL}/collections/{name}")
            resp.raise_for_status()
            info = resp.json().get("result", {})

        points = info.get("points_count", "?")
        status = info.get("status", "?")
        vectors_count = info.get("vectors_count", "?")
        segments = len(info.get("segments", []))
        # Extract vector config names if available
        vector_names = []
        vectors_config = info.get("config", {}).get("params", {}).get("vectors", {})
        if isinstance(vectors_config, dict):
            vector_names = list(vectors_config.keys())
        sparse_config = info.get("config", {}).get("params", {}).get("sparse_vectors", {})
        if isinstance(sparse_config, dict):
            vector_names.extend(f"{k}(sparse)" for k in sparse_config.keys())

        vec_str = f"  vectors=[{', '.join(vector_names)}]" if vector_names else ""
        return (
            f"  {name}: {points} points, status={status}, "
            f"vectors={vectors_count}, segments={segments}{vec_str}"
        )
    except Exception as exc:
        return f"  {name}: error fetching details ({exc})"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Tool 3: gpu_status
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@mcp.tool()
async def gpu_status() -> str:
    """GPU temperature, utilization, and VRAM usage via nvidia-smi."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=name,driver_version,temperature.gpu,utilization.gpu,"
            "utilization.memory,memory.used,memory.total,memory.free,"
            "power.draw,power.limit,clocks.current.graphics,clocks.current.memory",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)

        if proc.returncode != 0:
            err = stderr.decode().strip()
            return f"nvidia-smi failed (exit {proc.returncode}): {err}"

        output = stdout.decode().strip()
        if not output:
            return "nvidia-smi returned empty output."

        lines: list[str] = ["GPU Status:"]
        lines.append("")

        for i, line in enumerate(output.splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 12:
                lines.append(f"  GPU {i}: (parse error) {line}")
                continue

            name = parts[0]
            driver = parts[1]
            temp = parts[2]
            gpu_util = parts[3]
            mem_util = parts[4]
            mem_used = parts[5]
            mem_total = parts[6]
            mem_free = parts[7]
            power_draw = parts[8]
            power_limit = parts[9]
            clk_gpu = parts[10]
            clk_mem = parts[11]

            lines.append(f"  GPU {i}: {name}")
            lines.append(f"    Driver:       {driver}")
            lines.append(f"    Temperature:  {temp} C")
            lines.append(f"    GPU util:     {gpu_util}%")
            lines.append(f"    Memory util:  {mem_util}%")
            lines.append(f"    VRAM:         {mem_used} / {mem_total} MB ({mem_free} MB free)")
            lines.append(f"    Power:        {power_draw} / {power_limit} W")
            lines.append(f"    Clocks:       GPU {clk_gpu} MHz, Memory {clk_mem} MHz")

        return "\n".join(lines)
    except FileNotFoundError:
        return "nvidia-smi not found. No NVIDIA GPU drivers installed or not on PATH."
    except asyncio.TimeoutError:
        return "nvidia-smi timed out after 10 seconds."
    except Exception as exc:
        logger.error("gpu_status failed: %s", exc, exc_info=True)
        return f"Error querying GPU status: {exc}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Tool 4: llm_slots
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@mcp.tool()
async def llm_slots() -> str:
    """Active inference slots on the llama.cpp server."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{LLM_URL}/slots")

            if resp.status_code == 404:
                return (
                    "The /slots endpoint is not available on this llama.cpp build. "
                    "Slot monitoring requires llama-server compiled with LLAMA_SERVER_SLOT_MONITORING."
                )
            resp.raise_for_status()
            slots = resp.json()

        if not isinstance(slots, list):
            return f"Unexpected /slots response format: {type(slots).__name__}"

        if not slots:
            return "No inference slots reported by llama.cpp server."

        lines: list[str] = [f"LLM inference slots ({len(slots)}):"]
        lines.append("")

        for slot in slots:
            slot_id = slot.get("id", "?")
            state = slot.get("state", "?")
            # State: 0 = idle, 1 = processing
            state_label = {0: "idle", 1: "processing"}.get(state, str(state))

            n_prompt = slot.get("n_decoded", 0)
            n_predicted = slot.get("n_predicted", 0)
            model = slot.get("model", "")

            line = f"  Slot {slot_id}: {state_label}"
            if state == 1:
                # Show active slot details
                prompt_tokens = slot.get("prompt_tokens", slot.get("n_ctx", "?"))
                line += f"  decoded={n_prompt} predicted={n_predicted}"
                task_id = slot.get("task_id", None)
                if task_id is not None:
                    line += f"  task={task_id}"
            else:
                line += f"  (decoded={n_prompt}, predicted={n_predicted} total)"

            if model:
                line += f"  model={model}"

            lines.append(line)

        # Summary stats if available
        idle_count = sum(1 for s in slots if s.get("state") == 0)
        busy_count = sum(1 for s in slots if s.get("state") == 1)
        lines.append("")
        lines.append(f"  Summary: {busy_count} busy, {idle_count} idle, {len(slots)} total")

        return "\n".join(lines)
    except Exception as exc:
        logger.error("llm_slots failed: %s", exc, exc_info=True)
        return f"Error querying LLM slots: {exc}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Tool 5: docker_status
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@mcp.tool()
async def docker_status() -> str:
    """List running Docker containers matching the anyloom-* pattern."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "ps",
            "--filter", "name=anyloom",
            "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)

        if proc.returncode != 0:
            err = stderr.decode().strip()
            return f"docker ps failed (exit {proc.returncode}): {err}"

        output = stdout.decode().strip()
        if not output:
            return "No running Docker containers matching 'anyloom-*'."

        lines: list[str] = ["AnyLoom Docker containers:"]
        lines.append("")

        for row in output.splitlines():
            parts = row.split("\t")
            if len(parts) >= 4:
                name, status, ports, image = parts[0], parts[1], parts[2], parts[3]
                lines.append(f"  {name}")
                lines.append(f"    Image:  {image}")
                lines.append(f"    Status: {status}")
                if ports:
                    lines.append(f"    Ports:  {ports}")
            elif len(parts) >= 3:
                name, status, ports = parts[0], parts[1], parts[2]
                lines.append(f"  {name}")
                lines.append(f"    Status: {status}")
                if ports:
                    lines.append(f"    Ports:  {ports}")
            else:
                lines.append(f"  {row}")

        return "\n".join(lines)
    except FileNotFoundError:
        return "Docker CLI not found. Docker may not be installed or not on PATH."
    except asyncio.TimeoutError:
        return "docker ps timed out after 10 seconds."
    except Exception as exc:
        logger.error("docker_status failed: %s", exc, exc_info=True)
        return f"Error querying Docker status: {exc}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Tool 6: stack_config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@mcp.tool()
async def stack_config() -> str:
    """Current stack configuration: ports, model, embedding settings."""
    try:
        from dytopo.config import load_config

        config = load_config()

        lines: list[str] = ["AnyLoom Stack Configuration:"]
        lines.append("")

        # LLM section
        llm = config.get("llm", {})
        lines.append("LLM:")
        lines.append(f"  Base URL:              {llm.get('base_url', '?')}")
        lines.append(f"  Model:                 {llm.get('model', '?')}")
        lines.append(f"  Temperature (work):    {llm.get('temperature_work', '?')}")
        lines.append(f"  Temperature (manager): {llm.get('temperature_manager', '?')}")
        lines.append(f"  Max tokens (work):     {llm.get('max_tokens_work', '?')}")
        lines.append(f"  Max tokens (manager):  {llm.get('max_tokens_manager', '?')}")
        lines.append(f"  Timeout:               {llm.get('timeout_seconds', '?')}s")

        # Routing section
        routing = config.get("routing", {})
        lines.append("")
        lines.append("Routing:")
        lines.append(f"  Embedding model:  {routing.get('embedding_model', '?')}")
        lines.append(f"  Tau (threshold):  {routing.get('tau', '?')}")
        lines.append(f"  K_in (max edges): {routing.get('K_in', '?')}")
        lines.append(f"  Adaptive tau:     {routing.get('adaptive_tau', '?')}")
        lines.append(f"  Broadcast R1:     {routing.get('broadcast_round_1', '?')}")

        # Orchestration section
        orch = config.get("orchestration", {})
        lines.append("")
        lines.append("Orchestration:")
        lines.append(f"  Max rounds (T_max):        {orch.get('T_max', '?')}")
        lines.append(f"  Descriptor mode:           {orch.get('descriptor_mode', '?')}")
        lines.append(f"  State strategy:            {orch.get('state_strategy', '?')}")
        lines.append(f"  Convergence threshold:     {orch.get('convergence_threshold', '?')}")
        lines.append(f"  Fallback on isolation:     {orch.get('fallback_on_isolation', '?')}")
        lines.append(f"  Max agent context tokens:  {orch.get('max_agent_context_tokens', '?')}")

        # Concurrency section
        conc = config.get("concurrency", {})
        lines.append("")
        lines.append("Concurrency:")
        lines.append(f"  Backend:          {conc.get('backend', '?')}")
        lines.append(f"  Max concurrent:   {conc.get('max_concurrent', '?')}")
        lines.append(f"  Connect timeout:  {conc.get('connect_timeout', '?')}s")
        lines.append(f"  Read timeout:     {conc.get('read_timeout', '?')}s")

        # Traces section
        traces = config.get("traces", {})
        lines.append("")
        lines.append("Swarm Traces:")
        lines.append(f"  Enabled:        {traces.get('enabled', '?')}")
        lines.append(f"  Qdrant URL:     {traces.get('qdrant_url', '?')}")
        lines.append(f"  Collection:     {traces.get('collection', '?')}")
        lines.append(f"  Boost weight:   {traces.get('boost_weight', '?')}")

        # Health monitor section
        health = config.get("health_monitor", {})
        lines.append("")
        lines.append("Health Monitor:")
        lines.append(f"  Check interval:    {health.get('check_interval_seconds', '?')}s")
        lines.append(f"  Max restarts:      {health.get('max_restart_attempts', '?')}")
        lines.append(f"  Crash window:      {health.get('crash_window_minutes', '?')} min")
        lines.append(f"  Alert cooldown:    {health.get('alert_cooldown_minutes', '?')} min")

        # Active endpoint overrides from env vars
        lines.append("")
        lines.append("Active Endpoints (env overrides):")
        lines.append(f"  LLM_URL:          {LLM_URL}")
        lines.append(f"  QDRANT_URL:       {QDRANT_URL}")
        lines.append(f"  ANYTHINGLLM_URL:  {ANYTHINGLLM_URL}")

        return "\n".join(lines)
    except Exception as exc:
        logger.error("stack_config failed: %s", exc, exc_info=True)
        return f"Error loading stack configuration: {exc}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    logger.info("Starting AnyLoom System Status MCP Server")
    logger.info("LLM: %s, Qdrant: %s, AnythingLLM: %s", LLM_URL, QDRANT_URL, ANYTHINGLLM_URL)
    mcp.run(transport="stdio")
