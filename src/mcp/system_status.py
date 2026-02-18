"""AnyLoom System Status MCP Server.

Probes all stack components (llama.cpp, Qdrant, Docker, GPU) and returns
a unified health report. Each probe runs independently with a 3-second
timeout so one slow or unreachable service never blocks the rest.

Components monitored:
  - llama.cpp server (LLM inference) at http://localhost:8008
  - Qdrant vector DB at http://localhost:6333
  - Docker containers (via `docker ps`)
  - NVIDIA GPU (via `nvidia-smi`)

Usage:
  python -m mcp.system_status        # run as MCP server (stdio transport)
  python src/mcp/system_status.py    # run directly
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

import httpx
from fastmcp import FastMCP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LOGGING — stderr only (stdout is JSON-RPC for MCP)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("system-status")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROBE_TIMEOUT = 3.0  # seconds per probe

LLAMA_CPP_HEALTH_URL = "http://localhost:8008/health"
QDRANT_COLLECTIONS_URL = "http://localhost:6333/collections"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INDIVIDUAL PROBES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def probe_llm() -> dict[str, Any]:
    """Probe llama.cpp server health endpoint.

    GET http://localhost:8008/health returns JSON with status, model info,
    and context size when the server is running.
    """
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT) as client:
            resp = await client.get(LLAMA_CPP_HEALTH_URL)
            resp.raise_for_status()
            data = resp.json()

            # llama.cpp /health returns {"status": "ok"} at minimum;
            # extended fields vary by version
            status = data.get("status", "unknown")
            return {
                "healthy": status in ("ok", "no slot available"),
                "status": status,
                "model": data.get("model", None),
                "context_size": data.get("n_ctx", None),
                "slots_idle": data.get("slots_idle", None),
                "slots_processing": data.get("slots_processing", None),
            }
    except httpx.TimeoutException:
        return {"healthy": False, "error": "timeout (llama.cpp did not respond within 3s)"}
    except httpx.ConnectError:
        return {"healthy": False, "error": "connection refused (llama.cpp not running?)"}
    except Exception as exc:
        return {"healthy": False, "error": str(exc)}


async def probe_qdrant() -> dict[str, Any]:
    """Probe Qdrant vector DB collections endpoint.

    GET http://localhost:6333/collections returns a list of collections
    with their point counts and status.
    """
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT) as client:
            resp = await client.get(QDRANT_COLLECTIONS_URL)
            resp.raise_for_status()
            data = resp.json()

            collections_raw = data.get("result", {}).get("collections", [])
            collections = []
            for coll in collections_raw:
                collections.append({
                    "name": coll.get("name"),
                    "points_count": coll.get("points_count"),
                })

            return {
                "healthy": True,
                "collections_count": len(collections),
                "collections": collections,
            }
    except httpx.TimeoutException:
        return {"healthy": False, "error": "timeout (Qdrant did not respond within 3s)"}
    except httpx.ConnectError:
        return {"healthy": False, "error": "connection refused (Qdrant not running?)"}
    except Exception as exc:
        return {"healthy": False, "error": str(exc)}


async def probe_docker() -> dict[str, Any]:
    """Probe running Docker containers via `docker ps --format json`.

    Uses asyncio.subprocess to avoid blocking the event loop.
    Each line of output is a JSON object describing one container.
    """
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "docker", "ps", "--format", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=PROBE_TIMEOUT,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=PROBE_TIMEOUT,
        )

        if proc.returncode != 0:
            err_msg = stderr.decode(errors="replace").strip() if stderr else "unknown error"
            return {"healthy": False, "error": f"docker ps failed: {err_msg}"}

        # Parse each line as a JSON object (one per container)
        containers = []
        raw = stdout.decode(errors="replace").strip()
        if raw:
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    container = json.loads(line)
                    containers.append({
                        "name": container.get("Names", container.get("Name", "unknown")),
                        "image": container.get("Image", "unknown"),
                        "status": container.get("Status", "unknown"),
                        "ports": container.get("Ports", ""),
                    })
                except json.JSONDecodeError:
                    continue

        return {
            "healthy": True,
            "container_count": len(containers),
            "containers": containers,
        }
    except asyncio.TimeoutError:
        return {"healthy": False, "error": "timeout (docker ps did not respond within 3s)"}
    except FileNotFoundError:
        return {"healthy": False, "error": "docker command not found"}
    except Exception as exc:
        return {"healthy": False, "error": str(exc)}


async def probe_gpu() -> dict[str, Any]:
    """Probe NVIDIA GPU via nvidia-smi.

    Queries: name, temperature, utilization, memory used/total.
    Uses asyncio.subprocess to avoid blocking the event loop.
    """
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=PROBE_TIMEOUT,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=PROBE_TIMEOUT,
        )

        if proc.returncode != 0:
            err_msg = stderr.decode(errors="replace").strip() if stderr else "unknown error"
            return {"healthy": False, "error": f"nvidia-smi failed: {err_msg}"}

        raw = stdout.decode(errors="replace").strip()
        if not raw:
            return {"healthy": False, "error": "nvidia-smi returned empty output"}

        # Parse CSV: name, temp, util%, mem_used, mem_total
        # May have multiple GPUs (one per line)
        gpus = []
        for line in raw.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue

            name = parts[0]
            try:
                temp_c = int(parts[1])
            except (ValueError, IndexError):
                temp_c = None
            try:
                utilization_pct = int(parts[2])
            except (ValueError, IndexError):
                utilization_pct = None
            try:
                memory_used_mib = int(parts[3])
            except (ValueError, IndexError):
                memory_used_mib = None
            try:
                memory_total_mib = int(parts[4])
            except (ValueError, IndexError):
                memory_total_mib = None

            gpu_info: dict[str, Any] = {"name": name}
            if temp_c is not None:
                gpu_info["temperature_c"] = temp_c
            if utilization_pct is not None:
                gpu_info["utilization_pct"] = utilization_pct
            if memory_used_mib is not None and memory_total_mib is not None:
                gpu_info["memory_used_mib"] = memory_used_mib
                gpu_info["memory_total_mib"] = memory_total_mib
                gpu_info["memory_free_mib"] = memory_total_mib - memory_used_mib

            gpus.append(gpu_info)

        if not gpus:
            return {"healthy": False, "error": "could not parse nvidia-smi output"}

        return {
            "healthy": True,
            "gpu_count": len(gpus),
            "gpus": gpus,
        }
    except asyncio.TimeoutError:
        return {"healthy": False, "error": "timeout (nvidia-smi did not respond within 3s)"}
    except FileNotFoundError:
        return {"healthy": False, "error": "nvidia-smi command not found"}
    except Exception as exc:
        return {"healthy": False, "error": str(exc)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MCP SERVER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

mcp = FastMCP("AnyLoom System Status")


async def _system_status_impl() -> dict:
    """Core implementation: probe all components and return unified status.

    Separated from the MCP tool decorator so it can be called directly
    in tests without going through the FastMCP FunctionTool wrapper.
    """
    # Run all probes concurrently — each has its own timeout
    llm_result, qdrant_result, docker_result, gpu_result = await asyncio.gather(
        probe_llm(),
        probe_qdrant(),
        probe_docker(),
        probe_gpu(),
    )

    return {
        "llm": llm_result,
        "qdrant": qdrant_result,
        "docker": docker_result,
        "gpu": gpu_result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@mcp.tool()
async def system_status() -> dict:
    """Return current stack health and status.

    Probes all AnyLoom stack components concurrently (llama.cpp, Qdrant,
    Docker, GPU) with independent 3-second timeouts. Returns a dict with
    keys: llm, qdrant, docker, gpu, timestamp.

    Each component value is a dict with at minimum a ``healthy`` boolean
    and component-specific detail fields. If a component is unreachable,
    ``healthy`` is False and an ``error`` field describes the failure.
    """
    return await _system_status_impl()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    logger.info("Starting AnyLoom System Status MCP Server")
    mcp.run()
