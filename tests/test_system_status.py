"""Tests for the AnyLoom System Status MCP tool.

Validates that:
- system_status returns the expected structure even when all services are down
- Individual probe functions handle mocked success responses correctly
- Timeout handling works for each probe
- Failures are reported gracefully (healthy=False, error message)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp.system_status import (
    _system_status_impl,
    probe_docker,
    probe_gpu,
    probe_llm,
    probe_qdrant,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _make_httpx_response(status_code: int, json_body: dict) -> MagicMock:
    """Create a mock httpx.Response with the given status code and JSON body."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        resp.raise_for_status.return_value = None
    return resp


def _make_async_client_mock(response: MagicMock):
    """Create a mock httpx.AsyncClient context manager returning the given response."""
    client_mock = AsyncMock()
    client_mock.get = AsyncMock(return_value=response)
    client_mock.__aenter__ = AsyncMock(return_value=client_mock)
    client_mock.__aexit__ = AsyncMock(return_value=False)
    return client_mock


def _make_subprocess_mock(
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> AsyncMock:
    """Create a mock asyncio subprocess process."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(
        return_value=(stdout.encode(), stderr.encode())
    )
    return proc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SYSTEM_STATUS — FULL INTEGRATION (MOCKED)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_system_status_returns_expected_structure_all_down():
    """When all services are unreachable, system_status still returns
    the correct top-level structure with healthy=False for every component."""

    import httpx as real_httpx

    # Mock httpx to raise ConnectError for all HTTP calls
    connect_error = real_httpx.ConnectError("mocked connection refused")

    async def _mock_get(self, url, **kwargs):
        raise connect_error

    # Mock subprocess to raise FileNotFoundError (command not found)
    async def _mock_create_subprocess_exec(*args, **kwargs):
        raise FileNotFoundError("mocked: command not found")

    with (
        patch.object(real_httpx.AsyncClient, "get", _mock_get),
        patch(
            "mcp.system_status.asyncio.create_subprocess_exec",
            side_effect=_mock_create_subprocess_exec,
        ),
    ):
        result = await _system_status_impl()

    # Verify top-level keys
    assert "llm" in result
    assert "qdrant" in result
    assert "docker" in result
    assert "gpu" in result
    assert "timestamp" in result

    # Every component should be unhealthy
    assert result["llm"]["healthy"] is False
    assert result["qdrant"]["healthy"] is False
    assert result["docker"]["healthy"] is False
    assert result["gpu"]["healthy"] is False

    # Every unhealthy component should have an error message
    assert "error" in result["llm"]
    assert "error" in result["qdrant"]
    assert "error" in result["docker"]
    assert "error" in result["gpu"]

    # Timestamp should be an ISO string
    assert "T" in result["timestamp"]


@pytest.mark.asyncio
async def test_system_status_returns_expected_structure_all_healthy():
    """When all services respond successfully, system_status returns
    healthy=True for every component with detail fields populated."""

    # LLM mock
    llm_response = _make_httpx_response(200, {
        "status": "ok",
        "model": "qwen3-30b",
        "n_ctx": 32768,
        "slots_idle": 1,
        "slots_processing": 0,
    })

    # Qdrant mock
    qdrant_response = _make_httpx_response(200, {
        "result": {
            "collections": [
                {"name": "anyloom_docs", "points_count": 42},
                {"name": "test_coll", "points_count": 7},
            ]
        }
    })

    # Docker mock
    docker_lines = "\n".join([
        json.dumps({"Names": "qdrant", "Image": "qdrant/qdrant:latest", "Status": "Up 2 hours", "Ports": "6333->6333"}),
        json.dumps({"Names": "anythingllm", "Image": "mintplexlabs/anythingllm", "Status": "Up 1 hour", "Ports": "3001->3001"}),
    ])
    docker_proc = _make_subprocess_mock(returncode=0, stdout=docker_lines)

    # GPU mock
    gpu_output = "NVIDIA GeForce RTX 5090, 45, 12, 8192, 32768"
    gpu_proc = _make_subprocess_mock(returncode=0, stdout=gpu_output)

    call_count = {"n": 0}

    async def _mock_create_subprocess_exec(*args, **kwargs):
        call_count["n"] += 1
        if args[0] == "docker":
            return docker_proc
        elif args[0] == "nvidia-smi":
            return gpu_proc
        raise FileNotFoundError(f"unknown command: {args[0]}")

    import httpx as real_httpx

    original_get = real_httpx.AsyncClient.get

    async def _mock_get(self, url, **kwargs):
        if "8008" in url:
            return llm_response
        elif "6333" in url:
            return qdrant_response
        raise real_httpx.ConnectError("unexpected URL")

    with (
        patch.object(real_httpx.AsyncClient, "get", _mock_get),
        patch(
            "mcp.system_status.asyncio.create_subprocess_exec",
            side_effect=_mock_create_subprocess_exec,
        ),
    ):
        result = await _system_status_impl()

    # All healthy
    assert result["llm"]["healthy"] is True
    assert result["qdrant"]["healthy"] is True
    assert result["docker"]["healthy"] is True
    assert result["gpu"]["healthy"] is True

    # LLM details
    assert result["llm"]["model"] == "qwen3-30b"
    assert result["llm"]["context_size"] == 32768

    # Qdrant details
    assert result["qdrant"]["collections_count"] == 2
    assert len(result["qdrant"]["collections"]) == 2

    # Docker details
    assert result["docker"]["container_count"] == 2
    assert result["docker"]["containers"][0]["name"] == "qdrant"

    # GPU details
    assert result["gpu"]["gpu_count"] == 1
    assert result["gpu"]["gpus"][0]["name"] == "NVIDIA GeForce RTX 5090"
    assert result["gpu"]["gpus"][0]["temperature_c"] == 45
    assert result["gpu"]["gpus"][0]["memory_total_mib"] == 32768


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PROBE_LLM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_probe_llm_healthy():
    """probe_llm returns healthy=True with details when server responds OK."""
    response = _make_httpx_response(200, {
        "status": "ok",
        "model": "qwen3-30b-a3b",
        "n_ctx": 16384,
        "slots_idle": 2,
        "slots_processing": 0,
    })

    import httpx as real_httpx

    async def _mock_get(self, url, **kwargs):
        return response

    with patch.object(real_httpx.AsyncClient, "get", _mock_get):
        result = await probe_llm()

    assert result["healthy"] is True
    assert result["status"] == "ok"
    assert result["model"] == "qwen3-30b-a3b"
    assert result["context_size"] == 16384
    assert result["slots_idle"] == 2


@pytest.mark.asyncio
async def test_probe_llm_no_slot_available():
    """probe_llm returns healthy=True when status is 'no slot available'
    (server is running but all slots busy)."""
    response = _make_httpx_response(200, {
        "status": "no slot available",
    })

    import httpx as real_httpx

    async def _mock_get(self, url, **kwargs):
        return response

    with patch.object(real_httpx.AsyncClient, "get", _mock_get):
        result = await probe_llm()

    assert result["healthy"] is True
    assert result["status"] == "no slot available"


@pytest.mark.asyncio
async def test_probe_llm_connection_refused():
    """probe_llm returns healthy=False with error when connection is refused."""
    import httpx as real_httpx

    async def _mock_get(self, url, **kwargs):
        raise real_httpx.ConnectError("connection refused")

    with patch.object(real_httpx.AsyncClient, "get", _mock_get):
        result = await probe_llm()

    assert result["healthy"] is False
    assert "error" in result
    assert "connection refused" in result["error"]


@pytest.mark.asyncio
async def test_probe_llm_timeout():
    """probe_llm returns healthy=False when the server takes too long."""
    import httpx as real_httpx

    async def _mock_get(self, url, **kwargs):
        raise real_httpx.TimeoutException("read timed out")

    with patch.object(real_httpx.AsyncClient, "get", _mock_get):
        result = await probe_llm()

    assert result["healthy"] is False
    assert "timeout" in result["error"].lower()


@pytest.mark.asyncio
async def test_probe_llm_http_500():
    """probe_llm returns healthy=False when server returns 500."""
    response = _make_httpx_response(500, {})

    import httpx as real_httpx

    async def _mock_get(self, url, **kwargs):
        return response

    with patch.object(real_httpx.AsyncClient, "get", _mock_get):
        result = await probe_llm()

    assert result["healthy"] is False
    assert "error" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PROBE_QDRANT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_probe_qdrant_healthy():
    """probe_qdrant returns healthy=True with collections list."""
    response = _make_httpx_response(200, {
        "result": {
            "collections": [
                {"name": "anyloom_docs", "points_count": 100},
            ]
        }
    })

    import httpx as real_httpx

    async def _mock_get(self, url, **kwargs):
        return response

    with patch.object(real_httpx.AsyncClient, "get", _mock_get):
        result = await probe_qdrant()

    assert result["healthy"] is True
    assert result["collections_count"] == 1
    assert result["collections"][0]["name"] == "anyloom_docs"
    assert result["collections"][0]["points_count"] == 100


@pytest.mark.asyncio
async def test_probe_qdrant_empty_collections():
    """probe_qdrant returns healthy=True even with no collections."""
    response = _make_httpx_response(200, {
        "result": {"collections": []}
    })

    import httpx as real_httpx

    async def _mock_get(self, url, **kwargs):
        return response

    with patch.object(real_httpx.AsyncClient, "get", _mock_get):
        result = await probe_qdrant()

    assert result["healthy"] is True
    assert result["collections_count"] == 0
    assert result["collections"] == []


@pytest.mark.asyncio
async def test_probe_qdrant_connection_refused():
    """probe_qdrant returns healthy=False when Qdrant is unreachable."""
    import httpx as real_httpx

    async def _mock_get(self, url, **kwargs):
        raise real_httpx.ConnectError("connection refused")

    with patch.object(real_httpx.AsyncClient, "get", _mock_get):
        result = await probe_qdrant()

    assert result["healthy"] is False
    assert "connection refused" in result["error"]


@pytest.mark.asyncio
async def test_probe_qdrant_timeout():
    """probe_qdrant returns healthy=False on timeout."""
    import httpx as real_httpx

    async def _mock_get(self, url, **kwargs):
        raise real_httpx.TimeoutException("timed out")

    with patch.object(real_httpx.AsyncClient, "get", _mock_get):
        result = await probe_qdrant()

    assert result["healthy"] is False
    assert "timeout" in result["error"].lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PROBE_DOCKER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_probe_docker_healthy():
    """probe_docker parses docker ps JSON output correctly."""
    docker_output = "\n".join([
        json.dumps({
            "Names": "qdrant-server",
            "Image": "qdrant/qdrant:v1.12.0",
            "Status": "Up 3 hours",
            "Ports": "0.0.0.0:6333->6333/tcp",
        }),
        json.dumps({
            "Names": "anythingllm",
            "Image": "mintplexlabs/anythingllm:latest",
            "Status": "Up 1 hour",
            "Ports": "0.0.0.0:3001->3001/tcp",
        }),
    ])
    proc = _make_subprocess_mock(returncode=0, stdout=docker_output)

    async def _mock_create(*args, **kwargs):
        return proc

    with patch(
        "mcp.system_status.asyncio.create_subprocess_exec",
        side_effect=_mock_create,
    ):
        result = await probe_docker()

    assert result["healthy"] is True
    assert result["container_count"] == 2
    assert result["containers"][0]["name"] == "qdrant-server"
    assert result["containers"][1]["name"] == "anythingllm"


@pytest.mark.asyncio
async def test_probe_docker_no_containers():
    """probe_docker returns healthy=True with empty list when no containers run."""
    proc = _make_subprocess_mock(returncode=0, stdout="")

    async def _mock_create(*args, **kwargs):
        return proc

    with patch(
        "mcp.system_status.asyncio.create_subprocess_exec",
        side_effect=_mock_create,
    ):
        result = await probe_docker()

    assert result["healthy"] is True
    assert result["container_count"] == 0


@pytest.mark.asyncio
async def test_probe_docker_command_not_found():
    """probe_docker returns healthy=False when docker is not installed."""
    async def _mock_create(*args, **kwargs):
        raise FileNotFoundError("docker not found")

    with patch(
        "mcp.system_status.asyncio.create_subprocess_exec",
        side_effect=_mock_create,
    ):
        result = await probe_docker()

    assert result["healthy"] is False
    assert "not found" in result["error"]


@pytest.mark.asyncio
async def test_probe_docker_nonzero_exit():
    """probe_docker returns healthy=False when docker ps fails."""
    proc = _make_subprocess_mock(returncode=1, stderr="Cannot connect to Docker daemon")

    async def _mock_create(*args, **kwargs):
        return proc

    with patch(
        "mcp.system_status.asyncio.create_subprocess_exec",
        side_effect=_mock_create,
    ):
        result = await probe_docker()

    assert result["healthy"] is False
    assert "Cannot connect" in result["error"]


@pytest.mark.asyncio
async def test_probe_docker_timeout():
    """probe_docker returns healthy=False when docker ps hangs."""
    async def _mock_create(*args, **kwargs):
        raise asyncio.TimeoutError()

    with patch(
        "mcp.system_status.asyncio.create_subprocess_exec",
        side_effect=_mock_create,
    ):
        result = await probe_docker()

    assert result["healthy"] is False
    assert "timeout" in result["error"].lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PROBE_GPU
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_probe_gpu_healthy():
    """probe_gpu parses nvidia-smi CSV output correctly."""
    gpu_output = "NVIDIA GeForce RTX 5090, 52, 35, 12288, 32768"
    proc = _make_subprocess_mock(returncode=0, stdout=gpu_output)

    async def _mock_create(*args, **kwargs):
        return proc

    with patch(
        "mcp.system_status.asyncio.create_subprocess_exec",
        side_effect=_mock_create,
    ):
        result = await probe_gpu()

    assert result["healthy"] is True
    assert result["gpu_count"] == 1

    gpu = result["gpus"][0]
    assert gpu["name"] == "NVIDIA GeForce RTX 5090"
    assert gpu["temperature_c"] == 52
    assert gpu["utilization_pct"] == 35
    assert gpu["memory_used_mib"] == 12288
    assert gpu["memory_total_mib"] == 32768
    assert gpu["memory_free_mib"] == 32768 - 12288


@pytest.mark.asyncio
async def test_probe_gpu_multi_gpu():
    """probe_gpu handles multi-GPU output."""
    gpu_output = (
        "NVIDIA GeForce RTX 5090, 45, 10, 4096, 32768\n"
        "NVIDIA GeForce RTX 4090, 60, 80, 20000, 24576"
    )
    proc = _make_subprocess_mock(returncode=0, stdout=gpu_output)

    async def _mock_create(*args, **kwargs):
        return proc

    with patch(
        "mcp.system_status.asyncio.create_subprocess_exec",
        side_effect=_mock_create,
    ):
        result = await probe_gpu()

    assert result["healthy"] is True
    assert result["gpu_count"] == 2
    assert result["gpus"][0]["name"] == "NVIDIA GeForce RTX 5090"
    assert result["gpus"][1]["name"] == "NVIDIA GeForce RTX 4090"


@pytest.mark.asyncio
async def test_probe_gpu_command_not_found():
    """probe_gpu returns healthy=False when nvidia-smi is not installed."""
    async def _mock_create(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    with patch(
        "mcp.system_status.asyncio.create_subprocess_exec",
        side_effect=_mock_create,
    ):
        result = await probe_gpu()

    assert result["healthy"] is False
    assert "not found" in result["error"]


@pytest.mark.asyncio
async def test_probe_gpu_nonzero_exit():
    """probe_gpu returns healthy=False when nvidia-smi fails."""
    proc = _make_subprocess_mock(
        returncode=1, stderr="NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver"
    )

    async def _mock_create(*args, **kwargs):
        return proc

    with patch(
        "mcp.system_status.asyncio.create_subprocess_exec",
        side_effect=_mock_create,
    ):
        result = await probe_gpu()

    assert result["healthy"] is False
    assert "nvidia-smi failed" in result["error"]


@pytest.mark.asyncio
async def test_probe_gpu_timeout():
    """probe_gpu returns healthy=False when nvidia-smi hangs."""
    async def _mock_create(*args, **kwargs):
        raise asyncio.TimeoutError()

    with patch(
        "mcp.system_status.asyncio.create_subprocess_exec",
        side_effect=_mock_create,
    ):
        result = await probe_gpu()

    assert result["healthy"] is False
    assert "timeout" in result["error"].lower()


@pytest.mark.asyncio
async def test_probe_gpu_empty_output():
    """probe_gpu returns healthy=False when nvidia-smi gives empty output."""
    proc = _make_subprocess_mock(returncode=0, stdout="")

    async def _mock_create(*args, **kwargs):
        return proc

    with patch(
        "mcp.system_status.asyncio.create_subprocess_exec",
        side_effect=_mock_create,
    ):
        result = await probe_gpu()

    assert result["healthy"] is False
    assert "empty" in result["error"].lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MIXED SCENARIOS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_system_status_partial_failure():
    """When some services are up and some are down, each component
    independently reports its own status."""

    # LLM is healthy
    llm_response = _make_httpx_response(200, {"status": "ok"})

    # Qdrant is down
    import httpx as real_httpx

    async def _mock_get(self, url, **kwargs):
        if "8008" in url:
            return llm_response
        elif "6333" in url:
            raise real_httpx.ConnectError("qdrant down")
        raise real_httpx.ConnectError("unknown")

    # Docker works, GPU fails
    docker_output = json.dumps({"Names": "qdrant", "Image": "qdrant/qdrant", "Status": "Up", "Ports": ""})
    docker_proc = _make_subprocess_mock(returncode=0, stdout=docker_output)

    async def _mock_create(*args, **kwargs):
        if args[0] == "docker":
            return docker_proc
        elif args[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi not installed")
        raise FileNotFoundError("unknown")

    with (
        patch.object(real_httpx.AsyncClient, "get", _mock_get),
        patch(
            "mcp.system_status.asyncio.create_subprocess_exec",
            side_effect=_mock_create,
        ),
    ):
        result = await _system_status_impl()

    # LLM up, Qdrant down, Docker up, GPU down
    assert result["llm"]["healthy"] is True
    assert result["qdrant"]["healthy"] is False
    assert result["docker"]["healthy"] is True
    assert result["gpu"]["healthy"] is False

    # Verify error messages exist for failed components
    assert "error" in result["qdrant"]
    assert "error" in result["gpu"]

    # Verify no error key on healthy components
    assert "error" not in result["llm"]
    assert "error" not in result["docker"]
