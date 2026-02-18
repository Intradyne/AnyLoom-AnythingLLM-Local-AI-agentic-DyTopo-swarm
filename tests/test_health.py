"""Test suite for the health check module.

Tests all HealthChecker probes with mocked HTTP responses and subprocess
output, covering healthy, unhealthy, timeout, and aggregation scenarios.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dytopo.health.checker import HealthChecker, preflight_check
from dytopo.models import HealthStatus, StackHealth


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def checker():
    """Standard HealthChecker with default URLs."""
    return HealthChecker(
        llm_url="http://localhost:8008",
        qdrant_url="http://localhost:6333",
        anythingllm_url="http://localhost:3001",
        timeout=5.0,
    )


@pytest.fixture
def short_timeout_checker():
    """HealthChecker with a very short timeout for timeout tests."""
    return HealthChecker(timeout=0.001)


# ---------------------------------------------------------------------------
# Helper to build mock httpx responses
# ---------------------------------------------------------------------------

def _mock_response(json_data: dict, status_code: int = 200):
    """Build a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# check_llm tests
# ---------------------------------------------------------------------------

class TestCheckLLM:
    """Tests for the LLM (llama.cpp) health probe."""

    @pytest.mark.asyncio
    async def test_healthy(self, checker):
        """LLM endpoint returns 200 with valid JSON."""
        mock_resp = _mock_response({"status": "ok"})

        with patch("dytopo.health.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await checker.check_llm()

        assert isinstance(result, HealthStatus)
        assert result.component == "llm"
        assert result.healthy is True
        assert result.latency_ms > 0
        assert result.details == {"status": "ok"}
        assert result.error is None

    @pytest.mark.asyncio
    async def test_unhealthy_http_error(self, checker):
        """LLM endpoint returns non-200 status."""
        mock_resp = _mock_response({"error": "not ready"}, status_code=503)

        with patch("dytopo.health.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await checker.check_llm()

        assert result.component == "llm"
        assert result.healthy is False
        assert result.error is not None
        assert "503" in result.error

    @pytest.mark.asyncio
    async def test_unhealthy_connection_refused(self, checker):
        """LLM endpoint is unreachable."""
        with patch("dytopo.health.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await checker.check_llm()

        assert result.component == "llm"
        assert result.healthy is False
        assert "Connection refused" in result.error

    @pytest.mark.asyncio
    async def test_never_raises(self, checker):
        """check_llm must never raise, even on unexpected errors."""
        with patch("dytopo.health.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = RuntimeError("Something wild happened")
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await checker.check_llm()

        assert result.healthy is False
        assert "Something wild happened" in result.error


# ---------------------------------------------------------------------------
# check_qdrant tests
# ---------------------------------------------------------------------------

class TestCheckQdrant:
    """Tests for the Qdrant health probe."""

    @pytest.mark.asyncio
    async def test_healthy(self, checker):
        """Qdrant /collections returns 200."""
        collections_data = {"result": {"collections": []}, "status": "ok"}
        mock_resp = _mock_response(collections_data)

        with patch("dytopo.health.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await checker.check_qdrant()

        assert result.component == "qdrant"
        assert result.healthy is True
        assert result.details["status"] == "ok"

    @pytest.mark.asyncio
    async def test_unhealthy(self, checker):
        """Qdrant returns 500."""
        mock_resp = _mock_response({"status": "error"}, status_code=500)

        with patch("dytopo.health.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await checker.check_qdrant()

        assert result.component == "qdrant"
        assert result.healthy is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_connection_refused(self, checker):
        """Qdrant is unreachable."""
        with patch("dytopo.health.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await checker.check_qdrant()

        assert result.component == "qdrant"
        assert result.healthy is False
        assert "Connection refused" in result.error


# ---------------------------------------------------------------------------
# check_anythingllm tests
# ---------------------------------------------------------------------------

class TestCheckAnythingLLM:
    """Tests for the AnythingLLM health probe."""

    @pytest.mark.asyncio
    async def test_healthy(self, checker):
        """AnythingLLM /api/ping returns 200."""
        mock_resp = _mock_response({"online": True})

        with patch("dytopo.health.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await checker.check_anythingllm()

        assert result.component == "anythingllm"
        assert result.healthy is True
        assert result.details == {"online": True}

    @pytest.mark.asyncio
    async def test_unhealthy(self, checker):
        """AnythingLLM returns a server error."""
        mock_resp = _mock_response({}, status_code=502)

        with patch("dytopo.health.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await checker.check_anythingllm()

        assert result.component == "anythingllm"
        assert result.healthy is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_connection_refused(self, checker):
        """AnythingLLM is unreachable."""
        with patch("dytopo.health.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await checker.check_anythingllm()

        assert result.component == "anythingllm"
        assert result.healthy is False


# ---------------------------------------------------------------------------
# check_gpu tests
# ---------------------------------------------------------------------------

class TestCheckGPU:
    """Tests for the GPU (nvidia-smi) health probe."""

    @pytest.mark.asyncio
    async def test_healthy_single_gpu(self, checker):
        """nvidia-smi returns valid CSV for one GPU."""
        csv_output = "NVIDIA GeForce RTX 4090, 45, 32, 4096, 24576\n"

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (csv_output.encode(), b"")
        mock_proc.returncode = 0

        with patch("dytopo.health.checker.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await checker.check_gpu()

        assert result.component == "gpu"
        assert result.healthy is True
        assert result.details["gpu_count"] == 1
        gpu = result.details["gpus"][0]
        assert gpu["name"] == "NVIDIA GeForce RTX 4090"
        assert gpu["temperature_c"] == 45.0
        assert gpu["utilization_pct"] == 32.0
        assert gpu["memory_used_mb"] == 4096.0
        assert gpu["memory_total_mb"] == 24576.0

    @pytest.mark.asyncio
    async def test_healthy_multi_gpu(self, checker):
        """nvidia-smi returns valid CSV for multiple GPUs."""
        csv_output = (
            "NVIDIA GeForce RTX 4090, 45, 32, 4096, 24576\n"
            "NVIDIA GeForce RTX 4090, 50, 60, 8000, 24576\n"
        )

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (csv_output.encode(), b"")
        mock_proc.returncode = 0

        with patch("dytopo.health.checker.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await checker.check_gpu()

        assert result.healthy is True
        assert result.details["gpu_count"] == 2
        assert result.details["gpus"][1]["temperature_c"] == 50.0
        assert result.details["gpus"][1]["utilization_pct"] == 60.0

    @pytest.mark.asyncio
    async def test_nvidia_smi_not_found(self, checker):
        """nvidia-smi is not installed."""
        with patch(
            "dytopo.health.checker.asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("nvidia-smi not found"),
        ):
            result = await checker.check_gpu()

        assert result.component == "gpu"
        assert result.healthy is False
        assert "nvidia-smi not found" in result.error

    @pytest.mark.asyncio
    async def test_nvidia_smi_nonzero_exit(self, checker):
        """nvidia-smi exits with non-zero code."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"NVIDIA-SMI has failed")
        mock_proc.returncode = 1

        with patch("dytopo.health.checker.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await checker.check_gpu()

        assert result.healthy is False
        assert "exited with code 1" in result.error
        assert "NVIDIA-SMI has failed" in result.error

    @pytest.mark.asyncio
    async def test_nvidia_smi_empty_output(self, checker):
        """nvidia-smi returns empty output."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0

        with patch("dytopo.health.checker.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await checker.check_gpu()

        assert result.healthy is False
        assert "empty output" in result.error

    @pytest.mark.asyncio
    async def test_nvidia_smi_unparseable_output(self, checker):
        """nvidia-smi returns data that cannot be parsed into 5 fields."""
        csv_output = "bad data\n"

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (csv_output.encode(), b"")
        mock_proc.returncode = 0

        with patch("dytopo.health.checker.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await checker.check_gpu()

        assert result.healthy is False
        assert "Could not parse" in result.error

    @pytest.mark.asyncio
    async def test_gpu_never_raises(self, checker):
        """check_gpu must never raise, even on unexpected errors."""
        with patch(
            "dytopo.health.checker.asyncio.create_subprocess_exec",
            side_effect=RuntimeError("Unexpected kernel panic"),
        ):
            result = await checker.check_gpu()

        assert result.healthy is False
        assert "Unexpected kernel panic" in result.error


# ---------------------------------------------------------------------------
# Timeout handling tests
# ---------------------------------------------------------------------------

class TestTimeoutHandling:
    """Tests that each probe respects its timeout."""

    @pytest.mark.asyncio
    async def test_llm_timeout(self):
        """LLM probe times out gracefully."""
        checker = HealthChecker(timeout=0.001)

        async def slow_get(*args, **kwargs):
            await asyncio.sleep(10)

        with patch("dytopo.health.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ReadTimeout("timed out")
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await checker.check_llm()

        assert result.healthy is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_qdrant_timeout(self):
        """Qdrant probe times out gracefully."""
        checker = HealthChecker(timeout=0.001)

        with patch("dytopo.health.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ReadTimeout("timed out")
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await checker.check_qdrant()

        assert result.healthy is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_anythingllm_timeout(self):
        """AnythingLLM probe times out gracefully."""
        checker = HealthChecker(timeout=0.001)

        with patch("dytopo.health.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ReadTimeout("timed out")
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await checker.check_anythingllm()

        assert result.healthy is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_gpu_timeout(self):
        """GPU probe times out gracefully."""
        checker = HealthChecker(timeout=0.001)

        mock_proc = AsyncMock()
        mock_proc.communicate.side_effect = asyncio.TimeoutError()

        with patch("dytopo.health.checker.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await checker.check_gpu()

        assert result.component == "gpu"
        assert result.healthy is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# check_all aggregation tests
# ---------------------------------------------------------------------------

class TestCheckAll:
    """Tests for the check_all aggregation method."""

    @pytest.mark.asyncio
    async def test_all_healthy(self, checker):
        """All components healthy yields all_healthy=True."""
        healthy_llm = HealthStatus(component="llm", healthy=True, latency_ms=10.0)
        healthy_qdrant = HealthStatus(component="qdrant", healthy=True, latency_ms=5.0)
        healthy_allm = HealthStatus(component="anythingllm", healthy=True, latency_ms=8.0)
        healthy_gpu = HealthStatus(component="gpu", healthy=True, latency_ms=20.0)

        with (
            patch.object(checker, "check_llm", return_value=healthy_llm),
            patch.object(checker, "check_qdrant", return_value=healthy_qdrant),
            patch.object(checker, "check_anythingllm", return_value=healthy_allm),
            patch.object(checker, "check_gpu", return_value=healthy_gpu),
        ):
            result = await checker.check_all()

        assert isinstance(result, StackHealth)
        assert result.all_healthy is True
        assert len(result.components) == 4
        assert all(c.healthy for c in result.components)

    @pytest.mark.asyncio
    async def test_one_unhealthy(self, checker):
        """One unhealthy component makes all_healthy=False."""
        healthy_llm = HealthStatus(component="llm", healthy=True, latency_ms=10.0)
        unhealthy_qdrant = HealthStatus(
            component="qdrant", healthy=False, error="Connection refused"
        )
        healthy_allm = HealthStatus(component="anythingllm", healthy=True, latency_ms=8.0)
        healthy_gpu = HealthStatus(component="gpu", healthy=True, latency_ms=20.0)

        with (
            patch.object(checker, "check_llm", return_value=healthy_llm),
            patch.object(checker, "check_qdrant", return_value=unhealthy_qdrant),
            patch.object(checker, "check_anythingllm", return_value=healthy_allm),
            patch.object(checker, "check_gpu", return_value=healthy_gpu),
        ):
            result = await checker.check_all()

        assert result.all_healthy is False
        assert len(result.components) == 4
        qdrant_status = [c for c in result.components if c.component == "qdrant"][0]
        assert qdrant_status.healthy is False

    @pytest.mark.asyncio
    async def test_all_unhealthy(self, checker):
        """All components unhealthy yields all_healthy=False."""
        unhealthy_llm = HealthStatus(component="llm", healthy=False, error="down")
        unhealthy_qdrant = HealthStatus(component="qdrant", healthy=False, error="down")
        unhealthy_allm = HealthStatus(component="anythingllm", healthy=False, error="down")
        unhealthy_gpu = HealthStatus(component="gpu", healthy=False, error="no gpu")

        with (
            patch.object(checker, "check_llm", return_value=unhealthy_llm),
            patch.object(checker, "check_qdrant", return_value=unhealthy_qdrant),
            patch.object(checker, "check_anythingllm", return_value=unhealthy_allm),
            patch.object(checker, "check_gpu", return_value=unhealthy_gpu),
        ):
            result = await checker.check_all()

        assert result.all_healthy is False
        assert len(result.components) == 4
        assert all(not c.healthy for c in result.components)

    @pytest.mark.asyncio
    async def test_runs_probes_in_parallel(self, checker):
        """Verify check_all uses asyncio.gather to run probes concurrently."""
        call_order = []

        async def slow_llm():
            call_order.append("llm_start")
            await asyncio.sleep(0.05)
            call_order.append("llm_end")
            return HealthStatus(component="llm", healthy=True, latency_ms=50.0)

        async def slow_qdrant():
            call_order.append("qdrant_start")
            await asyncio.sleep(0.05)
            call_order.append("qdrant_end")
            return HealthStatus(component="qdrant", healthy=True, latency_ms=50.0)

        async def slow_allm():
            call_order.append("allm_start")
            await asyncio.sleep(0.05)
            call_order.append("allm_end")
            return HealthStatus(component="anythingllm", healthy=True, latency_ms=50.0)

        async def slow_gpu():
            call_order.append("gpu_start")
            await asyncio.sleep(0.05)
            call_order.append("gpu_end")
            return HealthStatus(component="gpu", healthy=True, latency_ms=50.0)

        with (
            patch.object(checker, "check_llm", side_effect=slow_llm),
            patch.object(checker, "check_qdrant", side_effect=slow_qdrant),
            patch.object(checker, "check_anythingllm", side_effect=slow_allm),
            patch.object(checker, "check_gpu", side_effect=slow_gpu),
        ):
            result = await checker.check_all()

        assert result.all_healthy is True
        # All probes should start before any finishes (parallel execution)
        starts = [i for i, v in enumerate(call_order) if v.endswith("_start")]
        ends = [i for i, v in enumerate(call_order) if v.endswith("_end")]
        assert max(starts) < min(ends), (
            f"Probes were not run in parallel. Order: {call_order}"
        )

    @pytest.mark.asyncio
    async def test_exception_in_probe_handled(self, checker):
        """An unexpected exception from a probe does not crash check_all."""
        healthy_llm = HealthStatus(component="llm", healthy=True, latency_ms=10.0)
        healthy_allm = HealthStatus(component="anythingllm", healthy=True, latency_ms=8.0)
        healthy_gpu = HealthStatus(component="gpu", healthy=True, latency_ms=20.0)

        async def exploding_qdrant():
            raise RuntimeError("kaboom")

        with (
            patch.object(checker, "check_llm", return_value=healthy_llm),
            patch.object(checker, "check_qdrant", side_effect=exploding_qdrant),
            patch.object(checker, "check_anythingllm", return_value=healthy_allm),
            patch.object(checker, "check_gpu", return_value=healthy_gpu),
        ):
            result = await checker.check_all()

        assert result.all_healthy is False
        assert len(result.components) == 4
        # The exploding component should be captured as unhealthy
        failed = [c for c in result.components if not c.healthy]
        assert len(failed) == 1
        assert "kaboom" in failed[0].error


# ---------------------------------------------------------------------------
# preflight_check convenience function
# ---------------------------------------------------------------------------

class TestPreflightCheck:
    """Tests for the preflight_check convenience function."""

    @pytest.mark.asyncio
    async def test_preflight_returns_stack_health(self):
        """preflight_check returns a StackHealth object."""
        healthy = HealthStatus(component="llm", healthy=True, latency_ms=1.0)

        with patch.object(HealthChecker, "check_all") as mock_check_all:
            mock_check_all.return_value = StackHealth(
                components=[healthy], all_healthy=True
            )
            result = await preflight_check()

        assert isinstance(result, StackHealth)
        assert result.all_healthy is True

    @pytest.mark.asyncio
    async def test_preflight_passes_urls(self):
        """preflight_check passes custom URLs through to HealthChecker."""
        with patch.object(HealthChecker, "__init__", return_value=None) as mock_init:
            with patch.object(HealthChecker, "check_all", return_value=StackHealth()):
                await preflight_check(
                    llm_url="http://custom:9999",
                    qdrant_url="http://custom:7777",
                )

            # Verify custom URLs were passed
            call_kwargs = mock_init.call_args
            assert call_kwargs[1]["llm_url"] == "http://custom:9999"
            assert call_kwargs[1]["qdrant_url"] == "http://custom:7777"


# ---------------------------------------------------------------------------
# HealthStatus model field tests
# ---------------------------------------------------------------------------

class TestHealthStatusModel:
    """Tests validating the HealthStatus model population."""

    @pytest.mark.asyncio
    async def test_latency_is_measured(self, checker):
        """Verify latency_ms is populated with a positive value."""
        mock_resp = _mock_response({"status": "ok"})

        with patch("dytopo.health.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await checker.check_llm()

        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_checked_at_is_populated(self, checker):
        """Verify checked_at gets a valid timestamp."""
        mock_resp = _mock_response({"status": "ok"})

        with patch("dytopo.health.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await checker.check_llm()

        assert result.checked_at > 0

    @pytest.mark.asyncio
    async def test_error_latency_is_measured(self, checker):
        """Even failed probes should have latency measured."""
        with patch("dytopo.health.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("refused")
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await checker.check_llm()

        assert result.latency_ms >= 0
