"""
Health Check Module
===================

Probes all AnyLoom stack components (LLM server, Qdrant, AnythingLLM, GPU)
and returns structured results via Pydantic models.

Called by the orchestrator before swarm runs to catch failures early.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from dytopo.models import HealthStatus, StackHealth


class HealthChecker:
    """Probes all stack components and returns structured health results."""

    def __init__(
        self,
        llm_url: str = "http://localhost:8008",
        qdrant_url: str = "http://localhost:6333",
        anythingllm_url: str = "http://localhost:3001",
        timeout: float = 5.0,
    ):
        self.llm_url = llm_url.rstrip("/")
        self.qdrant_url = qdrant_url.rstrip("/")
        self.anythingllm_url = anythingllm_url.rstrip("/")
        self.timeout = timeout

    async def check_all(self) -> StackHealth:
        """Probe all components in parallel, return aggregate health."""
        results = await asyncio.gather(
            self.check_llm(),
            self.check_qdrant(),
            self.check_anythingllm(),
            self.check_gpu(),
            return_exceptions=True,
        )

        components: list[HealthStatus] = []
        for result in results:
            if isinstance(result, BaseException):
                # Should not happen since each check_* catches its own errors,
                # but handle defensive for truly unexpected failures.
                components.append(
                    HealthStatus(
                        component="unknown",
                        healthy=False,
                        error=f"Unexpected: {result}",
                    )
                )
            else:
                components.append(result)

        all_healthy = all(c.healthy for c in components)
        return StackHealth(
            components=components,
            all_healthy=all_healthy,
        )

    async def check_llm(self) -> HealthStatus:
        """Probe llama.cpp /health endpoint."""
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{self.llm_url}/health")
                resp.raise_for_status()
                data = resp.json()
            latency = (time.perf_counter() - start) * 1000
            return HealthStatus(
                component="llm",
                healthy=True,
                latency_ms=latency,
                details=data if isinstance(data, dict) else {"response": data},
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return HealthStatus(
                component="llm",
                healthy=False,
                latency_ms=latency,
                error=str(e),
            )

    async def check_qdrant(self) -> HealthStatus:
        """Probe Qdrant /collections endpoint."""
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{self.qdrant_url}/collections")
                resp.raise_for_status()
                data = resp.json()
            latency = (time.perf_counter() - start) * 1000
            return HealthStatus(
                component="qdrant",
                healthy=True,
                latency_ms=latency,
                details=data if isinstance(data, dict) else {"response": data},
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return HealthStatus(
                component="qdrant",
                healthy=False,
                latency_ms=latency,
                error=str(e),
            )

    async def check_anythingllm(self) -> HealthStatus:
        """Probe AnythingLLM /api/ping endpoint."""
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{self.anythingllm_url}/api/ping")
                resp.raise_for_status()
                data = resp.json()
            latency = (time.perf_counter() - start) * 1000
            return HealthStatus(
                component="anythingllm",
                healthy=True,
                latency_ms=latency,
                details=data if isinstance(data, dict) else {"response": data},
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return HealthStatus(
                component="anythingllm",
                healthy=False,
                latency_ms=latency,
                error=str(e),
            )

    async def check_gpu(self) -> HealthStatus:
        """Query nvidia-smi for GPU status."""
        start = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
            latency = (time.perf_counter() - start) * 1000

            if proc.returncode != 0:
                return HealthStatus(
                    component="gpu",
                    healthy=False,
                    latency_ms=latency,
                    error=f"nvidia-smi exited with code {proc.returncode}: {stderr.decode().strip()}",
                )

            output = stdout.decode().strip()
            if not output:
                return HealthStatus(
                    component="gpu",
                    healthy=False,
                    latency_ms=latency,
                    error="nvidia-smi returned empty output",
                )

            # Parse CSV: name, temperature, utilization%, memory_used, memory_total
            gpus: list[dict[str, Any]] = []
            for line in output.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append(
                        {
                            "name": parts[0],
                            "temperature_c": float(parts[1]),
                            "utilization_pct": float(parts[2]),
                            "memory_used_mb": float(parts[3]),
                            "memory_total_mb": float(parts[4]),
                        }
                    )

            if not gpus:
                return HealthStatus(
                    component="gpu",
                    healthy=False,
                    latency_ms=latency,
                    error=f"Could not parse nvidia-smi output: {output}",
                )

            return HealthStatus(
                component="gpu",
                healthy=True,
                latency_ms=latency,
                details={"gpus": gpus, "gpu_count": len(gpus)},
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return HealthStatus(
                component="gpu",
                healthy=False,
                latency_ms=latency,
                error=str(e),
            )


async def preflight_check(
    llm_url: str = "http://localhost:8008",
    qdrant_url: str = "http://localhost:6333",
) -> StackHealth:
    """Quick check before a swarm run. Returns StackHealth."""
    checker = HealthChecker(llm_url=llm_url, qdrant_url=qdrant_url)
    return await checker.check_all()
