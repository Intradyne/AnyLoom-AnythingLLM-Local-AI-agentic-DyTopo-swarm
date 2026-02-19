"""
AnyLoom Health Monitor Sidecar
==============================

Standalone process for deterministic health monitoring and auto-recovery.
No LLM inference -- runs outside the agent stack.

Usage:
    python scripts/health_monitor.py                  # Uses dytopo_config.yaml
    CHECK_INTERVAL=5 python scripts/health_monitor.py  # Override via env

Logs JSONL to ~/anyloom-logs/health.jsonl
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Add project root to path so dytopo package is importable when running
# this script directly (e.g. ``python scripts/health_monitor.py``).
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dytopo.config import load_config
from dytopo.health.checker import HealthChecker

# ---------------------------------------------------------------------------
# Container mapping -- "gpu" is hardware-only and has no Docker container.
# ---------------------------------------------------------------------------
COMPONENT_TO_CONTAINER: dict[str, str] = {
    "llm": "anyloom-llm",
    "qdrant": "anyloom-qdrant",
    "anythingllm": "anyloom-anythingllm",
    "embedding": "anyloom-embedding",
}

# Module-level logger
logger = logging.getLogger("anyloom.health-monitor")


# ---- helpers ---------------------------------------------------------------

class CrashTracker:
    """Track restart attempts per component within a sliding crash window."""

    def __init__(self, max_attempts: int, window_minutes: float):
        self.max_attempts = max_attempts
        self.window_seconds = window_minutes * 60
        self.restart_times: dict[str, list[float]] = defaultdict(list)

    def _prune(self, component: str) -> None:
        """Remove restart timestamps that have fallen outside the window."""
        cutoff = time.time() - self.window_seconds
        self.restart_times[component] = [
            t for t in self.restart_times[component] if t >= cutoff
        ]

    def can_restart(self, component: str) -> bool:
        """Return *True* if the component has not exceeded its restart budget."""
        self._prune(component)
        return len(self.restart_times[component]) < self.max_attempts

    def record_restart(self, component: str) -> None:
        self.restart_times[component].append(time.time())

    def attempts_in_window(self, component: str) -> int:
        self._prune(component)
        return len(self.restart_times[component])


class AlertCooldown:
    """Prevent duplicate alerts within a cooldown period."""

    def __init__(self, cooldown_minutes: float):
        self.cooldown_seconds = cooldown_minutes * 60
        self.last_alert: dict[str, float] = {}

    def can_alert(self, key: str) -> bool:
        now = time.time()
        last = self.last_alert.get(key, 0)
        return now - last >= self.cooldown_seconds

    def record_alert(self, key: str) -> None:
        self.last_alert[key] = time.time()


# ---- core monitor ----------------------------------------------------------

class HealthMonitor:
    """Periodically probes AnyLoom services, logs results as JSONL, and
    auto-restarts failed Docker containers with crash-window protection."""

    def __init__(self, config: dict):
        hm_config = config.get("health_monitor", {})

        # Allow env-var overrides for quick iteration in CI / dev.
        self.check_interval = int(
            os.environ.get("CHECK_INTERVAL", hm_config.get("check_interval_seconds", 30))
        )
        self.max_restart_attempts = int(
            os.environ.get("MAX_RESTART_ATTEMPTS", hm_config.get("max_restart_attempts", 3))
        )
        crash_window = float(hm_config.get("crash_window_minutes", 15))
        alert_cooldown = float(hm_config.get("alert_cooldown_minutes", 30))

        log_dir = os.path.expanduser(hm_config.get("log_dir", "~/anyloom-logs"))
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_dir) / "health.jsonl"

        self.checker = HealthChecker()
        self.crash_tracker = CrashTracker(self.max_restart_attempts, crash_window)
        self.alert_cooldown = AlertCooldown(alert_cooldown)
        self._running = True

    # -- logging -------------------------------------------------------------

    def _log_jsonl(self, record: dict) -> None:
        """Append a single JSONL record with ISO and epoch timestamps."""
        record["timestamp"] = time.time()
        record["iso_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # -- container restart ---------------------------------------------------

    async def _restart_container(self, component: str) -> bool:
        """Attempt to restart the Docker container mapped to *component*.

        Returns *True* on success, *False* on failure or when the crash-window
        budget has been exhausted.
        """
        container = COMPONENT_TO_CONTAINER.get(component)
        if not container:
            return False

        if not self.crash_tracker.can_restart(component):
            attempts = self.crash_tracker.attempts_in_window(component)
            if self.alert_cooldown.can_alert(f"max_restarts_{component}"):
                logger.error(
                    "ALERT: %s has been restarted %d times in the crash window "
                    "-- stopping auto-restart",
                    component,
                    attempts,
                )
                self._log_jsonl({
                    "event": "restart_limit_reached",
                    "component": component,
                    "container": container,
                    "attempts": attempts,
                })
                self.alert_cooldown.record_alert(f"max_restarts_{component}")
            return False

        logger.warning("Restarting container: %s", container)
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "restart", container,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

            success = proc.returncode == 0
            self.crash_tracker.record_restart(component)

            self._log_jsonl({
                "event": "container_restart",
                "component": component,
                "container": container,
                "success": success,
                "returncode": proc.returncode,
                "stdout": stdout.decode().strip() if stdout else "",
                "stderr": stderr.decode().strip() if stderr else "",
            })

            if success:
                logger.info("Restarted %s successfully", container)
            else:
                logger.error(
                    "Failed to restart %s: %s",
                    container,
                    stderr.decode().strip() if stderr else "(no stderr)",
                )

            return success

        except asyncio.TimeoutError:
            logger.error("Restart timed out for %s", container)
            self._log_jsonl({
                "event": "restart_error",
                "component": component,
                "container": container,
                "error": "docker restart timed out after 60s",
            })
            return False

        except Exception as e:
            logger.error("Restart failed for %s: %s", container, e)
            self._log_jsonl({
                "event": "restart_error",
                "component": component,
                "container": container,
                "error": str(e),
            })
            return False

    # -- embedding probe (not in HealthChecker) ------------------------------

    async def _check_embedding(self) -> dict:
        """Direct HTTP probe for the embedding container on port 8009."""
        import httpx

        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("http://localhost:8009/health")
                resp.raise_for_status()
            latency = (time.perf_counter() - start) * 1000
            return {
                "component": "embedding",
                "healthy": True,
                "latency_ms": round(latency, 1),
            }
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return {
                "component": "embedding",
                "healthy": False,
                "latency_ms": round(latency, 1),
                "error": str(e),
            }

    # -- main loop -----------------------------------------------------------

    async def run(self) -> None:
        """Main monitoring loop -- runs until ``stop()`` is called."""
        logger.info(
            "Health Monitor started (interval=%ds, max_restarts=%d)",
            self.check_interval,
            self.max_restart_attempts,
        )
        logger.info("Logging to: %s", self.log_path)

        self._log_jsonl({
            "event": "monitor_started",
            "config": {
                "check_interval": self.check_interval,
                "max_restart_attempts": self.max_restart_attempts,
            },
        })

        while self._running:
            try:
                # Run the existing HealthChecker and the embedding probe in
                # parallel so a slow component does not block the others.
                stack_health, embedding_health = await asyncio.gather(
                    self.checker.check_all(),
                    self._check_embedding(),
                )

                # Normalise HealthStatus Pydantic objects into plain dicts.
                components: list[dict] = []
                for comp in stack_health.components:
                    entry: dict = {
                        "component": comp.component,
                        "healthy": comp.healthy,
                        "latency_ms": round(comp.latency_ms, 1),
                    }
                    if comp.error:
                        entry["error"] = comp.error
                    if comp.details:
                        entry["details"] = comp.details
                    components.append(entry)

                components.append(embedding_health)

                all_healthy = all(c.get("healthy", False) for c in components)

                self._log_jsonl({
                    "event": "health_check",
                    "all_healthy": all_healthy,
                    "components": components,
                })

                # Compact one-liner for operators tailing the journal.
                status_line = " | ".join(
                    f"{c['component']}:{'OK' if c.get('healthy') else 'FAIL'}"
                    for c in components
                )
                if all_healthy:
                    logger.info("All healthy: %s", status_line)
                else:
                    logger.warning("Issues detected: %s", status_line)

                # Auto-restart any failed services that map to a container.
                for comp in components:
                    if not comp.get("healthy", True) and comp["component"] in COMPONENT_TO_CONTAINER:
                        await self._restart_container(comp["component"])

            except Exception as e:
                logger.error("Health check cycle failed: %s", e, exc_info=True)
                self._log_jsonl({"event": "check_error", "error": str(e)})

            # Sleep in small increments so we can react to stop() quickly.
            for _ in range(self.check_interval):
                if not self._running:
                    break
                await asyncio.sleep(1)

        self._log_jsonl({"event": "monitor_stopped"})
        logger.info("Health Monitor stopped")

    def stop(self) -> None:
        """Signal the monitor loop to exit after the current cycle."""
        self._running = False


# ---- entry point -----------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve config relative to the project root (one level up from scripts/).
    config_path = os.path.join(os.path.dirname(__file__), "..", "dytopo_config.yaml")
    try:
        config = load_config(config_path)
    except Exception:
        logger.warning("Could not load dytopo_config.yaml -- using defaults")
        config = {}

    monitor = HealthMonitor(config)

    # Graceful shutdown on SIGINT / SIGTERM.
    def handle_signal(signum: int, _frame: object) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, shutting down...", sig_name)
        monitor.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    asyncio.run(monitor.run())


if __name__ == "__main__":
    main()
