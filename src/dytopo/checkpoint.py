"""
DyTopo Checkpoint System
========================

Persists ``SwarmTask`` state to disk after each orchestration step so that a
crashed or interrupted run can be resumed from the last good checkpoint.

Design decisions
----------------
* **Atomic writes** -- each save writes to a temporary file in the same
  directory, then calls ``os.replace()`` which is atomic on POSIX and
  as-close-to-atomic-as-possible on Windows (same-volume rename).
* **Stdlib only** -- no ``aiofiles`` dependency; all blocking I/O is
  delegated to ``asyncio.to_thread()``.
* **Pydantic v2 round-trip** -- ``model_dump(mode="json")`` for
  serialization, ``model_validate()`` for deserialization.
* **Envelope format** -- every checkpoint JSON carries version, step
  label, and ISO-8601 timestamp metadata alongside the task payload.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from dytopo.models import SwarmTask

logger = logging.getLogger("dytopo.checkpoint")

__checkpoint_version__ = 1


class CheckpointManager:
    """Manage swarm checkpoints for crash recovery.

    Checkpoints are stored as JSON files in ``<checkpoint_dir>/<task_id>/``.
    Each checkpoint wraps ``SwarmTask.model_dump(mode="json")`` in a metadata
    envelope that records the checkpoint version, a human-readable step label,
    and an ISO-8601 timestamp.
    """

    def __init__(
        self,
        task_id: str,
        checkpoint_dir: str | Path = "~/dytopo-checkpoints",
    ) -> None:
        self.task_id = task_id
        self.base_dir = Path(checkpoint_dir).expanduser()
        self.task_dir = self.base_dir / task_id
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self._counter = self._next_counter()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_counter(self) -> int:
        """Scan existing checkpoints and return the next counter value."""
        existing = list(self.task_dir.glob("*.json"))
        if not existing:
            return 0
        max_counter = -1
        for f in existing:
            # Filename format: {step_label}_{counter:04d}.json
            stem = f.stem  # e.g. "round_2_goal_0003"
            parts = stem.rsplit("_", 1)
            if len(parts) == 2:
                try:
                    max_counter = max(max_counter, int(parts[1]))
                except ValueError:
                    pass
        return max_counter + 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save(self, task: SwarmTask, step_label: str) -> Path:
        """Save a checkpoint atomically.

        Args:
            task: The current ``SwarmTask`` instance to persist.
            step_label: Human-readable label such as ``"round_2_goal"``.

        Returns:
            Path to the saved checkpoint file.
        """
        envelope = {
            "__checkpoint_version__": __checkpoint_version__,
            "__step_label__": step_label,
            "__timestamp__": datetime.now().isoformat(),
            "task": task.model_dump(mode="json"),
        }

        counter = self._counter
        self._counter += 1
        filename = f"{step_label}_{counter:04d}.json"
        target = self.task_dir / filename

        def _atomic_write() -> None:
            # Write to a temp file in the same directory so os.replace()
            # is guaranteed to be atomic (same filesystem).
            tmp_path = str(target) + ".tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(envelope, f, indent=2, default=str)
                os.replace(tmp_path, str(target))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        await asyncio.to_thread(_atomic_write)
        logger.info("Checkpoint saved: %s", target.name)
        return target

    async def load_latest(self) -> tuple[SwarmTask, str] | None:
        """Load the most recent checkpoint for this task.

        Checkpoints are sorted by file modification time (newest first).
        Corrupt files are logged and skipped.

        Returns:
            A ``(SwarmTask, step_label)`` tuple, or ``None`` if no valid
            checkpoint exists.
        """

        def _extract_counter(p: Path) -> int:
            """Extract the numeric counter suffix from checkpoint filename."""
            parts = p.stem.rsplit("_", 1)
            if len(parts) == 2:
                try:
                    return int(parts[1])
                except ValueError:
                    pass
            return -1

        def _load() -> tuple[SwarmTask, str] | None:
            files = sorted(
                self.task_dir.glob("*.json"),
                key=lambda p: (p.stat().st_mtime, _extract_counter(p)),
                reverse=True,
            )
            for f in files:
                try:
                    with open(f, encoding="utf-8") as fh:
                        data = json.load(fh)
                    task = SwarmTask.model_validate(data["task"])
                    return (task, data["__step_label__"])
                except Exception as exc:
                    logger.warning(
                        "Skipping corrupt checkpoint %s: %s", f.name, exc
                    )
                    continue
            return None

        return await asyncio.to_thread(_load)

    def list_hot_tasks(self) -> list[dict]:
        """Scan all task directories for incomplete (non-completed) tasks.

        A task is considered *hot* (in-progress) when its directory does
        **not** contain a ``_completed`` marker file.

        Returns:
            List of dicts, each containing:
            - ``task_id``: directory name (matches the original task ID)
            - ``last_modified``: ISO-8601 timestamp of the newest checkpoint
            - ``checkpoint_count``: number of checkpoint files present
        """
        hot: list[dict] = []
        if not self.base_dir.exists():
            return hot
        for task_dir in self.base_dir.iterdir():
            if not task_dir.is_dir():
                continue
            if (task_dir / "_completed").exists():
                continue
            checkpoints = list(task_dir.glob("*.json"))
            if not checkpoints:
                continue
            newest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            hot.append({
                "task_id": task_dir.name,
                "last_modified": datetime.fromtimestamp(
                    newest.stat().st_mtime
                ).isoformat(),
                "checkpoint_count": len(checkpoints),
            })
        return hot

    def mark_completed(self) -> None:
        """Touch a ``_completed`` marker file in the task directory."""
        marker = self.task_dir / "_completed"
        marker.touch()
        logger.info("Task %s marked as completed", self.task_id)

    def cleanup(self) -> None:
        """Remove the entire task checkpoint directory."""
        if self.task_dir.exists():
            shutil.rmtree(self.task_dir)
            logger.info("Checkpoints cleaned up for task %s", self.task_id)
