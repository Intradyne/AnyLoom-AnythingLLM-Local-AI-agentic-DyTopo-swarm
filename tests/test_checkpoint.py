"""Tests for dytopo.checkpoint module."""

import asyncio
import json
import pytest

from dytopo.models import SwarmTask, SwarmDomain
from dytopo.checkpoint import CheckpointManager


@pytest.fixture
def task():
    return SwarmTask(task="test task", domain=SwarmDomain.CODE)


@pytest.fixture
def mgr(task, tmp_path):
    return CheckpointManager(task.task_id, checkpoint_dir=str(tmp_path))


@pytest.mark.asyncio
async def test_save_load_roundtrip(mgr, task):
    await mgr.save(task, "step1")
    result = await mgr.load_latest()
    assert result is not None
    loaded_task, step_label = result
    assert step_label == "step1"
    assert loaded_task.task == task.task
    assert loaded_task.task_id == task.task_id
    assert loaded_task.domain == task.domain


@pytest.mark.asyncio
async def test_atomic_write(mgr, task):
    path = await mgr.save(task, "atomic_test")
    assert path.exists()
    # No tmp files should remain
    tmp_files = list(mgr.task_dir.glob("*.tmp"))
    assert len(tmp_files) == 0
    # Verify content is valid JSON
    with open(path) as f:
        data = json.load(f)
    assert "__checkpoint_version__" in data
    assert data["__step_label__"] == "atomic_test"


@pytest.mark.asyncio
async def test_hot_task_detection(mgr, task):
    await mgr.save(task, "hot_test")
    hot = mgr.list_hot_tasks()
    task_ids = [h["task_id"] for h in hot]
    assert task.task_id in task_ids


@pytest.mark.asyncio
async def test_mark_completed(mgr, task):
    await mgr.save(task, "before_complete")
    mgr.mark_completed()
    hot = mgr.list_hot_tasks()
    task_ids = [h["task_id"] for h in hot]
    assert task.task_id not in task_ids


@pytest.mark.asyncio
async def test_cleanup(mgr, task):
    await mgr.save(task, "before_cleanup")
    assert mgr.task_dir.exists()
    mgr.cleanup()
    assert not mgr.task_dir.exists()


@pytest.mark.asyncio
async def test_load_latest_no_checkpoints(task, tmp_path):
    mgr = CheckpointManager(task.task_id, checkpoint_dir=str(tmp_path))
    # Remove any default files
    for f in mgr.task_dir.glob("*"):
        f.unlink()
    result = await mgr.load_latest()
    assert result is None


@pytest.mark.asyncio
async def test_multiple_saves_load_latest(mgr, task):
    await mgr.save(task, "step1")
    await mgr.save(task, "step2")
    await mgr.save(task, "step3")
    result = await mgr.load_latest()
    assert result is not None
    _, step_label = result
    assert step_label == "step3"
