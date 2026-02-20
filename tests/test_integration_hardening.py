"""Integration tests for reliability hardening modules."""

import asyncio
import json
import pytest

from dytopo.models import SwarmTask, SwarmDomain, AgentDescriptor, RoundRecord
from dytopo.checkpoint import CheckpointManager
from dytopo.governance import StalemateDetector, StalemateResult, get_generalist_fallback_agent
from dytopo.policy import PolicyEnforcer, PolicyDecision
from dytopo.verifier import OutputVerifier, VerificationResult


class TestCheckpointWithStalemate:
    """Test checkpoint saving at each orchestration step with stalemate detection."""

    @pytest.mark.asyncio
    async def test_full_2round_checkpoint_cycle(self, tmp_path):
        """Simulate a 2-round swarm with checkpoints at every step."""
        task = SwarmTask(task="Integration test task", domain=SwarmDomain.CODE)
        mgr = CheckpointManager(task.task_id, checkpoint_dir=str(tmp_path))
        sd = StalemateDetector(max_ping_pong=3, max_no_progress_rounds=2)

        # Round 1: goal
        task.progress_message = "Round 1"
        path1 = await mgr.save(task, "round_1_goal")
        assert path1.exists()

        # Round 1: descriptors
        path2 = await mgr.save(task, "round_1_descriptors")
        assert path2.exists()

        # Round 1: routing — record edges + convergence
        edges_r1 = [("developer", "researcher", 0.7), ("researcher", "tester", 0.6)]
        sd.record_round(edges_r1, 0.3)

        # Round 1: complete
        task.rounds.append(RoundRecord(round_num=1, goal="Implement REST API"))
        path3 = await mgr.save(task, "round_1_complete")
        assert path3.exists()

        # Stalemate check after round 1 — should not be stalled
        result = sd.detect()
        assert not result.is_stalled

        # Round 2: goal
        path4 = await mgr.save(task, "round_2_goal")
        assert path4.exists()

        # Round 2: routing with different edges
        edges_r2 = [("developer", "tester", 0.8), ("tester", "developer", 0.75)]
        sd.record_round(edges_r2, 0.5)

        # Round 2: complete
        task.rounds.append(RoundRecord(round_num=2, goal="Add authentication"))
        path5 = await mgr.save(task, "round_2_complete")
        assert path5.exists()

        # Verify load_latest returns most recent
        loaded = await mgr.load_latest()
        assert loaded is not None
        loaded_task, label = loaded
        assert label == "round_2_complete"
        assert len(loaded_task.rounds) == 2

        # Mark completed
        mgr.mark_completed()
        hot = mgr.list_hot_tasks()
        assert not any(h["task_id"] == task.task_id for h in hot)


class TestPolicyWithVerifier:
    """Test policy enforcement combined with verification."""

    @pytest.mark.asyncio
    async def test_policy_blocks_then_verifier_checks(self, tmp_path):
        """Policy denies dangerous paths, verifier checks code quality."""
        # Create a policy file
        policy = {
            "rules": {
                "file_write": {
                    "allow_paths": ["./workspace/*"],
                    "deny_paths": ["./src/*"],
                },
                "shell_exec": {
                    "allow_commands": ["python"],
                    "deny_commands": ["rm -rf", "sudo"],
                    "deny_patterns": ["|", "&&"],
                },
            },
            "enforcement": "strict",
            "log_denials": False,
        }
        policy_path = tmp_path / "policy.json"
        policy_path.write_text(json.dumps(policy))

        enforcer = PolicyEnforcer(policy_path=policy_path)

        # Policy should block writing to src/
        d = enforcer.check_tool_request("file_write", {"path": "./src/exploit.py"})
        assert not d.allowed

        # Policy should allow workspace writes
        d = enforcer.check_tool_request("file_write", {"path": "./workspace/result.txt"})
        assert d.allowed

        # Now verifier checks the output
        verifier = OutputVerifier({
            "enabled": True,
            "specs": {"developer": {"type": "syntax_check"}},
        })

        # Valid code passes
        vr = await verifier.verify("developer", "```python\nprint('hello')\n```")
        assert vr.passed

        # Invalid code fails
        vr = await verifier.verify("developer", "```python\ndef broken(\n```")
        assert not vr.passed
        assert vr.fix_hint  # Should contain guidance


class TestStalemateBreaksCheckpointFlow:
    """Test stalemate detection triggering early termination."""

    @pytest.mark.asyncio
    async def test_stalemate_triggers_force_terminate(self, tmp_path):
        """Simulate stalemate → force_terminate with checkpoints."""
        task = SwarmTask(task="Stalemate test", domain=SwarmDomain.GENERAL)
        mgr = CheckpointManager(task.task_id, checkpoint_dir=str(tmp_path))
        sd = StalemateDetector(max_no_progress_rounds=2)

        # 3 rounds with identical convergence score → no progress
        for i in range(1, 4):
            task.progress_message = f"Round {i}"
            await mgr.save(task, f"round_{i}_goal")
            sd.record_round([("A", "B", 0.6)], 0.5)
            task.rounds.append(RoundRecord(round_num=i, goal=f"Round {i} goal"))
            await mgr.save(task, f"round_{i}_complete")

        # Should detect stalemate
        result = sd.detect()
        assert result.is_stalled
        assert result.suggested_action == "force_terminate"

        # Verify checkpoints are intact
        loaded = await mgr.load_latest()
        assert loaded is not None
        _, label = loaded
        assert label == "round_3_complete"


class TestAllModulesEndToEnd:
    """Full end-to-end test with all hardening modules."""

    @pytest.mark.asyncio
    async def test_complete_hardened_flow(self, tmp_path):
        """Run through checkpoint → route → stalemate check → verify → policy → complete."""
        # Setup
        task = SwarmTask(task="E2E hardening test", domain=SwarmDomain.CODE)
        cp_mgr = CheckpointManager(task.task_id, checkpoint_dir=str(tmp_path / "checkpoints"))
        sd = StalemateDetector()

        policy = {
            "rules": {
                "file_write": {"allow_paths": ["./workspace/*"], "deny_paths": ["./src/*"]},
                "shell_exec": {"allow_commands": ["python"], "deny_commands": ["sudo"], "deny_patterns": []},
            },
            "enforcement": "strict",
            "log_denials": False,
        }
        policy_path = tmp_path / "policy.json"
        policy_path.write_text(json.dumps(policy))
        enforcer = PolicyEnforcer(policy_path=policy_path)

        verifier = OutputVerifier({
            "enabled": True,
            "specs": {"developer": {"type": "syntax_check"}},
        })

        # Simulate Round 1
        await cp_mgr.save(task, "round_1_goal")

        # Simulate agent output + verification
        agent_output = '```python\ndef add(a, b):\n    return a + b\n```'
        vr = await verifier.verify("developer", agent_output)
        assert vr.passed

        # Policy check for hypothetical file write
        pd = enforcer.check_tool_request("file_write", {"path": "./workspace/add.py"})
        assert pd.allowed

        # Record routing
        sd.record_round([("developer", "tester", 0.8)], 0.4)
        task.rounds.append(RoundRecord(round_num=1, goal="Build add function"))
        await cp_mgr.save(task, "round_1_complete")

        # Simulate Round 2
        await cp_mgr.save(task, "round_2_goal")
        sd.record_round([("tester", "developer", 0.75)], 0.7)
        task.rounds.append(RoundRecord(round_num=2, goal="Add tests"))
        await cp_mgr.save(task, "round_2_complete")

        # No stalemate — healthy progress (scores: 0.4, 0.7 → increasing)
        result = sd.detect()
        assert not result.is_stalled

        # Mark completed
        cp_mgr.mark_completed()

        # Verify final state
        loaded = await cp_mgr.load_latest()
        assert loaded is not None
        loaded_task, label = loaded
        assert label == "round_2_complete"
        assert len(loaded_task.rounds) == 2

        # Verify hot tasks cleared
        hot = cp_mgr.list_hot_tasks()
        assert not any(h["task_id"] == task.task_id for h in hot)
