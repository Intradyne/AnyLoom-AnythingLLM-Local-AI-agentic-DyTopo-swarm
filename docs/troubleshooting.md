# DyTopo Troubleshooting Guide

## Common Issues and Solutions

### Issue: Swarm Takes Too Long (>5 minutes)

**Symptoms:**
- Swarm execution exceeds expected time
- Some rounds take significantly longer than others

**Diagnosis:**
1. Check trace analysis for bottlenecks:
   ```python
   from dytopo.observability import TraceCollector
   analysis = await TraceCollector.analyze_trace(trace_id)
   print(analysis["slowest_operations"])
   ```

2. Review performance score:
   ```python
   from dytopo.observability import BottleneckAnalyzer
   report = await BottleneckAnalyzer.analyze(trace_id)
   print(f"Score: {report['performance_score']}/100")
   ```

**Solutions:**
- If parallelization_factor < 2.0: Increase concurrent agent execution
- If long-pole operations detected: Optimize slow agents or reduce max_tokens
- If routing takes >1s: Reduce agent count or simplify descriptors

---

### Issue: Agent Failures / Timeouts

**Symptoms:**
- Agents repeatedly fail or timeout
- Error rate >10%

**Diagnosis:**
```python
from dytopo.observability import failure_analyzer
patterns = await failure_analyzer.analyze_patterns()
print(patterns["most_common_error"])
```

**Solutions:**
- **TimeoutError**: Increase timeout in config, reduce max_tokens
- **JSONDecodeError**: Enable JSON repair, review LLM response format
- **ConnectionError**: Check llama.cpp is running (`docker compose ps`), verify base_url

---

### Issue: Convergence Detected Too Early

**Symptoms:**
- Swarm terminates before task is complete
- Final answer is incomplete or wrong

**Diagnosis:**
Check convergence settings in config:
```yaml
orchestration:
  convergence_threshold: 0.95  # Increase to 0.98 for stricter
```

**Solutions:**
- Increase convergence_threshold (0.95 → 0.98)
- Increase convergence window_size (3 → 4 rounds)
- Add diversity to agent prompts to encourage exploration

---

### Issue: Routing Graph Is Too Sparse/Dense

**Symptoms:**
- Too sparse: Agents isolated, missing relevant information
- Too dense: Information overload, no filtering benefit

**Diagnosis:**
```python
# Check routing density per round
for round in swarm.rounds:
    print(f"Round {round.round_num}: {round.routing_stats['density']:.2%}")
```

**Solutions:**
- **Too sparse** (density <10%): Decrease tau (0.3 → 0.2)
- **Too dense** (density >70%): Increase tau (0.3 → 0.4) or decrease K_in

---

### Issue: High Token Consumption

**Symptoms:**
- total_tokens exceeds budget
- LLM API costs too high

**Diagnosis:**
```python
from dytopo.observability import metrics
stats = await metrics.get_stats("llm_tokens_total")
print(f"Average tokens/call: {stats['mean']}")
```

**Solutions:**
- Reduce max_tokens_work in config (4096 → 2048)
- Enable work truncation in routing
- Reduce T_max (5 → 3 rounds)
- Use broadcast_round_1: false to skip round 1 full broadcast

---

### Issue: Memory Issues / OOM

**Symptoms:**
- Process killed by OOM
- Memory usage grows unbounded

**Diagnosis:**
```python
import psutil
process = psutil.Process()
print(f"Memory: {process.memory_info().rss / 1024**2:.0f} MB")
```

**Solutions:**
- Enable metric cleanup:
  ```python
  await metrics.cleanup_old_metrics()
  ```
- Clear trace collector periodically:
  ```python
  await TraceCollector.clear()
  ```
- Reduce retention_hours for metrics (24 → 6)

---

### Issue: Health Check / Preflight Failures

**Symptoms:**
- Swarm aborts with `RuntimeError` (LLM unreachable)
- Warnings about Qdrant, AnythingLLM, or GPU being unhealthy

**Diagnosis:**
```bash
# Check Docker container status
docker compose ps

# Use the system-status MCP server (if running)
# Or check endpoints directly:
curl http://localhost:8008/v1/models      # LLM
curl http://localhost:8009/health          # Embedding
curl http://localhost:6333/health          # Qdrant
curl http://localhost:3001/api/v1/system   # AnythingLLM
```

**Solutions:**
- **LLM unreachable**: Check `docker logs anyloom-llm` — model may still be loading (~1-2 min on startup)
- **Qdrant down**: `docker compose restart qdrant`
- **Embedding down**: `docker compose restart embedding`
- **GPU not detected**: Verify NVIDIA driver (`nvidia-smi`) and Docker GPU access

**Health Monitor:** Run `python scripts/health_monitor.py` for continuous monitoring with auto-restart. Logs to `~/anyloom-logs/health.jsonl`.

---

### Issue: Stigmergic Traces Not Being Used

**Symptoms:**
- `trace_context` is empty in routing results
- No improvement in routing over time

**Diagnosis:**
```python
from dytopo.stigmergic_router import StigmergicRouter
router = StigmergicRouter()
stats = await router.get_trace_stats()
print(stats)  # Check point count
```

**Solutions:**
- Verify `traces.enabled: true` in `dytopo_config.yaml`
- Check Qdrant is reachable at the configured `traces.qdrant_url`
- Ensure swarms are completing with quality >= `min_quality` (default 0.5) — low-quality runs don't deposit traces
- Check that `swarm_traces` collection exists in Qdrant dashboard (http://localhost:6333/dashboard)
- Trace boost is subtle (`boost_weight: 0.15`) — run several swarms on similar tasks to build up traces

---

### Issue: Checkpoint Recovery Issues

**Symptoms:**
- Swarm does not resume from previous checkpoint after crash
- `CheckpointManager.load_latest()` returns `None`
- Corrupt checkpoint warnings in logs

**Diagnosis:**
```bash
# Check checkpoint directory exists and has files
ls -la ~/dytopo-checkpoints/

# Check for specific task checkpoints
ls -la ~/dytopo-checkpoints/{task_id}/

# Look for corrupt checkpoint warnings in logs
grep "Skipping corrupt checkpoint" ~/dytopo-logs/*/audit.jsonl
```

**Solutions:**
- **No checkpoint directory**: Verify `checkpoint.enabled: true` and `checkpoint.checkpoint_dir` in `dytopo_config.yaml`
- **Corrupt checkpoints**: The manager skips corrupt files automatically and tries the next oldest. If all are corrupt, the swarm starts fresh
- **Stale hot tasks**: `list_hot_tasks()` returns tasks without a `_completed` marker. Call `mark_completed()` after successful runs or `cleanup()` to remove stale checkpoints
- **Disk full**: Checkpoints accumulate per-task. Run `CheckpointManager.cleanup()` for old task IDs or delete `~/dytopo-checkpoints/` manually

---

### Issue: Policy Enforcement Blocks

**Symptoms:**
- Agent tool calls rejected with `"error": "policy_denied"`
- Unexpected `PolicyDecision(allowed=False)` results
- Swarm cannot write files or execute commands needed for the task

**Diagnosis:**
```bash
# Check policy file exists and is valid JSON
python -c "import json; json.load(open('policy.json'))"

# Check for policy denial logs
grep "Policy denied" ~/dytopo-logs/*/audit.jsonl
```

```python
from dytopo.policy import PolicyEnforcer
enforcer = PolicyEnforcer()
# Test a specific file path
result = enforcer.check_tool_request("file_write", {"path": "/tmp/test.txt"})
print(result)
# Test a specific command
result = enforcer.check_tool_request("shell_exec", {"command": "python test.py"})
print(result)
```

**Solutions:**
- **Legitimate tool call blocked**: Add the path or command to the `allow_paths` / `allow_commands` list in `policy.json`
- **Path traversal false positive**: The enforcer resolves all paths to absolute. Check that your `allow_paths` patterns match the resolved absolute path
- **Missing policy.json**: Without a policy file, all requests are allowed. If the file is missing unexpectedly, create one from the template
- **Enforcement mode**: Set `"enforcement": "permissive"` in `policy.json` to log denials without blocking (useful for debugging)

---

### Issue: Verification Gate Failures

**Symptoms:**
- Agent outputs flagged by `OutputVerifier` with `passed=False`
- `VerificationResult` shows syntax errors or missing JSON fields
- Swarm rounds produce lower-quality outputs than expected

**Diagnosis:**
```python
from dytopo.verifier import OutputVerifier

config = {"enabled": True, "specs": {"developer": {"type": "syntax_check"}}}
verifier = OutputVerifier(config)
result = await verifier.verify("developer", agent_output)
print(f"Passed: {result.passed}, Method: {result.method}")
if not result.passed:
    print(f"Error: {result.stderr}")
    print(f"Fix hint: {result.fix_hint}")
```

**Solutions:**
- **Syntax check failures**: The verifier extracts Python code from markdown fences. If the agent output wraps code in unusual formatting, the extraction may fail. Check `fix_hint` for the specific syntax error
- **Schema validation failures**: Ensure the agent's JSON output contains all `required_fields` defined in the verification spec
- **Code execution timeouts**: Increase `timeout_seconds` in the verification spec for the affected role
- **False positives from infrastructure errors**: The verifier is fail-open by design — infrastructure errors return `passed=True`. If you see unexpected failures, check the `method` field to confirm which verification strategy ran
- **Disable for specific roles**: Remove the role from `verification.specs` in `dytopo_config.yaml` to skip verification for that role

---

### Issue: Stalemate Detection False Positives

**Symptoms:**
- `StalemateDetector` reports stalemate when the swarm is making progress
- Generalist fallback agent injected unnecessarily
- Forced termination triggered too early

**Diagnosis:**
```python
from dytopo.governance import StalemateDetector, StalemateResult

detector = StalemateDetector()
result = detector.check(round_history, convergence_scores)
print(f"Stalled: {result.is_stalled}, Reason: {result.reason}")
print(f"Action: {result.suggested_action}, Pair: {result.stale_pair}")
```

**Solutions:**
- **Ping-pong false positive**: Two agents routing to each other can be legitimate collaboration. If the convergence score is improving, the stalemate detection may be too aggressive. This pattern requires 3+ consecutive rounds of bidirectional routing to trigger
- **No-progress false positive**: The detector flags convergence score changes < 0.01 as stalled. If your task naturally has a long plateau before improvement, consider adjusting the convergence threshold in config
- **Regression false positive**: Temporary convergence score dips can happen when new information is introduced. The detector requires sustained regression over 2+ rounds
- **Disable stalemate detection**: Set `_HAS_STALEMATE = False` in the orchestrator (not recommended for production) or handle the `StalemateResult` at the orchestrator level

---

## Performance Optimization Checklist

- [ ] Run performance profiler to identify bottlenecks
- [ ] Check parallelization factor (target: >2.0)
- [ ] Review slowest operations in trace
- [ ] Optimize routing density (target: 20-50%)
- [ ] Enable convergence detection
- [ ] Tune tau and K_in for routing
- [ ] Reduce max_tokens if appropriate
- [ ] Monitor failure patterns and address root causes

---

## Debugging Tools

### View Trace
```python
from dytopo.observability import TraceCollector
trace_id = trace_id_var.get()
await TraceCollector.export_trace(trace_id, Path("trace.json"))
```

### View Metrics
```python
from dytopo.observability import metrics
summary = await metrics.get_summary()
print(summary)
```

### View Failures
```python
from dytopo.observability import failure_analyzer
patterns = await failure_analyzer.analyze_patterns()
print(patterns)
```

### Generate Performance Report
```python
from dytopo.observability import BottleneckAnalyzer
report = await BottleneckAnalyzer.analyze(trace_id)
print(BottleneckAnalyzer.format_report(report))
```

---

### Visualize Trace (CLI)
```bash
# Generate HTML timeline from audit log
python scripts/visualize_trace.py ~/dytopo-logs/task_abc123 --format html --output report.html

# Generate Mermaid flowchart
python scripts/visualize_trace.py ~/dytopo-logs/task_abc123 --format mermaid

# Enable loop/stall detection
python scripts/visualize_trace.py ~/dytopo-logs/task_abc123 --detect-loops

# Direct path to audit file
python scripts/visualize_trace.py ~/dytopo-logs/task_abc123/audit.jsonl --format mermaid --output flow.md
```

### View Health Monitor Logs
```bash
# Tail the health monitor JSONL log
tail -f ~/anyloom-logs/health.jsonl | python -m json.tool

# Filter for failures only
grep '"healthy": false' ~/anyloom-logs/health.jsonl
```

### Check Stack Status (via MCP)
```python
# If system-status MCP server is running:
# Tools: service_health, qdrant_collections, gpu_status,
#        llm_slots, docker_status, stack_config
```

---

## Getting Help

If you encounter an issue not covered here:

1. Export your trace: `await TraceCollector.export_trace(trace_id, Path("debug.json"))`
2. Export failures: `await failure_analyzer.export_failures_json(Path("failures.json"))`
3. Check audit log: `~/dytopo-logs/{task_id}/audit.jsonl`
4. Check health log: `~/anyloom-logs/health.jsonl`
5. Open an issue with the exported data
