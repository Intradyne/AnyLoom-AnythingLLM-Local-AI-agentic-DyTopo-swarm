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

## Getting Help

If you encounter an issue not covered here:

1. Export your trace: `await TraceCollector.export_trace(trace_id, Path("debug.json"))`
2. Export failures: `await failure_analyzer.export_failures_json(Path("failures.json"))`
3. Check audit log: `~/dytopo-logs/{task_id}/audit.jsonl`
4. Open an issue with the exported data
