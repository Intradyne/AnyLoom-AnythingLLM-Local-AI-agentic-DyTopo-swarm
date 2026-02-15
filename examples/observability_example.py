#!/usr/bin/env python3
"""
DyTopo Observability Layer Example
===================================

Demonstrates comprehensive observability instrumentation:
- Distributed tracing
- Metrics collection
- Performance profiling
- Failure analysis

Run this example to see the observability layer in action.

Usage:
    python examples/observability_example.py
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dytopo.observability import (
    TraceContext,
    TraceCollector,
    metrics,
    profiler,
    failure_analyzer,
    BottleneckAnalyzer,
    trace_id_var,
)


async def simulated_llm_call(agent_id: str, duration: float = 1.0):
    """
    Simulate an LLM API call with tracing and metrics.

    Args:
        agent_id: Which agent is making the call
        duration: How long the call takes (simulated)
    """
    async with TraceContext("llm_call", agent_id=agent_id):
        start = time.time()

        # Simulate LLM processing
        await asyncio.sleep(duration)

        # Record metrics
        actual_duration = time.time() - start
        await metrics.record(
            "llm_call_duration_seconds",
            actual_duration,
            agent=agent_id,
            status="success",
        )

        await metrics.record(
            "llm_tokens_total",
            duration * 1000,  # Fake token count
            agent=agent_id,
            token_type="completion",
        )

        return {"response": f"Result from {agent_id}"}


async def simulated_agent_execution(agent_id: str):
    """
    Simulate a full agent execution with potential failures.

    Args:
        agent_id: Agent identifier
    """
    async with TraceContext("agent_execution", agent_id=agent_id):
        try:
            # Descriptor generation
            async with TraceContext("descriptor_generation"):
                await asyncio.sleep(0.1)

            # LLM call
            result = await simulated_llm_call(agent_id, duration=0.5)

            # Post-processing
            async with TraceContext("post_processing"):
                await asyncio.sleep(0.05)

            return result

        except Exception as e:
            # Record failure
            await failure_analyzer.record_failure(
                component="agent",
                operation="execution",
                error=e,
                agent_id=agent_id,
            )
            raise


async def simulated_routing(agent_count: int):
    """
    Simulate routing graph construction.

    Args:
        agent_count: Number of agents
    """
    async with TraceContext("routing", agent_count=agent_count):
        # Embedding
        async with TraceContext("embedding"):
            await asyncio.sleep(0.2)

        # Similarity computation
        async with TraceContext("similarity_computation"):
            await asyncio.sleep(0.1 * agent_count)

        # Graph construction
        async with TraceContext("graph_construction"):
            await asyncio.sleep(0.05)


async def simulated_swarm(task: str, num_agents: int = 4, num_rounds: int = 3):
    """
    Simulate a full swarm execution with observability.

    Args:
        task: The task to execute
        num_agents: Number of agents in swarm
        num_rounds: Number of execution rounds
    """
    print(f"\n{'='*70}")
    print(f"SIMULATED SWARM EXECUTION")
    print(f"{'='*70}")
    print(f"Task: {task}")
    print(f"Agents: {num_agents}")
    print(f"Rounds: {num_rounds}")
    print()

    async with TraceContext("swarm_execution", task=task):
        for round_num in range(1, num_rounds + 1):
            print(f"\nRound {round_num}/{num_rounds}")

            async with TraceContext("round", round_num=round_num):
                # Manager call
                async with TraceContext("manager_call"):
                    await simulated_llm_call("manager", duration=0.3)

                # Descriptor generation (parallel)
                print("  - Generating descriptors...")
                descriptor_tasks = [
                    simulated_llm_call(f"agent_{i}", duration=0.2)
                    for i in range(num_agents)
                ]
                await asyncio.gather(*descriptor_tasks)

                # Routing
                print("  - Building routing graph...")
                await simulated_routing(num_agents)

                # Agent execution (sequential by topology)
                print("  - Executing agents...")
                for i in range(num_agents):
                    await simulated_agent_execution(f"agent_{i}")

            await asyncio.sleep(0.1)

    print(f"\n{'='*70}\n")


async def demonstrate_observability():
    """Main demonstration of observability features."""
    print("\n🔍 DyTopo Observability Demonstration\n")

    # Run a simulated swarm
    await simulated_swarm(
        task="Prove that 1+2+...+(2n-1) = n²",
        num_agents=4,
        num_rounds=3,
    )

    # Get the trace ID
    trace_id = trace_id_var.get()

    # ══════════════════════════════════════════════════════════════
    # 1. DISTRIBUTED TRACING
    # ══════════════════════════════════════════════════════════════

    print("📊 1. TRACE ANALYSIS")
    print("-" * 70)

    analysis = await TraceCollector.analyze_trace(trace_id)

    print(f"Trace ID: {trace_id}")
    print(f"Total Spans: {analysis['total_spans']}")
    print(f"Total Duration: {analysis['total_duration']:.2f}s")
    print(f"Error Rate: {analysis['error_rate']:.1%}")
    print()

    print("Operation Statistics:")
    for op, stats in sorted(
        analysis['operation_stats'].items(),
        key=lambda x: x[1]['total_duration'],
        reverse=True
    )[:5]:
        print(f"  • {op}:")
        print(f"      Count: {stats['count']}")
        print(f"      Avg Duration: {stats['avg_duration']:.3f}s")
        print(f"      Total Duration: {stats['total_duration']:.3f}s")

    # Export trace
    traces_dir = Path("traces")
    traces_dir.mkdir(exist_ok=True)
    trace_file = traces_dir / f"trace_{trace_id}.json"
    await TraceCollector.export_trace(trace_id, trace_file)
    print(f"\n✓ Trace exported to: {trace_file}")

    # ══════════════════════════════════════════════════════════════
    # 2. METRICS COLLECTION
    # ══════════════════════════════════════════════════════════════

    print(f"\n📈 2. METRICS ANALYSIS")
    print("-" * 70)

    # LLM latency stats
    llm_stats = await metrics.get_stats("llm_call_duration_seconds")
    if llm_stats:
        print(f"LLM Call Latency:")
        print(f"  Count: {llm_stats['count']}")
        print(f"  Mean: {llm_stats['mean']:.3f}s")
        print(f"  p50: {llm_stats['p50']:.3f}s")
        print(f"  p95: {llm_stats['p95']:.3f}s")
        print(f"  p99: {llm_stats['p99']:.3f}s")
        print(f"  Min: {llm_stats['min']:.3f}s")
        print(f"  Max: {llm_stats['max']:.3f}s")

    # Token stats
    token_stats = await metrics.get_stats("llm_tokens_total")
    if token_stats:
        print(f"\nToken Usage:")
        print(f"  Total: {token_stats['sum']:.0f} tokens")
        print(f"  Average per call: {token_stats['mean']:.0f} tokens")

    # Export metrics
    metrics_file = traces_dir / "metrics.prom"
    await metrics.export_prometheus(metrics_file)
    print(f"\n✓ Metrics exported to: {metrics_file}")

    # ══════════════════════════════════════════════════════════════
    # 3. PERFORMANCE PROFILING
    # ══════════════════════════════════════════════════════════════

    print(f"\n⚡ 3. BOTTLENECK ANALYSIS")
    print("-" * 70)

    bottleneck_report = await BottleneckAnalyzer.analyze(trace_id)

    print(f"Performance Score: {bottleneck_report['performance_score']}/100")
    print(f"Parallelization Factor: {bottleneck_report['parallelization_factor']:.2f}x")
    print()

    print("Slowest Operations:")
    for i, op in enumerate(bottleneck_report['slowest_operations'][:5], 1):
        print(f"  {i}. {op['operation']}: {op['duration']:.3f}s")

    print()
    print("Recommendations:")
    for rec in bottleneck_report['recommendations']:
        priority_mark = {
            'high': '🔴',
            'medium': '🟡',
            'low': '🟢',
        }.get(rec['priority'], '⚪')

        print(f"  {priority_mark} [{rec['priority'].upper()}] {rec['category']}")
        print(f"     {rec['message']}")

    # Generate detailed report
    report_text = BottleneckAnalyzer.format_report(bottleneck_report)
    report_file = traces_dir / "performance_report.txt"
    report_file.write_text(report_text)
    print(f"\n✓ Detailed report: {report_file}")

    # ══════════════════════════════════════════════════════════════
    # 4. FAILURE ANALYSIS
    # ══════════════════════════════════════════════════════════════

    print(f"\n❌ 4. FAILURE ANALYSIS")
    print("-" * 70)

    failure_patterns = await failure_analyzer.analyze_patterns()

    if failure_patterns['total_failures'] > 0:
        print(f"Total Failures: {failure_patterns['total_failures']}")
        print(f"Recovery Rate: {failure_patterns['recovery_rate']:.1%}")
        print(f"Most Common Error: {failure_patterns['most_common_error']}")

        print("\nFailures by Type:")
        for error_type, count in failure_patterns['by_type'].items():
            print(f"  • {error_type}: {count}")
    else:
        print("No failures detected ✅")

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("✅ OBSERVABILITY DEMONSTRATION COMPLETE")
    print(f"{'='*70}\n")

    print("Generated artifacts:")
    print(f"  • Trace: {trace_file}")
    print(f"  • Metrics: {metrics_file}")
    print(f"  • Performance Report: {report_file}")
    print()


if __name__ == "__main__":
    asyncio.run(demonstrate_observability())
