"""Run all benchmark phases, optionally in parallel pairs.

Usage:
    python scripts/benchmarks/bench_run_all.py              # run all 6 phases
    python scripts/benchmarks/bench_run_all.py 1 2 4        # run only phases 1, 2, 4
    python scripts/benchmarks/bench_run_all.py --serial     # no parallelism, run 1-6 sequentially

Default execution order (matches GPU contention constraints):
    Pair 1: Phase 1 + Phase 2  (both workspace c, independent queries)
    Pair 2: Phase 3 + Phase 4  (workspace a + workspace c, independent)
    Pair 3: Phase 5 + Phase 6  (LM Studio direct + workspace c showcase)

Phase 5 hits LM Studio directly (competes for GPU with AnythingLLM queries),
but in practice the 3-second sleep between queries avoids contention.
"""
import subprocess, sys, os, time, argparse

SCRIPTS = {
    1: "bench_phase1_explanation.py",
    2: "bench_phase2_adversarial.py",
    3: "bench_phase3_workspace.py",
    4: "bench_phase4_stability.py",
    5: "bench_phase5_lmstudio.py",
    6: "bench_phase6_showcase.py",
}

PARALLEL_PAIRS = [(1, 2), (3, 4), (5, 6)]

SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def run_phase(phase_num):
    """Run a single phase script, return (phase_num, returncode, elapsed_sec)."""
    script = os.path.join(SRC_DIR, SCRIPTS[phase_num])
    print(f"  [P{phase_num}] Starting {SCRIPTS[phase_num]}", flush=True)
    t0 = time.time()

    # Suppress Python warnings to avoid Windows beeps
    env = os.environ.copy()
    env['PYTHONWARNINGS'] = 'ignore'

    proc = subprocess.run(
        [sys.executable, '-W', 'ignore', script],
        capture_output=True, text=True, timeout=600, env=env
    )
    elapsed = time.time() - t0
    if proc.returncode == 0:
        # Print last 3 lines of stdout (score + save path)
        lines = proc.stdout.strip().splitlines()
        for line in lines[-3:]:
            print(f"  [P{phase_num}] {line}", flush=True)
    else:
        print(f"  [P{phase_num}] FAILED (exit {proc.returncode})", flush=True)
        print(f"  [P{phase_num}] stderr: {proc.stderr[:500]}", flush=True)
    return phase_num, proc.returncode, elapsed


def run_pair_parallel(a, b):
    """Run two phases in parallel using subprocess."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run_phase, n): n for n in (a, b)}
        for f in as_completed(futures):
            pnum, rc, elapsed = f.result()
            results[pnum] = (rc, elapsed)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run AnyLoom benchmark phases")
    parser.add_argument("phases", nargs="*", type=int,
                        help="Phase numbers to run (default: all)")
    parser.add_argument("--serial", action="store_true",
                        help="Run all phases sequentially (no parallelism)")
    args = parser.parse_args()

    requested = set(args.phases) if args.phases else set(SCRIPTS.keys())
    invalid = requested - set(SCRIPTS.keys())
    if invalid:
        print(f"Unknown phases: {invalid}. Valid: {sorted(SCRIPTS.keys())}")
        sys.exit(1)

    print(f"Running phases: {sorted(requested)}", flush=True)
    print(f"Mode: {'serial' if args.serial else 'parallel pairs'}\n", flush=True)

    all_results = {}
    t_start = time.time()

    if args.serial:
        for p in sorted(requested):
            pnum, rc, elapsed = run_phase(p)
            all_results[pnum] = (rc, elapsed)
            print(f"  [P{pnum}] done in {elapsed:.0f}s\n", flush=True)
    else:
        for a, b in PARALLEL_PAIRS:
            pair_phases = [p for p in (a, b) if p in requested]
            if len(pair_phases) == 2:
                print(f"--- Pair: P{pair_phases[0]} + P{pair_phases[1]} ---",
                      flush=True)
                results = run_pair_parallel(*pair_phases)
                all_results.update(results)
            elif len(pair_phases) == 1:
                pnum, rc, elapsed = run_phase(pair_phases[0])
                all_results[pnum] = (rc, elapsed)
            print(flush=True)

    total = time.time() - t_start
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    for p in sorted(all_results):
        rc, elapsed = all_results[p]
        status = "OK" if rc == 0 else f"FAIL (exit {rc})"
        print(f"  Phase {p}: {status}  ({elapsed:.0f}s)")
    print(f"\nTotal wall time: {total:.0f}s")

    failed = [p for p, (rc, _) in all_results.items() if rc != 0]
    if failed:
        print(f"\nFailed phases: {failed}")
        sys.exit(1)
    else:
        print("\nAll phases passed.")


if __name__ == "__main__":
    main()
