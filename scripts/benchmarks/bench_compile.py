"""Compile all benchmark phase results into a unified summary.

Reads results/benchmarker_phase{1-6}*.json and prints:
  - Per-phase score table
  - Combined graded score
  - Notable findings (regressions, improvements)
  - Optionally writes results/benchmarker_summary.json

Usage:
    python scripts/benchmarks/bench_compile.py              # print summary to stdout
    python scripts/benchmarks/bench_compile.py --save       # also write summary JSON
"""
import os, json, argparse

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

PHASE_FILES = {
    1: "benchmarker_phase1_explanation.json",
    2: "benchmarker_phase2_adversarial.json",
    3: "benchmarker_phase3_workspace.json",
    4: "benchmarker_phase4_stability.json",
    5: "benchmarker_phase5_lmstudio.json",
    6: "benchmarker_phase6_showcase.json",
}

PHASE_NAMES = {
    1: "Explanation Tier",
    2: "Adversarial Fabrication",
    3: "Cross-Workspace Parity",
    4: "Depth Stability",
    5: "LM Studio Validation",
    6: "Showcase Gallery",
}


def load_phase(phase_num):
    path = os.path.join(RESULTS_DIR, PHASE_FILES[phase_num])
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_graded(data):
    """Score a list of result dicts with 'grade' field. Returns (passed, total)."""
    passed = sum(1 for r in data if r.get("grade") == "PASS")
    return passed, len(data)


def score_stability(data):
    """Score Phase 4 stability data. Returns (deterministic_count, total)."""
    if "analysis" in data:
        analysis = data["analysis"]
    else:
        analysis = data
    det = sum(1 for a in analysis.values()
              if isinstance(a, dict) and a.get("verdict") == "Deterministic")
    # Fallback for old format without 'verdict'
    if det == 0:
        det = sum(1 for a in analysis.values()
                  if isinstance(a, dict) and a.get("deterministic", False))
    return det, len(analysis)


def compile_all():
    summary = {}

    for phase_num in sorted(PHASE_FILES):
        data = load_phase(phase_num)
        if data is None:
            summary[phase_num] = {
                "name": PHASE_NAMES[phase_num],
                "status": "MISSING",
                "score": None,
            }
            continue

        entry = {"name": PHASE_NAMES[phase_num], "status": "OK"}

        if phase_num == 4:
            det, total = score_stability(data)
            entry["score"] = f"{det}/{total} deterministic"
            entry["passed"] = det
            entry["total"] = total
            entry["graded"] = False
            if "analysis" in data:
                for qid, a in data["analysis"].items():
                    entry.setdefault("details", {})[qid] = {
                        "avg": a.get("avg"),
                        "spread": a.get("spread"),
                        "verdict": a.get("verdict", "Unknown"),
                    }
        elif phase_num == 6:
            entry["score"] = f"{len(data)} collected"
            entry["graded"] = False
            entry["total"] = len(data)
            word_counts = {r["id"]: r["word_count"] for r in data}
            entry["word_counts"] = word_counts
        else:
            passed, total = score_graded(data)
            entry["score"] = f"{passed}/{total}"
            entry["passed"] = passed
            entry["total"] = total
            entry["graded"] = True
            # Collect failures
            failures = [r for r in data if r.get("grade") != "PASS"]
            if failures:
                entry["failures"] = [
                    {"id": r["id"], "reason": r.get("reason", ""),
                     "words": r.get("word_count")}
                    for r in failures
                ]

        summary[phase_num] = entry

    return summary


def print_summary(summary):
    print("=" * 60)
    print("ANYLOOM BENCHMARK SUMMARY")
    print("=" * 60)

    graded_pass = 0
    graded_total = 0

    for p in sorted(summary):
        s = summary[p]
        status = s.get("status", "?")
        if status == "MISSING":
            print(f"  Phase {p} ({s['name']}): NOT FOUND")
            continue

        score = s.get("score", "?")
        print(f"  Phase {p} ({s['name']}): {score}")

        if s.get("graded"):
            graded_pass += s.get("passed", 0)
            graded_total += s.get("total", 0)

        # Show failures
        for fail in s.get("failures", []):
            print(f"    FAIL {fail['id']}: {fail['reason']}")

        # Show stability details
        for qid, det in s.get("details", {}).items():
            v = det.get("verdict", "?")
            avg = det.get("avg", "?")
            spread = det.get("spread", "?")
            print(f"    {qid}: avg={avg}w, spread={spread} -> {v}")

    print()
    if graded_total > 0:
        pct = round(100 * graded_pass / graded_total)
        print(f"  Combined graded score: {graded_pass}/{graded_total} ({pct}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compile benchmark results")
    parser.add_argument("--save", action="store_true",
                        help="Write summary to results/benchmarker_summary.json")
    args = parser.parse_args()

    summary = compile_all()
    print_summary(summary)

    if args.save:
        out = os.path.join(RESULTS_DIR, "benchmarker_summary.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
