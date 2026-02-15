"""Compare two benchmark result files and show regressions/improvements.

Usage:
    python scripts/benchmarks/bench_diff.py results/old_phase1.json results/new_phase1.json
    python scripts/benchmarks/bench_diff.py --before results/backup/ --after results/

The --before/--after form compares all matching phase files between two
directories (matches on filename).

Output legend:
    IMPROVED   FAIL -> PASS
    REGRESSED  PASS -> FAIL
    UNCHANGED  same grade
    WORDS      word count changed (shows delta)
    NEW        query exists only in the new file
    REMOVED    query exists only in the old file
"""
import os, json, sys, argparse, glob


def load_results(path):
    """Load a JSON file and return a dict keyed by result ID."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Phase 4 has a different structure: {"runs": {...}, "analysis": {...}}
    if isinstance(data, dict) and "analysis" in data:
        results = {}
        for qid, a in data["analysis"].items():
            results[qid] = {
                "id": qid,
                "grade": "PASS" if a.get("verdict") == "Deterministic"
                         or a.get("deterministic") else "FAIL",
                "word_count": a.get("avg", 0),
                "spread": a.get("spread", 0),
            }
        return results

    # Standard list-of-dicts format (Phases 1, 2, 3, 5, 6)
    if isinstance(data, list):
        return {r.get("id", str(i)): r for i, r in enumerate(data)}

    return {}


def diff_results(old, new):
    """Compare two result dicts. Returns list of change tuples."""
    changes = []
    all_ids = sorted(set(old.keys()) | set(new.keys()))

    for rid in all_ids:
        if rid not in old:
            changes.append((rid, "NEW", None, new[rid]))
            continue
        if rid not in new:
            changes.append((rid, "REMOVED", old[rid], None))
            continue

        o, n = old[rid], new[rid]
        og = o.get("grade", "?")
        ng = n.get("grade", "?")
        ow = o.get("word_count", 0)
        nw = n.get("word_count", 0)

        if og != ng:
            if og == "FAIL" and ng == "PASS":
                changes.append((rid, "IMPROVED", o, n))
            elif og == "PASS" and ng == "FAIL":
                changes.append((rid, "REGRESSED", o, n))
            else:
                changes.append((rid, "CHANGED", o, n))
        elif ow != nw:
            changes.append((rid, "WORDS", o, n))
        else:
            changes.append((rid, "UNCHANGED", o, n))

    return changes


def print_diff(changes, label=""):
    if label:
        print(f"\n{'=' * 50}")
        print(f"  {label}")
        print(f"{'=' * 50}")

    improved = [(c, o, n) for c, t, o, n in changes if t == "IMPROVED"]
    regressed = [(c, o, n) for c, t, o, n in changes if t == "REGRESSED"]
    word_changes = [(c, o, n) for c, t, o, n in changes if t == "WORDS"]
    new_items = [(c, o, n) for c, t, o, n in changes if t == "NEW"]
    removed = [(c, o, n) for c, t, o, n in changes if t == "REMOVED"]
    unchanged = [(c, o, n) for c, t, o, n in changes if t == "UNCHANGED"]

    if regressed:
        print(f"\n  REGRESSIONS ({len(regressed)}):")
        for rid, o, n in regressed:
            ow = o.get("word_count", "?") if o else "?"
            nw = n.get("word_count", "?") if n else "?"
            reason = n.get("reason", "") if n else ""
            print(f"    {rid}: PASS -> FAIL  ({ow}w -> {nw}w)  {reason}")

    if improved:
        print(f"\n  IMPROVEMENTS ({len(improved)}):")
        for rid, o, n in improved:
            ow = o.get("word_count", "?") if o else "?"
            nw = n.get("word_count", "?") if n else "?"
            print(f"    {rid}: FAIL -> PASS  ({ow}w -> {nw}w)")

    if word_changes:
        print(f"\n  WORD COUNT CHANGES ({len(word_changes)}):")
        for rid, o, n in word_changes:
            ow = o.get("word_count", 0) if o else 0
            nw = n.get("word_count", 0) if n else 0
            delta = nw - ow
            sign = "+" if delta > 0 else ""
            grade = n.get("grade", "?") if n else "?"
            print(f"    {rid}: {ow}w -> {nw}w ({sign}{delta})  [{grade}]")

    if new_items:
        print(f"\n  NEW ({len(new_items)}):")
        for rid, _, n in new_items:
            nw = n.get("word_count", "?") if n else "?"
            ng = n.get("grade", "?") if n else "?"
            print(f"    {rid}: {ng} ({nw}w)")

    if removed:
        print(f"\n  REMOVED ({len(removed)}):")
        for rid, o, _ in removed:
            print(f"    {rid}")

    total = len(changes)
    print(f"\n  Summary: {len(improved)} improved, {len(regressed)} regressed, "
          f"{len(word_changes)} word-count changes, {len(unchanged)} unchanged"
          f"{f', {len(new_items)} new' if new_items else ''}"
          f"{f', {len(removed)} removed' if removed else ''}")


def diff_files(old_path, new_path, label=""):
    old = load_results(old_path)
    new = load_results(new_path)
    changes = diff_results(old, new)
    print_diff(changes, label or f"{os.path.basename(old_path)} -> {os.path.basename(new_path)}")
    return changes


def diff_dirs(old_dir, new_dir):
    """Compare all matching phase files between two directories."""
    new_files = glob.glob(os.path.join(new_dir, "benchmarker_phase*.json"))
    all_changes = []
    for new_path in sorted(new_files):
        fname = os.path.basename(new_path)
        old_path = os.path.join(old_dir, fname)
        if os.path.exists(old_path):
            changes = diff_files(old_path, new_path, fname)
            all_changes.extend(changes)
        else:
            print(f"\n  {fname}: no baseline in {old_dir} (skipped)")
    return all_changes


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("files", nargs="*",
                        help="Two JSON files to compare (old new)")
    parser.add_argument("--before", help="Directory with old results")
    parser.add_argument("--after", help="Directory with new results")
    args = parser.parse_args()

    if args.before and args.after:
        diff_dirs(args.before, args.after)
    elif len(args.files) == 2:
        diff_files(args.files[0], args.files[1])
    else:
        print("Usage:")
        print("  python bench_diff.py old.json new.json")
        print("  python bench_diff.py --before old_dir/ --after new_dir/")
        sys.exit(1)


if __name__ == "__main__":
    main()
