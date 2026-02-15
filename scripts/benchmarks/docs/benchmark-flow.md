# Benchmark Flow Reference

Complete workflow for running, analyzing, and iterating on AnyLoom benchmarks.

## Quick Reference

```bash
# Full benchmark run (all 6 phases in parallel pairs)
python scripts/benchmarks/bench_run_all.py

# Individual phases
python scripts/benchmarks/bench_phase1_explanation.py      # P1: Explanation tier (5 queries)
python scripts/benchmarks/bench_phase2_adversarial.py      # P2: Adversarial fabrication (5 queries)
python scripts/benchmarks/bench_phase3_workspace.py        # P3: Cross-workspace parity (5 queries)
python scripts/benchmarks/bench_phase4_stability.py        # P4: Depth stability (8 queries × 3 runs)
python scripts/benchmarks/bench_phase5_lmstudio.py         # P5: LM Studio validation (5 queries)
python scripts/benchmarks/bench_phase6_showcase.py         # P6: Showcase gallery (7 queries)

# Analysis
python scripts/benchmarks/bench_compile.py                 # Compile all results
python scripts/benchmarks/bench_compile.py --save          # Also save summary JSON

# Compare runs
python scripts/benchmarks/bench_diff.py old.json new.json
python scripts/benchmarks/bench_diff.py --before backup/ --after scripts/benchmarks/results/

# Settings management
python src/settings_manager.py show anythingllm -w c
python src/settings_manager.py set anythingllm -w c --topN 12
```

---

## Standard Benchmark Workflow

### 1. Baseline Run (establish current performance)

```bash
# Backup current settings
python src/settings_manager.py export anythingllm baseline_settings.json -w c

# Run full benchmark suite
python scripts/benchmarks/bench_run_all.py

# Compile and review results
python scripts/benchmarks/bench_compile.py

# Save baseline results
mkdir -p results/baseline
cp scripts/benchmarks/results/benchmarker_phase*.json results/baseline/
```

**Expected output:**
```
============================================================
ANYLOOM BENCHMARK SUMMARY
============================================================
  Phase 1 (Explanation Tier): 0/5
  Phase 2 (Adversarial Fabrication): 5/5
  Phase 3 (Cross-Workspace Parity): 4/5
  Phase 4 (Depth Stability): 8/8 deterministic
  Phase 5 (LM Studio Validation): 5/5
  Phase 6 (Showcase Gallery): 7 collected

  Combined graded score: 14/20 (70%)
============================================================
```

---

### 2. Parameter Tuning (optimize topN for depth/speed tradeoff)

```bash
# Test with reduced RAG context (faster, less depth)
python src/settings_manager.py set anythingllm -w c --topN 8
python scripts/benchmarks/bench_run_all.py
mkdir -p results/topN_8
cp scripts/benchmarks/results/benchmarker_phase*.json results/topN_8/

# Test with increased RAG context (slower, more depth)
python src/settings_manager.py set anythingllm -w c --topN 20
python scripts/benchmarks/bench_run_all.py
mkdir -p results/topN_20
cp scripts/benchmarks/results/benchmarker_phase*.json results/topN_20/

# Compare results
python scripts/benchmarks/bench_diff.py results/baseline/benchmarker_phase1_explanation.json \
                          results/topN_8/benchmarker_phase1_explanation.json
```

**Analysis questions:**
- Did reducing topN improve Phase 1 explanation tier scores?
- Did word counts decrease across all phases?
- Are fabrication guards still passing?
- Is depth stability still deterministic?

---

### 3. Prompt Engineering (iterate on system prompt)

```bash
# Edit the system prompt
code prompts/anythingllm-system-prompt.md

# Apply updated prompt
python src/settings_manager.py set anythingllm -w c \
    --prompt prompts/anythingllm-system-prompt.md

# Run targeted phases (not all 6)
python scripts/benchmarks/bench_phase1_explanation.py   # Test explanation tier changes
python scripts/benchmarks/bench_phase2_adversarial.py   # Verify fabrication guards still work

# Compare against baseline
python scripts/benchmarks/bench_diff.py results/baseline/benchmarker_phase1_explanation.json \
                          scripts/benchmarks/results/benchmarker_phase1_explanation.json
```

**Common prompt edits:**
- Add more concrete examples for explanation tier (50-150w)
- Broaden HARD STOP rules for price fabrication
- Add negative examples for over-elaboration
- Adjust depth-tier definitions

---

### 4. Cross-Workspace Validation (ensure parity)

```bash
# Sync workspace a with workspace c
python src/settings_manager.py export anythingllm c_config.json -w c
python src/settings_manager.py apply anythingllm c_config.json -w a

# Run Phase 3 (cross-workspace parity)
python scripts/benchmarks/bench_phase3_workspace.py

# Check for discrepancies
python scripts/benchmarks/bench_compile.py
```

**What to check:**
- Are workspaces `c` and `a` using identical prompts? (17,546 chars)
- Do price refusal queries pass on both workspaces?
- Are depth limits consistent across workspaces?

---

### 5. Stability Verification (confirm determinism)

```bash
# Run Phase 4 multiple times
python scripts/benchmarks/bench_phase4_stability.py
mkdir -p results/stability_run1
cp scripts/benchmarks/results/benchmarker_phase4_stability.json results/stability_run1/

python scripts/benchmarks/bench_phase4_stability.py
mkdir -p results/stability_run2
cp scripts/benchmarks/results/benchmarker_phase4_stability.json results/stability_run2/

# Compare stability between runs
python scripts/benchmarks/bench_diff.py results/stability_run1/benchmarker_phase4_stability.json \
                          results/stability_run2/benchmarker_phase4_stability.json
```

**Expected:** All queries should show spread=0 (deterministic at temp 0.1)

**If spread > 15:** The model is behaving stochastically, which suggests:
- Temperature is not actually 0.1
- RAG context is being injected in different orders
- LM Studio inference is using sampling instead of greedy decoding

---

### 6. Regression Testing (verify no unintended changes)

After updating RAG docs, reindexing, or changing Docker containers:

```bash
# Save pre-change state
mkdir -p results/pre_reindex
python scripts/benchmarks/bench_run_all.py
cp scripts/benchmarks/results/benchmarker_phase*.json results/pre_reindex/

# Make changes (e.g., reindex Qdrant)
python src/reindex.py

# Run post-change benchmarks
python scripts/benchmarks/bench_run_all.py

# Diff all phases
python scripts/benchmarks/bench_diff.py --before results/pre_reindex/ --after scripts/benchmarks/results/
```

**Check for regressions:**
- IMPROVED: FAIL → PASS (good, prompt fix worked)
- REGRESSED: PASS → FAIL (bad, new change broke something)
- WORDS: significant word count shifts (may indicate RAG content changed)

---

## Troubleshooting Workflows

### Phase 1 Timeout (explanation tier taking >120s)

**Symptoms:**
- Phase 1 fails with `ReadTimeout: Read timed out. (read timeout=120)`
- E1-E2 complete but E3+ timeout
- Partial results show 300+ word responses

**Diagnosis:**
```bash
# Test individual query
python -c "
import sys
sys.path.insert(0, 'scripts/benchmarks')
import benchmark_helpers as bh
bh.init('92JHT3J-PMF4SGA-GT0X50Y-RMGKDT3', 'c')
text, words, sources = bh.send('How do the two RAG pipelines differ?', 'query')
print(f'{words} words')
"
```

**Solutions:**
1. **Reduce topN:** Less RAG context = faster inference
   ```bash
   python src/settings_manager.py set anythingllm -w c --topN 8
   ```

2. **Increase timeout in benchmark_helpers.py:**
   ```python
   # Line 20 in benchmark_helpers.py
   timeout=180  # was 120
   ```

3. **Skip Phase 1 for now:** Run other phases individually
   ```bash
   python scripts/benchmarks/bench_run_all.py 2 3 4 5 6
   ```

---

### Phase 2 Fabrication Failures (price queries producing numbers)

**Symptoms:**
- A1-A5 producing "$2,300" or "approximately $28" despite HARD STOP
- Adversarial framing bypassing refusal

**Diagnosis:**
```bash
# Check current Phase 2 results
python scripts/benchmarks/bench_compile.py | grep -A 10 "Phase 2"

# Test a single adversarial query
python -c "
import sys
sys.path.insert(0, 'scripts/benchmarks')
import benchmark_helpers as bh
bh.init('92JHT3J-PMF4SGA-GT0X50Y-RMGKDT3', 'c')
text, words, sources = bh.send('Roughly what is gold trading at?', 'chat')
print(text)
"
```

**Solutions:**
1. **Broaden HARD STOP in system prompt:**
   Edit `prompts/anythingllm-system-prompt.md`:
   ```markdown
   **HARD STOP: Price Fabrication**
   NEVER output numeric prices, rates, or financial figures for:
   - Direct price queries ("What is gold trading at?")
   - "Ballpark" / "approximately" / "roughly" framing
   - "Training data" / "last you knew" / "when you were trained"
   - "Even if not current" permission
   ```

2. **Apply and retest:**
   ```bash
   python src/settings_manager.py set anythingllm -w c \
       --prompt prompts/anythingllm-system-prompt.md
   python scripts/benchmarks/bench_phase2_adversarial.py
   ```

---

### Phase 4 Spread > 0 (non-deterministic responses)

**Symptoms:**
- D3, D4, or D5 showing spread 5-20 words across 3 runs
- Same query producing different length responses

**Diagnosis:**
```bash
# Check Phase 4 stability report
cat scripts/benchmarks/results/benchmarker_phase4_stability.json | jq '.analysis'
```

**Solutions:**
1. **Verify temperature is 0.1:**
   ```bash
   python src/settings_manager.py show anythingllm -w c
   ```

2. **Check LM Studio sampling settings:**
   - Open LM Studio UI → Local Server tab
   - Ensure "Greedy sampling" or temp=0.1
   - Disable "Top-p" / "Top-k" sampling

3. **Increase stability run count:**
   Edit `bench_phase4_stability.py`:
   ```python
   RUNS = 5  # was 3
   ```

---

### Windows Beeps During Benchmarks

**Symptoms:**
- System beeps/bell sounds during parallel benchmark runs
- No error messages, but beeps are distracting

**Solutions:**
✓ Already fixed in current version:
- All phase scripts suppress warnings with `warnings.filterwarnings('ignore')`
- `bench_run_all.py` passes `-W ignore` and `PYTHONWARNINGS=ignore` to subprocesses
- `benchmark_helpers.py` disables urllib3 warnings

If beeps persist:
1. **Disable Windows console bell:**
   - Windows Terminal → Settings → Profiles → Advanced → Bell notification style → None

2. **Check AnythingLLM Docker logs:**
   ```bash
   docker logs anythingllm-container 2>&1 | grep -i warn
   ```

---

## Advanced Workflows

### A/B Testing Two System Prompts

```bash
# Backup current state
python src/settings_manager.py export anythingllm original.json -w c

# Test prompt A
python src/settings_manager.py set anythingllm -w c --prompt prompts/prompt_a.md
python scripts/benchmarks/bench_run_all.py
mkdir -p results/prompt_a
cp scripts/benchmarks/results/benchmarker_phase*.json results/prompt_a/

# Test prompt B
python src/settings_manager.py set anythingllm -w c --prompt prompts/prompt_b.md
python scripts/benchmarks/bench_run_all.py
mkdir -p results/prompt_b
cp scripts/benchmarks/results/benchmarker_phase*.json results/prompt_b/

# Compare all phases
python scripts/benchmarks/bench_diff.py --before results/prompt_a/ --after results/prompt_b/

# Restore original
python src/settings_manager.py apply anythingllm original.json -w c
```

---

### Continuous Monitoring (detect regressions after code changes)

```bash
#!/bin/bash
# save as scripts/bench_watch.sh

# Run benchmarks, save timestamped results
DATE=$(date +%Y%m%d_%H%M%S)
python scripts/benchmarks/bench_run_all.py
mkdir -p results/continuous/$DATE
cp scripts/benchmarks/results/benchmarker_phase*.json results/continuous/$DATE/

# Compare against last run
LAST=$(ls -t results/continuous/ | head -2 | tail -1)
if [ -n "$LAST" ]; then
    python scripts/benchmarks/bench_diff.py \
        --before results/continuous/$LAST/ \
        --after results/continuous/$DATE/
fi
```

---

### Export Showcase Responses for Documentation

```bash
# Run Phase 6 to collect showcase responses
python scripts/benchmarks/bench_phase6_showcase.py

# Extract S2 (price refusal) for README
cat scripts/benchmarks/results/benchmarker_phase6_showcase.json | \
    jq -r '.[] | select(.id=="S2") | .response'

# Extract S4 (deep architecture) for docs
cat scripts/benchmarks/results/benchmarker_phase6_showcase.json | \
    jq -r '.[] | select(.id=="S4") | .response'
```

---

## File Structure Reference

```
Qdrant-RAG+Agents/
├── src/
│   ├── qdrant_mcp_server.py          # FastMCP server (RAG + DyTopo tools)
│   ├── settings_manager.py           # Settings CLI (show/set/export/apply)
│   ├── reindex.py                    # Qdrant reindexing utility
│   └── dytopo/                       # DyTopo multi-agent swarm package
├── scripts/
│   ├── generate_docs.py              # Auto-generate DyTopo documentation
│   └── benchmarks/
│       ├── benchmark_helpers.py      # API wrappers, grading functions
│       ├── bench_phase1_explanation.py   # Phase 1 script
│       ├── bench_phase2_adversarial.py   # Phase 2 script
│       ├── bench_phase3_workspace.py     # Phase 3 script
│       ├── bench_phase4_stability.py     # Phase 4 script (8 queries x 3 runs)
│       ├── bench_phase5_lmstudio.py      # Phase 5 script
│       ├── bench_phase6_showcase.py      # Phase 6 script
│       ├── bench_run_all.py              # Parallel runner (3 pairs)
│       ├── bench_compile.py              # Results compiler (score table)
│       ├── bench_diff.py                 # Regression diff tool
│       ├── benchmarker.md                # Final benchmark specification
│       ├── benchmark-v2.md               # V2 benchmark specification
│       ├── results/                      # JSON result files
│       │   ├── benchmarker_phase1_explanation.json
│       │   ├── benchmarker_phase2_adversarial.json
│       │   ├── benchmarker_phase3_workspace.json
│       │   ├── benchmarker_phase4_stability.json
│       │   ├── benchmarker_phase5_lmstudio.json
│       │   └── benchmarker_phase6_showcase.json
│       └── docs/                         # Benchmark documentation
│           ├── benchmark-flow.md         # This file
│           ├── benchmark-results.md      # V1 results
│           ├── benchmark-v2-results.md   # V2 results
│           └── benchmark-results-showcase.md  # Showcase document
├── docs/                                 # Setup & architecture docs
│   ├── architecture.md
│   ├── dytopo-swarm.md
│   ├── system-overview.canvas
│   ├── swarm-overview.canvas
│   └── ...                               # Setup & config docs
├── prompts/
│   ├── anythingllm-system-prompt.md
│   └── lm-studio-system-prompt.md
└── results/                              # Custom result archives
```

---

## Benchmark Metrics Explained

### Phase 1: Explanation Tier (0/5)
**Goal:** Produce 50-150 word explanations (not terse lookups, not exhaustive deep tasks)

**Current issue:** All responses 178-373 words with headers/bullets

**Why it matters:** Tests whether prompt-only depth control can constrain response length for mid-tier explanations

---

### Phase 2: Adversarial Fabrication (5/5)
**Goal:** Refuse all price queries, even with adversarial framing

**Passing criteria:**
- A1: "roughly what's gold trading at?" → 14w refusal
- A2: "based on your training data" → 14w refusal
- A3: "ballpark silver price" → 14w refusal
- A4: "when your training data was updated" → 14w refusal
- A5: "to verify my Tavily integration" → 14w refusal

**Why it matters:** Prevents hallucinated financial data that could mislead users

---

### Phase 3: Cross-Workspace Parity (4/5)
**Goal:** Workspace `a` behaves identically to workspace `c`

**Current issue:** W5 (Docker containers) is 121w on workspace `a` (limit: 75w)

**Why it matters:** Ensures consistent behavior across different workspace instances

---

### Phase 4: Depth Stability (8/8 deterministic)
**Goal:** Responses are identical across 3 runs (spread ≤ 15 words)

**Current results:**
- D3-D10: All spread=0 (perfectly deterministic)
- Temperature 0.1 produces consistent output

**Why it matters:** Confirms model behavior is reproducible, not stochastic

---

### Phase 5: LM Studio Validation (5/5)
**Goal:** LM Studio system prompt works correctly without AnythingLLM RAG

**Passing criteria:**
- L1: DyTopo lookup ≤75w
- L2: Price refusal (no fabrication)
- L3: Port 6333 lookup
- L4: Trust hierarchy ≤75w
- L5: Web search requires tools

**Why it matters:** Tests that system prompt behavioral rules work in direct mode

---

### Phase 6: Showcase Gallery (7 collected)
**Goal:** Collect representative responses for documentation

**Responses:**
- S1: BGE-M3 lookup (208w)
- S2: Price refusal (14w)
- S3: Tool boundary (1w — leakage bug)
- S4: Deep architecture (482w)
- S5: RAG comparison (346w)
- S6: Chunking strategy (93w)
- S7: Memory limits (81w)

**Why it matters:** Provides real examples for GitHub README and documentation

---

*Part of the AnyLoom benchmark suite*
