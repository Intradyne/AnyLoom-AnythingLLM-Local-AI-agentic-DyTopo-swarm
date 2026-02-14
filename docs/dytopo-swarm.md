# DyTopo Swarm

DyTopo (arXiv 2602.06039) dynamically constructs agent communication topology each round using semantic similarity between agent-generated descriptors.

## Round Lifecycle

1. **Manager** sets round goal (or terminates)
2. **Round 1 — broadcast:** all agents see all outputs (no routing yet)
3. **Rounds 2+ — two-phase split:**
   - Phase A: each agent generates key/query descriptors only (fast, `/no_think`, temp 0.1, 256 tokens)
   - Phase B: MiniLM embeds descriptors → cosine similarity matrix → threshold τ → directed graph → cycle breaking → topological sort
   - Phase C: agents execute in topological order with routed messages injected (temp 0.3, 4096 tokens)
4. **Convergence check:** if outputs stop changing, force termination

## Two-Phase Architecture

The two-phase split is the key correctness fix: descriptors are generated *before* routing, and work is generated *after* routing with incoming messages injected. Without this, routing would be decorative.

## Agent Domains

| Domain | Manager + Workers | Use case |
|---|---|---|
| `code` | Manager, Developer, Researcher, Tester, Designer | Code generation, debugging, algorithm design |
| `math` | Manager, ProblemParser, Solver, Verifier | Mathematical proofs, calculations |
| `general` | Manager, Analyst, Critic, Synthesizer | Open-ended analysis, multi-perspective reasoning |

## Temperature Differentiation

| Call type | Temperature | Rationale |
|---|---|---|
| Descriptor generation | 0.1 | Near-deterministic structured output for routing accuracy |
| Manager decisions | 0.1 | Consistent goal-setting and termination logic |
| Agent work output | 0.3 | Matches LM Studio default — enough diversity for reasoning |

## Error Isolation

Swarm orchestration runs via `asyncio.create_task()`. Crashes return error strings to the caller — they never propagate to the RAG event loop or affect `rag_search`. The BGE-M3 model, Qdrant client, and MiniLM model are independent singletons that DyTopo code accesses but never mutates.

## Dependencies

```bash
pip install sentence-transformers>=3.0 networkx>=3.0 openai>=1.40
pip install tenacity>=9.0 json-repair>=0.39
```

`sentence-transformers` shares torch with FlagEmbedding — no duplicate install. MiniLM-L6-v2 weights (~80 MB) auto-download from HuggingFace on first swarm.
