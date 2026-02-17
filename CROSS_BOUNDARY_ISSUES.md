# Cross-Boundary Issues — vLLM → llama.cpp Migration

**Status: ALL RESOLVED**

All cross-boundary vLLM references identified by Agent A have been cleaned up.

## Resolved Items

### src/dytopo/orchestrator.py
- Import updated from `inference.vllm_client` to `inference.llm_client`
- All comments updated to reference `llama.cpp` / `llm_client`

### src/dytopo/governance.py
- Default `backend` parameters updated to `"llama-cpp"`
- Docstrings updated to reference generic inference backend

### src/dytopo/safeguards/limits.py
- Comments updated to reference `llama-cpp`
- Default `backend` parameter updated to `"llama-cpp"`
- Backend-specific threshold logic updated: `if backend == "llama-cpp":`

### src/dytopo/documentation/generator.py
- Comment updated from vLLM to LLM server

### src/qdrant_mcp_server.py
- All comments updated to reference `llama.cpp`

### tests/test_governance.py
- All vLLM-specific test names and assertions updated to `llama-cpp`

### tests/test_safeguards.py
- All vLLM-specific test names and assertions updated to `llama-cpp`
- Generic circuit breaker tests use `backend="generic"` to avoid threshold override

## Remaining Historical References

The following files contain vLLM references in **historical context** (explaining why the migration happened, recording past benchmark results):
- `docs/llm-engine.md` — Migration rationale section
- `scripts/benchmarks/` docs — Historical benchmark results from vLLM era
- `prompts/refactor-agent-*.md` — Migration spec documents
