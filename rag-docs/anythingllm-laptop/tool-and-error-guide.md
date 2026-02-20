# Tool Reference and Error Recovery Guide

## AnythingLLM Agent Skills (available in @agent mode)

- **Web Browsing** — web search (configured in admin settings)
- **Web Scraper** — URL-to-text extraction (native, zero MCP dependency)
- **RAG** — automatic from Qdrant port 6333 (fires every query in chat mode)
- **Save File to Browser** — download generated content
- **Chart Generation** — create visual charts
- **SQL Connector** — database queries
- **asset-price** — custom skill for market price lookups
- **smart-web-reader** — custom skill for clean web page extraction

## MCP Tools (NOT available to AnythingLLM)

Desktop Commander, Filesystem, Memory, Context7, Tavily, Fetch, Playwright, Sequential Thinking, n8n, rag_search, rag_status, rag_reindex, rag_sources, rag_file_info, swarm_start, swarm_status, swarm_result.

**Shared resource:** Memory knowledge graph (both agents read/write the same graph).

When asked what tools AnythingLLM has, list ONLY the Agent Skills above. Never attribute MCP tools to AnythingLLM.

## Tool Routing (Decision Cascade)

Route queries cheapest-first:

1. **Tier 0 — Auto RAG:** Qdrant port 6333 injects chunks automatically. Zero tool calls. Check Context: section first.
2. **Tier 1 — Memory:** search_nodes for stable facts (ports, configs, decisions). Millisecond lookups.
3. **Tier 2 — Context7:** Library/API documentation. Two-step: resolve-library-id then get-library-docs.
4. **Tier 3 — External info:** Tavily (live data, mandatory for prices), Fetch (known URLs), Web Scraper (native URL-to-text).
5. **Tier 4 — Operations:** Desktop Commander (shell), Filesystem (files), Sequential Thinking (planning).
6. **Tier 5 — DyTopo swarms:** Multi-agent collaboration. Most expensive. MCP-only.

Use highest tier that answers the question. Most queries resolve at Tier 0 or 1.

## Iteration Limit (8-Round Cap)

AIbitat terminates tool-calling after 8 consecutive rounds. This is a hard limit.

**Remediation:**
- Break complex tasks into focused sub-requests, each getting a fresh 8-round budget.
- Front-load efficiency: `docker ps` shows all containers at once; `nvidia-smi` gives full GPU snapshot.
- If truncated Fetch already has the answer, deliver it immediately — don't request more.
- After 3 Fetch truncation cycles on same URL, switch to Tavily.
- Financial data sites (investing.com, bloomberg.com) are JS-heavy — use Tavily directly.

## JSON / Tool Call Failures

AIbitat provides dual-mode JSON repair (native API parsing + text-based extraction fallback).

**Common causes:**
- Extended conversation history degrades tool call quality ("JSON drift")
- Context window near capacity reduces attention on tool definitions
- repeat_penalty > 1.0 corrupts JSON by penalizing structural tokens ({, }, ", ,)

**Fixes:**
- Start a fresh conversation (resets history and KV cache)
- Be specific: "@agent use Desktop Commander to run docker ps" > "@agent check containers"
- For persistent failures: check container status, verify tool server is responding

## Connection Failures

### Port 6333 (Qdrant) — connection refused
- Run `docker ps` — check for `anyloom-qdrant` container
- If stopped: `docker start anyloom-qdrant`
- Check port conflict: `netstat -ano | findstr 6333`
- Common causes: host reboot, Docker Desktop not running, disk space exhaustion

### Port 8008 (llama.cpp) — connection refused or timeout
- Run `docker ps` — check for `anyloom-llm` container
- Check logs: `docker logs anyloom-llm`
- Check VRAM: `nvidia-smi` — usage should be ~5.0-5.5 GB on RTX 2070 Max-Q
- Timeout (not refused) usually means inference congestion, not a downed service
- Verify API: `curl http://localhost:8008/v1/models`

## System Prompt Ignored

**Causes:**
- Context compression truncated the prompt tail (Memory conventions, citation rules lost first)
- U-shaped attention: middle-section instructions get less attention in chat mode (expected)
- Training data overrides prompt conventions (e.g., snake_case instead of PascalCase)

**Fix:** Start a fresh conversation to reset context budget and restore full system prompt.

## Agent Framework (AIbitat)

Four automatic safety nets (no prompt instructions needed):
1. **8-round iteration cap** — prevents runaway loops
2. **Dual-mode JSON repair** — handles malformed tool calls
3. **Deduplication guard** — prevents identical consecutive tool calls
4. **Tool schema injection** — presents current tool definitions at session start

The system prompt deliberately omits iteration budgets, JSON format rules, and tool schemas because AIbitat handles all of these automatically.
