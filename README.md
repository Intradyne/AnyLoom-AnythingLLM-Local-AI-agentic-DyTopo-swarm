# AnyLoom

AnythingLLM Local AI agentic stack:
Qwen3-30B-A3B MoE · BGE-M3 hybrid embedding · dual Qdrant RAG · DyTopo multi-agent swarm.

- `Turn AnythingLLM into a multi-agent swarm with dynamic topology routing, hybrid RAG fusion, and 12 MCP tools ·  fully local`
- `DyTopo-powered (cutting-edge) multi-agent swarm with hybrid dense+sparse RAG, dual Qdrant pipelines, and MCP tooling for AnythingLLM + LM Studio · runs on one GPU`

# Usage: 
- LMStudio launch Qwen3-30B-A3B-Instruct-2507
- Docker launch containers
- Open AnythingLLM and enjoy powerful agentic tool use without @agent
- Note: Memory (MCP for storing secrets and more) is not shared between notebooks

## Prerequisites

- [LM Studio](https://lmstudio.ai/)
- [AnythingLLM](https://anythingllm.com/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running
- Python 3.12+
- Optimized for RTX 5090 - Still the best solution on even smaller ram pools

## Quickstart

1. **Start Qdrant containers**
   ```bash
   docker run -d --name anythingllm-qdrant \
     -p 6333:6333 -v qdrant_anythingllm:/qdrant/storage \
     --restart always --memory=4g --cpus=4 qdrant/qdrant:latest

   docker run -d --name lmstudio-qdrant \
     -p 6334:6333 -v qdrant_lmstudio:/qdrant/storage \
     --restart always --memory=4g --cpus=4 qdrant/qdrant:latest
   ```

2. **Load models in LM Studio** — download `ggml-org/bge-m3-Q8_0` and `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` (Q6_K).

3. **Install dependencies**
   ```bash
   pip install FlagEmbedding torch --index-url https://download.pytorch.org/whl/cpu
   pip install qdrant-client>=1.12.0 mcp[cli]>=1.0.0
   pip install sentence-transformers>=3.0 networkx>=3.0 openai>=1.40
   pip install tenacity>=9.0 json-repair>=0.39
   ```

4. **Copy MCP config** — place `lmstudio-mcp.json` at `C:\Users\User\.lmstudio\config\mcp.json` !!!modify the path to point at your install!!!

5. **Configure AnythingLLM** — point LLM and Embedding at `http://127.0.0.1:1234/v1`, Vector DB at `http://127.0.0.1:6333`. See [anythingllm-settings.md](docs/anythingllm-settings.md) for full settings.

6. **Apply LM Studio settings** — see [lm-studio-settings.md](docs/lm-studio-settings.md)
	1. copy system prompt into my models(not developer) so its remembered.
	2. **JS Code Sandbox** —  disabled built in mcp server
	3. ragv1 embedder — disabled built in mcp server

7. **Verify**
   - LM Studio: only qwen visible, endpoint responding on `:1234`
   - Qdrant: `http://localhost:6333` and `http://localhost:6334` returning dashboard
   - MCP: restart LM Studio, verify 8 tools register (5 RAG + 3 DyTopo)
   - AnythingLLM: create workspace, embed a test doc, query it
   - if docker containers are not running you will get an error

## Documentation

Full reference documentation lives in `docs/`:

- [architecture.md](docs/architecture.md) — system diagram, VRAM budget, port map
- [mcp-servers.md](qdrant-servers.md) — MCP config, tool inventory, env vars
- [qdrant-topology.md](docs/qdrant-topology.md) — dual Qdrant setup, collection schema, sync
- [dytopo-swarm.md](docs/dytopo-swarm.md) — multi-agent swarm architecture
- [lm-studio-settings.md](docs/lm-studio-settings.md) — LM Studio model config
- [anythingllm-settings.md](docs/anythingllm-settings.md) — AnythingLLM workspace config
- [bge-m3-embedding.md](docs/bge-m3-embedding.md) — embedding model rationale
- [qwen3-model.md](docs/qwen3-model.md) — Qwen3 MoE model details
