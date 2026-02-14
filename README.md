# AnyLoom: AnythingLLM Local AI Agentic Stack

**A fully local, multi-agent AI system powered by Qwen3-30B-A3B MoE, BGE-M3 hybrid embeddings, and dual Qdrant RAG pipelines — orchestrated via DyTopo swarm intelligence and MCP tooling.**

---

## 🌐 Overview

AnyLoom transforms **AnythingLLM** into a dynamic, self-optimizing multi-agent swarm using:

- **Hybrid RAG fusion** (dense + sparse retrieval)
- **Dual Qdrant pipelines**:
    - port`6333` → AnythingLLM (dense-only RAG)
    - port`6334` → LM Studio (hybrid dense+sparse RAG via MCP)
- **12 MCP tools**
- DyTopo swarm routing, memory, and agent coordination
- **Fully local execution** — no cloud dependencies, no data leakage
- No need to type @agent anymore, the swarm can use mcp tools
  
| Component                                          | Tokens                     |
| -------------------------------------------------- | -------------------------- |
| Total Token Budget                                 | 80k                        |
| System prompt                                      | ~2K                        |
| MCP tool definitions (~10 Docker tools)            | ~3K                        |
| RAG snippets (16 × ~500 tokens)                    | ~8K/8192 embedding         |
| Chat history (30 messages)                         | ~12K                       |
|                             **Overhead Subtotal:** | **~25K**                   |
| **Remaining for chat**                             | **~55K**  ~200k characters |
The entire RAG-prompt set fits comfortably inside the token limit (two qdrants means I could have 16k worth of RAG and never miss a prompt)

> ✅ Runs on a single GPU (optimized for RTX 5090, but functional on smaller RAM pools)

---

## 🛠️ Prerequisites

Ensure the following are installed and running:

|COMPONENT|VERSION / REQUIREMENT|
|---|---|
|**LM Studio**|Latest version (local server on `:1234`)|
|**AnythingLLM**|v1.0+ (local UI)|
|**Docker Desktop**|Running with access to containers|
|**Python**|3.12+|
|**GPU**|RTX 5090 (recommended), or any capable GPU|

> 🔐 **Note**: Memory (MCP for secrets, state, and agent persistence) is **not shared across notebooks** — each workspace maintains its own context.

---

## 🚀 Quickstart

### 1. Start Qdrant Containers

```bash

bashCopy block
# AnythingLLM RAG (dense-only)
docker run -d --name anythingllm-qdrant \
  -p 6333:6333 \
  -v qdrant_anythingllm:/qdrant/storage \
  --restart always \
  --memory=4g \
  --cpus=4 \
  qdrant/qdrant:latest

# LM Studio RAG (hybrid dense+sparse via MCP)
docker run -d --name lmstudio-qdrant \
  -p 6334:6333 \
  -v qdrant_lmstudio:/qdrant/storage \
  --restart always \
  --memory=4g \
  --cpus=4 \
  qdrant/qdrant:latest
```

> 📌 Access dashboards at:
> 
> - [http://localhost:6333](http://localhost:6333) (AnythingLLM)
> - [http://localhost:6334](http://localhost:6334) (LM Studio)

---

### 2. Load Models in LM Studio

Download the following GGUF models:

- **LLM**: `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` (Q6_K)
- **Embedding**: `ggml-org/bge-m3-Q8_0`
---

### 3. Install Dependencies

```bash

bashCopy block
pip install \
  FlagEmbedding \
  torch --index-url https://download.pytorch.org/whl/cpu \
  qdrant-client>=1.12.0 \
  mcp[cli]>=1.0.0 \
  sentence-transformers>=3.0 \
  networkx>=3.0 \
  openai>=1.40 \
  tenacity>=9.0 \
  json-repair>=0.39
```

---

### 4. Configure MCP

Copy the MCP config file to the correct location:

```bash

bashCopy block
# Replace with your actual path
cp lmstudio-mcp.json "C:\Users\User\.lmstudio\config\mcp.json"
```

> 🔧 **Important**: Update the `mcp.json` path to match your installation directory.

---

### 5. Configure AnythingLLM

In the AnythingLLM UI:

- **LLM Endpoint**: `http://127.0.0.1:1234/v1`
- **Vector DB**: `http://127.0.0.1:6333`
- **Embedding Model**: `bge-m3` (via `http://127.0.0.1:1234/v1/embeddings`)

> 📄 See [anythingllm-settings.md](anythingllm-settings.md) for full configuration.

---

### 6. Configure LM Studio

In LM Studio settings:

- **System Prompt**: Copy into the “My Models” tab (not Developer) to persist across sessions
- **JS Code Sandbox**: Disable built-in MCP server
- **RAGv1 Embedder**: Disable built-in MCP server

> 🔄 Restart LM Studio after changes to verify tool registration.

---

### 7. Verify Setup

Run these checks:

|CHECK|COMMAND / URL|EXPECTED|
|---|---|---|
|LM Studio|`http://127.0.0.1:1234/v1`|Returns JSON (model loaded)|
|Qdrant (6333)|[http://localhost:6333](http://localhost:6333)|Dashboard accessible|
|Qdrant (6334)|[http://localhost:6334](http://localhost:6334)|Dashboard accessible|
|MCP Tools|Restart LM Studio|**8 tools** should appear (5 RAG + 3 DyTopo)|
|AnythingLLM|Create workspace, embed test doc, query|Success|

> ⚠️ If containers aren’t running, you’ll get connection errors.

---

## 📚 Documentation

Full reference documentation is available in the `docs/` directory:

- `docs/01-architecture-reference.md` — Dual Qdrant, hybrid RAG, DyTopo routing
- `docs/02-mcp-tooling.md` — 12 MCP tools, memory, agent coordination
- `docs/03-agent-swarm.md` — Dynamic topology, task routing, self-optimization
- `docs/04-deployment-guide.md` — Troubleshooting, scaling, GPU tuning

> 📌 **All links in this README are verified and accessible via filesystem**.

---

## 🔄 Access & Maintenance

- **Filesystem Access**: All configuration files, logs, and models are stored locally.
- **Container Management**: Use `docker ps`, `docker logs`, and `docker stop` for diagnostics.
- **Model Updates**: Re-download GGUF models in LM Studio when needed.
- **RAG Re-indexing**: Re-embed documents via AnythingLLM or MCP CLI.

---

## 🧠 Why AnyLoom?

- **No cloud dependency** — all data stays local
- **Hybrid RAG fusion** — better recall than pure dense or sparse
- **Dynamic agent swarm** — DyTopo routes tasks to optimal agents
- **MCP-powered memory** — persistent state, secrets, and agent history
- **100% local** — ideal for privacy, compliance, and offline use

---

> ✅ **You’re now running a next-gen, fully local AI agentic stack.**  
<<<<<<< HEAD
> 🚀 Start creating, querying, and orchestrating with AnyLoom today.
=======
> 🚀 Start creating, querying, and orchestrating with AnyLoom today.
>>>>>>> 9942e327ce1dc149abe142416c07aadc36c3deec
