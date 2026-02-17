# AnythingLLM Settings

## Providers

| Provider        | Setting                | Value                              |
| --------------- | ---------------------- | ---------------------------------- |
| LLM             | Endpoint               | http://anyloom-llm:8080/v1 (Docker internal; host: http://localhost:8008/v1) |
| LLM             | Model                  | Qwen3-30B-A3B-Instruct-2507        |
| LLM             | Token Context Window   | 131072                             |
| Embedding       | Provider               | Generic OpenAI (llama.cpp GPU container) |
| Embedding       | Endpoint               | http://anyloom-embedding:8080/v1 (Docker internal; host: http://localhost:8009/v1) |
| Embedding       | Model                  | bge-m3-q8_0                        |
| Embedding       | API Key                | not-needed                         |
| Embedding       | Max Embed Chunk Length | 2500 characters (~625 tokens)      |
| Vector Database | Provider               | Qdrant                             |
| Vector Database | Endpoint               | http://anyloom-qdrant:6333 (Docker internal) |

**Note:** llama.cpp runs in Docker on container port 8080 (host port 8008) and provides GPU-accelerated inference with 131K context.

## Text Splitting & Chunking

| Setting | Value | Notes |
|---|---|---|
| EmbeddingModelMaxChunkLength | 2500 characters (~625 tokens) | **Only effective chunk control** via AnythingLLM API |
| TextSplitterChunkSize | _(not functional)_ | Silently ignored by AnythingLLM's system API |
| TextSplitterChunkOverlap | _(not functional)_ | Silently ignored by AnythingLLM's system API |

## Workspace

| Setting            | Value                                                                      |
| ------------------ | -------------------------------------------------------------------------- |
| Chat History       | 30                                                                         |
| Query Mode Refusal | "There is no relevant information in this workspace to answer your query." |
| LLM Temp           | 0.3                                                                        |

## VectorDB

| Setting | Value |
|---|---|
| Max Context Snippets | 16 |
| Document Similarity Threshold | Low |

---

## Notes

**LLM Temperature → 0.1.** Lower temperature for agentic workflows — AnythingLLM's tool calling needs maximum determinism on JSON output. Qwen3's hybrid thinking mode provides internal reasoning diversity. Use 0.3 for general chat workspaces.

**Token Context Window → 131072.** Matches llama.cpp's `--ctx-size 131072`. Q4_K_M model weights are ~18.6 GiB, leaving ~12.4 GiB for KV cache on a 32GB GPU. With quantized KV cache (Q8_0 keys / Q4_0 values at ~39 KiB/token), 131K context fits comfortably. The full RAG pipeline overhead (~25K tokens) leaves ~106K remaining for the current exchange.

**EmbeddingModelMaxChunkLength → 2500.** This is the **only effective chunk size control** when configuring AnythingLLM via the system API. Despite their names, `TextSplitterChunkSize` and `TextSplitterChunkOverlap` are silently ignored by AnythingLLM's `/api/v1/system/update-env` endpoint -- they appear accepted but have no effect on actual chunking behavior. `EmbeddingModelMaxChunkLength` controls the maximum character length of each text chunk. Set to 2500 characters (~625 tokens), which is well within BGE-M3's 8192-token input window and optimized for retrieval precision -- smaller chunks produce tighter semantic matches and reduce noise in retrieved context.

**Max Context Snippets → 16.** Up from 12. At ~500 tokens average per snippet, 16 snippets consume ~8K tokens — tight but manageable at 131K context — monitor token usage if adding more snippets. The extra 4 snippets reduce missed retrievals on multi-topic queries without meaningful impact on response quality.

**Chat History → 30.** Up from 20. At ~400 tokens per message pair, 30 messages consume ~12K tokens. Maintains conversational continuity across longer agentic sessions where the model may need to reference earlier tool results or user instructions.

**Similarity Threshold → Low.** Admits more candidate chunks at the cost of potential noise. With BGE-M3's stronger embeddings and 131K context, the extra candidates are manageable but leave less room for chat and reduce missed retrievals.

**Qdrant instance (port 6333) is shared with the MCP server.** The single hybrid instance stores BGE-M3 dense vectors (1024-dim) produced by the llama.cpp embedding container (port 8009, GPU). Both AnythingLLM workspace queries and MCP RAG tool searches benefit from the improved retrieval quality. The MCP server manages multi-source indexing with source_dir payload filtering.

**DyTopo swarm tools are available via the qdrant-rag MCP server running natively on the host.** The `qdrant-rag` MCP server hosts both RAG and DyTopo tools. AnythingLLM connects to the Docker MCP Gateway for its tool needs. To use DyTopo from AnythingLLM, you would need to add the `qdrant-rag` server entry to AnythingLLM's `mcp.json` — but note that DyTopo swarm calls generate significant inference API traffic that may compete with AnythingLLM's own llama.cpp usage.

**Context budget at 131K:**

| Component                               | Tokens   |
| --------------------------------------- | -------- |
| System prompt                           | ~2K      |
| MCP tool definitions (9 Docker servers)  | ~3K      |
| RAG snippets (16 × ~500 tokens)         | ~8K      |
| Chat history (30 messages)              | ~12K     |
| **Subtotal overhead**                   | **~25K** |
| **Remaining for current exchange**      | **~106K** |
| **Total Token Budget**                  | **131K** |



With 131K context, the token budget is generous. Increasing snippets or history has minimal impact on remaining chat space.



mcp.json
```json
{
  "mcpServers": {
    "MCP_DOCKER": {
      "command": "docker",
      "args": [
        "mcp",
        "gateway",
        "run"
      ],
      "env": {
        "LOCALAPPDATA": "C:\\Users\\User\\AppData\\Local",
        "ProgramData": "C:\\ProgramData",
        "ProgramFiles": "C:\\Program Files"
      }
    }
  }
}
```