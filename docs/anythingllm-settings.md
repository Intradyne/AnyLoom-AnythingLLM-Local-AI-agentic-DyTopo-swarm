# AnythingLLM Settings

## Providers

| Provider        | Setting                | Value                              |
| --------------- | ---------------------- | ---------------------------------- |
| LLM             | Endpoint               | http://127.0.0.1:1234/v1           |
| LLM             | Model                  | Qwen3-30B-A3B-Instruct-2507 (Q6_K) |
| LLM             | Token Context Window   | 80000                              |
| Embedding       | Provider               | LM Studio                          |
| Embedding       | Endpoint               | http://127.0.0.1:1234/v1           |
| Embedding       | Model                  | ggml-org/bge-m3-Q8_0               |
| Embedding       | Max Embed Chunk Length | 32768 characters (~8192 tokens)    |
| Vector Database | Provider               | Qdrant                             |
| Vector Database | Endpoint               | http://127.0.0.1:6333              |

## Text Splitting & Chunking

| Setting | Value |
|---|---|
| Chunk Size | 6600 characters |
| Chunk Overlap | 1000 characters |

## Workspace

| Setting            | Value                                                                      |
| ------------------ | -------------------------------------------------------------------------- |
| Chat History       | 30                                                                         |
| Query Mode Refusal | "There is no relevant information in this workspace to answer your query." |
| LLM Temp           | 0.1                                                                        |

## VectorDB

| Setting | Value |
|---|---|
| Max Context Snippets | 16 |
| Document Similarity Threshold | Low |

---

## Notes

**LLM Temperature → 0.1.** Lower than LM Studio's 0.3 because AnythingLLM's agentic workflows need maximum determinism on tool call JSON. Qwen3's hybrid thinking mode provides internal reasoning diversity. Use 0.3 for general chat workspaces.

**Token Context Window → 80000.** Matches LM Studio's context length. VRAM at 80K: ~30.5 GB of 32 GB with Q8_0 KV cache — 1.5 GB headroom. The extra 16K over the previous 64K default goes directly to supporting more RAG snippets (16, up from 12) and deeper chat history (30, up from 20).

**Embed Chunk Length → 32768.** Matches BGE-M3's 8192-token input window. Actual chunks are 6600 characters (~1650 tokens) — this setting prevents truncation. If chunks ever exceed the embedding model's context, they silently lose trailing content.

**Chunk Size → 6600 / Overlap → 1000.** 6600-character chunks (~1650 tokens) sit well inside BGE-M3's 8192-token limit. Increasing chunk size captures more context per retrieval hit but dilutes semantic precision. Current values balance context capture with accuracy.

**Max Context Snippets → 16.** Up from 12. At ~500 tokens average per snippet, 16 snippets consume ~8K tokens — affordable at 80K context. The extra 4 snippets reduce missed retrievals on multi-topic queries without meaningful impact on response quality.

**Chat History → 30.** Up from 20. At ~400 tokens per message pair, 30 messages consume ~12K tokens. Maintains conversational continuity across longer agentic sessions where the model may need to reference earlier tool results or user instructions.

**Similarity Threshold → Low.** Admits more candidate chunks at the cost of potential noise. With BGE-M3's stronger embeddings and 80K context headroom, the extra candidates are affordable and reduce missed retrievals.

**Qdrant instance (port 6333) is independent of the MCP server's Qdrant (port 6334).** AnythingLLM manages its own workspace documents, chunking, and retrieval. The MCP server manages multi-source reference documents (LM Studio docs + AnythingLLM docs) through its own pipeline. Both use BGE-M3 embeddings (GGUF dense-only for AnythingLLM, FlagEmbedding dense+sparse for MCP).

**DyTopo swarm tools are only available in LM Studio.** The `qdrant-rag` MCP server (which hosts both RAG and DyTopo tools) is configured in LM Studio's `mcp.json` only. AnythingLLM connects to the Docker MCP Gateway for its tool needs. To use DyTopo from AnythingLLM, you would need to add the `qdrant-rag` server entry to AnythingLLM's `mcp.json` — but note that DyTopo swarm calls generate significant LM Studio API traffic that may compete with AnythingLLM's own inference.

**Context budget at 80K:**

| Component                               | Tokens   |
| --------------------------------------- | -------- |
| System prompt                           | ~2K      |
| MCP tool definitions (~10 Docker tools) | ~3K      |
| RAG snippets (16 × ~500 tokens)         | ~8K      |
| Chat history (30 messages)              | ~12K     |
| **Subtotal overhead**                   | **~25K** |
| **Remaining for current exchange**      | **~55K** |



Room to increase Chat History to 40 or Max Context Snippets to 20 if needed. LM Studio's tool budget is higher (~6.9K tokens across ~15 tools including 8 from qdrant-rag) because it loads the full qdrant-rag server.



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