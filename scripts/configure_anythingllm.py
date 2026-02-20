#!/usr/bin/env python3
"""Configure AnythingLLM to use the AnyLoom llama.cpp Docker stack.

Sets system-level defaults (LLM provider, vector DB, embedding),
creates an 'AnyLoom' workspace with the tuned system prompt,
uploads and embeds RAG documents, and runs a smoke-test query.

Handles fresh installs: opens the browser for setup wizard / API key
generation, verifies the embedding server context size, and catches
per-document embedding errors with actionable diagnostics.

Usage:
    python scripts/configure_anythingllm.py

    # Skip interactive prompts (CI / re-run):
    ANYTHINGLLM_API_KEY=xxx python scripts/configure_anythingllm.py
"""
import glob
import json
import os
import re
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ANYTHINGLLM_BASE = "http://localhost:3001/api/v1"
ANYTHINGLLM_UI = "http://localhost:3001"
LLM_ENDPOINT = "http://localhost:8008/v1"
EMBEDDING_HOST = "http://localhost:8009"
EMBEDDING_ENDPOINT = f"{EMBEDDING_HOST}/v1"
QDRANT_ENDPOINT = "http://localhost:6333"

# Docker-internal addresses (AnythingLLM container -> sibling containers)
DOCKER_LLM_BASE = "http://anyloom-llm:8080/v1"
DOCKER_EMBEDDING_BASE = "http://anyloom-embedding:8080/v1"
DOCKER_QDRANT_URL = "http://anyloom-qdrant:6333"

WORKSPACE_NAME = "AnyLoom"
SYSTEM_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "prompts", "anythingllm-system-prompt-laptop.md"
)
RAG_DOCS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "rag-docs", "anythingllm-laptop")
)
ENV_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", ".env")
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def open_in_browser(url, label=""):
    """Open a URL in the default browser."""
    try:
        webbrowser.open(url)
        if label:
            print(f"  Opened {label} in browser: {url}")
        else:
            print(f"  Opened browser: {url}")
    except Exception as e:
        print(f"  Could not open browser ({e}) — open manually: {url}")


def wait_for_service(url, name, timeout=120, interval=5):
    """Poll a URL until it responds (HTTP status < 500)."""
    print(f"  Waiting for {name}...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code < 500:
                print(" ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(interval)
    print(f" timeout after {timeout}s!")
    return False


def get_api_key():
    """Get AnythingLLM API key from env, .env file, or interactive prompt."""
    # 1. Environment variable
    key = os.environ.get("ANYTHINGLLM_API_KEY", "").strip()
    if key:
        print("  Using ANYTHINGLLM_API_KEY from environment")
        return key

    # 2. .env file
    if os.path.isfile(ENV_PATH):
        with open(ENV_PATH, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if stripped.startswith("ANYTHINGLLM_API_KEY="):
                    val = stripped.split("=", 1)[1].strip().strip('"').strip("'")
                    if val and val != "your_key_here":
                        print("  Using ANYTHINGLLM_API_KEY from .env")
                        return val

    # 3. Interactive — open browser and ask
    print("\n  No API key found in environment or .env file.")
    print("  Opening AnythingLLM in your browser...\n")
    print("  If this is a FRESH INSTALL:")
    print("    1. Complete the setup wizard (pick any defaults — we override them)")
    print("    2. Click the gear icon (Settings) > Developer")
    print("    3. Click 'Generate New API Key' and copy it")
    print("")
    print("  If you ALREADY completed setup:")
    print("    1. Go to Settings > Developer > copy your API key")
    print("")
    open_in_browser(ANYTHINGLLM_UI, "AnythingLLM")

    key = input("  Paste API Key: ").strip()
    if not key:
        print("\n[FAIL] No API key provided.")
        sys.exit(1)

    # Save to .env for next time
    _save_api_key_to_env(key)
    return key


def _save_api_key_to_env(key):
    """Persist the API key to .env so future runs find it automatically."""
    if not os.path.isfile(ENV_PATH):
        return
    try:
        with open(ENV_PATH, "r") as f:
            lines = f.readlines()

        updated = False
        for i, line in enumerate(lines):
            if "ANYTHINGLLM_API_KEY" in line:
                lines[i] = f"ANYTHINGLLM_API_KEY={key}\n"
                updated = True
                break

        if updated:
            with open(ENV_PATH, "w") as f:
                f.writelines(lines)
            print(f"  Saved API key to .env (future runs will use it automatically)")
    except Exception as e:
        print(f"  [WARN] Could not save to .env: {e}")


def hdrs(api_key):
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def load_system_prompt():
    """Load the system prompt from the prompts directory (cached)."""
    if not hasattr(load_system_prompt, "_cache"):
        prompt_path = os.path.normpath(SYSTEM_PROMPT_PATH)
        if os.path.isfile(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                load_system_prompt._cache = f.read()
        else:
            print(f"  [WARN] System prompt not found at {prompt_path}")
            load_system_prompt._cache = ""
    return load_system_prompt._cache


# ---------------------------------------------------------------------------
# Service checks
# ---------------------------------------------------------------------------
def check_llm():
    """Check if llama.cpp is reachable on the host port."""
    try:
        r = requests.get(f"{LLM_ENDPOINT}/models", timeout=5)
        r.raise_for_status()
        models = r.json().get("data", [])
        if models:
            print(f"  [OK] llama.cpp running — model: {models[0].get('id')}")
            return True
        print("  [WARN] llama.cpp running but no models loaded")
        return False
    except Exception as e:
        print(f"  [FAIL] llama.cpp not accessible: {e}")
        return False


def check_embedding():
    """Check if the BGE-M3 embedding server is reachable and verify context size."""
    try:
        r = requests.get(f"{EMBEDDING_ENDPOINT}/models", timeout=5)
        r.raise_for_status()
        models = r.json().get("data", [])
        if not models:
            print("  [WARN] Embedding server running but no models loaded")
            return False, 0
        print(f"  [OK] Embedding server running — model: {models[0].get('id')}")
    except Exception as e:
        print(f"  [FAIL] Embedding server not accessible: {e}")
        return False, 0

    # Verify per-slot context size via llama.cpp /slots endpoint
    ctx_per_slot = 0
    try:
        r = requests.get(f"{EMBEDDING_HOST}/slots", timeout=5)
        if r.status_code == 200:
            slots = r.json()
            if slots:
                ctx_per_slot = slots[0].get("n_ctx", 0)
                n_slots = len(slots)
                print(f"         {n_slots} slot(s), {ctx_per_slot} tokens/slot")
                if ctx_per_slot < 8192:
                    print(f"  [WARN] Context per slot ({ctx_per_slot}) < BGE-M3 max (8192)")
                    print(f"         Large chunks will fail to embed!")
                    print(f"         Fix: in docker-compose.yml, set --ctx-size = 8192 * --parallel")
                    print(f"         Currently need: --ctx-size {8192 * n_slots} --parallel {n_slots}")
    except Exception:
        pass  # /slots may not exist in all llama.cpp versions

    return True, ctx_per_slot


def check_qdrant():
    """Check if Qdrant is reachable on the host port."""
    try:
        r = requests.get(f"{QDRANT_ENDPOINT}/collections", timeout=5)
        r.raise_for_status()
        collections = r.json().get("result", {}).get("collections", [])
        print(f"  [OK] Qdrant running — {len(collections)} collection(s)")
        return True
    except Exception as e:
        print(f"  [FAIL] Qdrant not accessible: {e}")
        return False


def check_anythingllm(api_key):
    """Check if AnythingLLM API is reachable and the key is valid."""
    try:
        r = requests.get(
            f"{ANYTHINGLLM_BASE}/system",
            headers=hdrs(api_key),
            timeout=5,
        )
        r.raise_for_status()
        print("  [OK] AnythingLLM API accessible")
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            print("  [FAIL] 403 — invalid API key or setup wizard not completed.")
            print("         Opening AnythingLLM in your browser...")
            open_in_browser(ANYTHINGLLM_UI, "AnythingLLM")
            print("    1. Complete the setup wizard if prompted")
            print("    2. Go to Settings > Developer and generate a new API key")
            print("    3. Re-run this script with the new key")
        else:
            print(f"  [FAIL] AnythingLLM error: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] AnythingLLM not accessible: {e}")
        return False


# ---------------------------------------------------------------------------
# System-level configuration
# ---------------------------------------------------------------------------
def configure_system_defaults(api_key):
    """Set system-wide LLM provider, vector DB, embedding, chunking, and
    default system prompt via /system/update-env.

    IMPORTANT: All values MUST be strings. AnythingLLM's updateENV.js calls
    .includes() on every value, which crashes on integers/booleans.
    """
    env_updates = {
        # LLM — Generic OpenAI pointing at llama.cpp inside Docker
        "LLMProvider": "generic-openai",
        "GenericOpenAiBasePath": DOCKER_LLM_BASE,
        "GenericOpenAiModelPref": "gpt-4",  # Matches llama.cpp --alias; avoids tiktoken errors
        "GenericOpenAiKey": "not-needed",       # llama.cpp ignores auth but field must be non-empty
        "GenericOpenAiTokenLimit": "16384",      # context window — must be string (16K for @agent headroom)
        "GenericOpenAiMaxTokens": "1024",       # max generation tokens per response
        # Vector DB — Qdrant inside Docker
        "VectorDB": "qdrant",
        "QdrantEndpoint": DOCKER_QDRANT_URL,
        "QdrantApiKey": "",
        # Embedding — BGE-M3 via llama.cpp (GPU container)
        "EmbeddingEngine": "generic-openai",
        "EmbeddingBasePath": DOCKER_EMBEDDING_BASE,
        "EmbeddingModelPref": "bge-m3-q8_0",
        "GenericOpenAiEmbeddingApiKey": "not-needed",
        # Chunk size — this is the ONLY setting AnythingLLM respects for splitting.
        # Despite the name, it controls max chunk size for text splitting.
        # 2500 chars (~625 tokens) is optimal for BGE-M3 retrieval precision.
        # NOTE: TextSplitterChunkSize/ChunkOverlap keys are silently ignored
        # by AnythingLLM's update-env API — do NOT add them here.
        "EmbeddingModelMaxChunkLength": "2500",
    }

    # Add default system prompt so new workspaces inherit it
    system_prompt = load_system_prompt()
    if system_prompt:
        env_updates["DefaultSystemPrompt"] = system_prompt

    print("\n  Applying system defaults:")
    for k, v in env_updates.items():
        display = v if len(str(v)) < 60 else f"{str(v)[:57]}..."
        print(f"    {k}: {display}")

    r = requests.post(
        f"{ANYTHINGLLM_BASE}/system/update-env",
        headers=hdrs(api_key),
        json=env_updates,
    )
    r.raise_for_status()
    body = r.json()
    if body.get("error"):
        print(f"  [FAIL] update-env error: {body['error']}")
        return False

    # Show which keys were accepted
    accepted = body.get("newValues", {})
    if accepted:
        print(f"  Accepted: {', '.join(sorted(accepted.keys()))}")

    # Verify critical settings took effect
    r2 = requests.get(f"{ANYTHINGLLM_BASE}/system", headers=hdrs(api_key), timeout=5)
    r2.raise_for_status()
    settings = r2.json().get("settings", {})
    checks = {
        "LLMProvider": "generic-openai",
        "VectorDB": "qdrant",
        "EmbeddingEngine": "generic-openai",
        "EmbeddingBasePath": DOCKER_EMBEDDING_BASE,
        "EmbeddingModelPref": "bge-m3-q8_0",
        "GenericOpenAiBasePath": DOCKER_LLM_BASE,
        "GenericOpenAiModelPref": "gpt-4",  # Matches llama.cpp --alias; avoids tiktoken errors
        "QdrantEndpoint": DOCKER_QDRANT_URL,
    }
    failed = []
    for key, expected in checks.items():
        actual = str(settings.get(key, ""))
        if actual != expected:
            failed.append(f"{key}: expected '{expected}', got '{actual}'")

    if failed:
        print("  [FAIL] Verification failed:")
        for msg in failed:
            print(f"    {msg}")
        return False
    print("  [OK] System defaults applied and verified")
    return True


# ---------------------------------------------------------------------------
# Workspace creation + update
# ---------------------------------------------------------------------------
def find_workspace(api_key, name):
    """Return the workspace dict if a workspace with this name exists, else None."""
    r = requests.get(f"{ANYTHINGLLM_BASE}/workspaces", headers=hdrs(api_key))
    r.raise_for_status()
    for ws in r.json().get("workspaces", []):
        if ws.get("name", "").lower() == name.lower():
            return ws
    return None


def create_or_get_workspace(api_key):
    """Create the AnyLoom workspace if it doesn't exist, return slug."""
    existing = find_workspace(api_key, WORKSPACE_NAME)
    if existing:
        slug = existing.get("slug")
        print(f"  [OK] Workspace '{WORKSPACE_NAME}' already exists (slug: {slug})")
        return slug

    r = requests.post(
        f"{ANYTHINGLLM_BASE}/workspace/new",
        headers=hdrs(api_key),
        json={"name": WORKSPACE_NAME},
    )
    r.raise_for_status()
    ws = r.json().get("workspace", {})
    slug = ws.get("slug", "")
    if not slug:
        print(f"  [FAIL] Workspace creation returned no slug: {r.json()}")
        return None
    print(f"  [OK] Created workspace '{WORKSPACE_NAME}' (slug: {slug})")
    return slug


def update_workspace(api_key, slug):
    """Push system prompt and tuning parameters to the workspace."""
    prompt_text = load_system_prompt()

    settings = {
        "openAiTemp": 0.5,
        "openAiHistory": 1,
        "topN": 4,
        "similarityThreshold": 0.25,
        "chatMode": "chat",
        "queryRefusalResponse": (
            "There is no relevant information in this workspace "
            "to answer your query."
        ),
        # Agent provider — must be set explicitly on the workspace so @agent
        # mode knows which LLM to use (workspace.agentProvider takes priority
        # over system LLM_PROVIDER in AnythingLLM's agent resolution order).
        "agentProvider": "generic-openai",
        "agentModel": "gpt-4",  # llama.cpp --alias
    }
    if prompt_text:
        settings["openAiPrompt"] = prompt_text

    print("  Pushing workspace settings:")
    for k, v in settings.items():
        if k == "openAiPrompt":
            print(f"    {k}: {len(v)} chars")
        else:
            print(f"    {k}: {v}")

    r = requests.post(
        f"{ANYTHINGLLM_BASE}/workspace/{slug}/update",
        headers=hdrs(api_key),
        json=settings,
    )
    r.raise_for_status()
    ws = r.json().get("workspace", {})
    actual_prompt = ws.get("openAiPrompt", "")
    if prompt_text and len(actual_prompt) < 100:
        print(f"  [FAIL] System prompt not applied (got {len(actual_prompt)} chars)")
        return False
    print(f"  [OK] Workspace updated (prompt: {len(actual_prompt)} chars)")
    return True


# ---------------------------------------------------------------------------
# RAG document upload + embedding
# ---------------------------------------------------------------------------
def list_uploaded_documents(api_key):
    """Return a set of already-uploaded document filenames (stems) from
    AnythingLLM's document store."""
    try:
        r = requests.get(
            f"{ANYTHINGLLM_BASE}/documents",
            headers=hdrs(api_key),
            timeout=15,
        )
        r.raise_for_status()
        items = r.json().get("localFiles", {}).get("items", [])
        names = set()
        for folder in items:
            for doc in folder.get("items", []):
                name = doc.get("name", "")
                if name:
                    names.add(name)
        return names
    except Exception as e:
        print(f"    [WARN] Could not list documents: {e}")
        return set()


def list_workspace_documents(api_key, slug):
    """Return a set of document names already embedded in the workspace."""
    try:
        r = requests.get(
            f"{ANYTHINGLLM_BASE}/workspace/{slug}",
            headers=hdrs(api_key),
            timeout=15,
        )
        r.raise_for_status()
        ws = r.json().get("workspace", [])
        # workspace response includes a 'documents' list
        docs = []
        if isinstance(ws, dict):
            docs = ws.get("documents", [])
        elif isinstance(ws, list):
            for item in ws:
                docs.extend(item.get("documents", []))
        names = set()
        for doc in docs:
            meta = doc.get("metadata", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            title = meta.get("title", "")
            if title:
                names.add(title)
            # Also track the docpath for idempotency
            docpath = doc.get("docpath", "")
            if docpath:
                names.add(docpath)
        return names
    except Exception as e:
        print(f"    [WARN] Could not list workspace documents: {e}")
        return set()


def upload_file(api_key, filepath):
    """Upload a single file to AnythingLLM's document store.
    Returns the document location string on success, None on failure."""
    filename = os.path.basename(filepath)
    try:
        with open(filepath, "rb") as f:
            r = requests.post(
                f"{ANYTHINGLLM_BASE}/document/upload",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (filename, f, "text/markdown")},
                timeout=60,
            )
        r.raise_for_status()
        body = r.json()
        if not body.get("success", False):
            error = body.get("error", "unknown error")
            print(f"    [FAIL] Upload {filename}: {error}")
            return None
        docs = body.get("documents", [])
        if docs:
            location = docs[0].get("location", "")
            print(f"    [OK] Uploaded {filename} -> {location}")
            return location
        print(f"    [FAIL] Upload {filename}: no document location returned")
        return None
    except Exception as e:
        print(f"    [FAIL] Upload {filename}: {e}")
        return None


def embed_documents(api_key, slug, locations, ctx_per_slot=0):
    """Embed a list of document locations into the workspace.
    Returns True on full success, False if any documents failed.

    Parses AnythingLLM error responses to report per-document failures
    and suggests fixes for common issues (context size, batch size).
    """
    if not locations:
        print("    [OK] No new documents to embed")
        return True
    try:
        r = requests.post(
            f"{ANYTHINGLLM_BASE}/workspace/{slug}/update-embeddings",
            headers=hdrs(api_key),
            json={"adds": locations, "deletes": []},
            timeout=300,
        )
        r.raise_for_status()
        body = r.json()

        # Check for per-document embedding errors
        error_msg = body.get("error", "")
        if error_msg:
            # Parse failure count
            fail_match = re.search(r"(\d+) documents? failed", error_msg)
            fail_count = int(fail_match.group(1)) if fail_match else 0
            success_count = len(locations) - fail_count

            print(f"\n    [PARTIAL] {success_count}/{len(locations)} embedded, "
                  f"{fail_count} failed:")

            # Extract individual error messages
            # Format: "GenericOpenAI Failed to embed: 400 input (N tokens)..."
            errors = re.findall(
                r"Failed to embed: (\d+) (.+?)(?=\s*GenericOpenAI Failed|$)",
                error_msg,
            )
            seen = set()
            for status_code, msg in errors:
                # Deduplicate similar errors (same status + similar message)
                short = f"{status_code}:{msg[:60]}"
                if short not in seen:
                    seen.add(short)
                    print(f"      HTTP {status_code}: {msg.strip()}")

            # Actionable fix for context/batch size errors
            if any(kw in error_msg.lower()
                   for kw in ["too large", "context size", "batch size"]):
                # Extract token counts from error messages
                token_counts = [int(t) for t in re.findall(r"\((\d+) tokens?\)", error_msg)]
                max_tokens = max(token_counts) if token_counts else 0
                print(f"\n    [FIX] {fail_count} chunk(s) exceed the embedding server limit.")
                if ctx_per_slot:
                    print(f"           Server allows {ctx_per_slot} tokens/slot, "
                          f"largest chunk is {max_tokens} tokens.")
                if max_tokens > 8192:
                    print(f"           Chunks are larger than BGE-M3's 8192 max — "
                          f"reduce EmbeddingModelMaxChunkLength.")
                    print(f"           Current: 2500 chars. The failing chunks may "
                          f"predate this setting.")
                    print(f"           Fix: delete documents in AnythingLLM UI, "
                          f"then re-run this script.")
                else:
                    print(f"           Fix: in docker-compose.yml embedding service,")
                    print(f"           set --ctx-size >= {max_tokens} * --parallel N,")
                    print(f"           then: docker compose up -d embedding")
            return False

        ws = body.get("workspace", {})
        if ws:
            print(f"    [OK] Embedded {len(locations)} document(s) into workspace")
            return True
        print(f"    [FAIL] Embedding failed: unexpected response")
        return False
    except requests.exceptions.Timeout:
        print(f"    [FAIL] Embedding timed out after 300s")
        print(f"           Documents may be very large or the server is overloaded.")
        return False
    except Exception as e:
        print(f"    [FAIL] Embedding failed: {e}")
        return False


def upload_and_embed_rag_docs(api_key, slug, ctx_per_slot=0):
    """Scan rag-docs/anythingllm/*.md, upload each file, and embed into
    the workspace.  Idempotent — skips files already in the document store
    and already embedded in the workspace."""
    if not os.path.isdir(RAG_DOCS_DIR):
        print(f"  [WARN] RAG docs directory not found: {RAG_DOCS_DIR}")
        return False

    md_files = sorted(glob.glob(os.path.join(RAG_DOCS_DIR, "*.md")))
    if not md_files:
        print(f"  [WARN] No .md files found in {RAG_DOCS_DIR}")
        return False
    print(f"  Found {len(md_files)} RAG document(s) in {RAG_DOCS_DIR}")

    # --- Check what's already uploaded ---
    existing_docs = list_uploaded_documents(api_key)
    ws_docs = list_workspace_documents(api_key, slug)
    print(f"  Document store has {len(existing_docs)} file(s); "
          f"workspace has {len(ws_docs)} embedded doc(s)")

    # --- Upload phase ---
    print("\n  Uploading documents...")
    locations_to_embed = []
    all_ok = True
    for filepath in md_files:
        filename = os.path.basename(filepath)
        stem = os.path.splitext(filename)[0]
        # Check if already uploaded (collector turns name.md into name-<hash>.json)
        already_uploaded = any(stem in name for name in existing_docs)
        if already_uploaded:
            # Still need the location for embedding — find it from doc list
            matching = [n for n in existing_docs if stem in n]
            if matching:
                # Build the expected location path
                location = f"custom-documents/{matching[0]}"
                # Check if already embedded
                already_embedded = any(stem in d for d in ws_docs)
                if already_embedded:
                    print(f"    [SKIP] {filename} (already uploaded and embedded)")
                else:
                    print(f"    [SKIP] {filename} (already uploaded, will embed)")
                    locations_to_embed.append(location)
            else:
                print(f"    [SKIP] {filename} (already uploaded)")
            continue
        # Upload the file
        location = upload_file(api_key, filepath)
        if location:
            locations_to_embed.append(location)
        else:
            all_ok = False
        # Small delay to avoid overwhelming the collector
        time.sleep(0.5)

    # --- Embed phase ---
    if locations_to_embed:
        print(f"\n  Embedding {len(locations_to_embed)} document(s) "
              f"into workspace '{slug}'...")
        if not embed_documents(api_key, slug, locations_to_embed, ctx_per_slot):
            all_ok = False
    else:
        print("\n  All documents already embedded — nothing to do")

    if all_ok:
        print(f"  [OK] RAG documents ready ({len(md_files)} files)")
    else:
        print(f"  [WARN] Some documents failed — check messages above")
    return all_ok


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def smoke_test(api_key, slug):
    """Send a query and validate response quality (not just non-empty).

    Quality checks:
    1. Length bounds: 50-1500 chars (not too short, not runaway)
    2. Repetition detection: any 3-gram appearing >3 times => FAIL
    3. Keyword check: must contain at least one relevant keyword
    """
    print("\n  Sending smoke-test query: 'What is AnyLoom?'")
    try:
        r = requests.post(
            f"{ANYTHINGLLM_BASE}/workspace/{slug}/chat",
            headers=hdrs(api_key),
            json={"message": "What is AnyLoom?", "mode": "chat"},
            timeout=300,
        )
        r.raise_for_status()
        text = r.json().get("textResponse", "")
    except Exception as e:
        print(f"  [FAIL] Query failed: {e}")
        return False

    if not text:
        print("  [FAIL] Empty response")
        return False

    preview = text[:200].replace("\n", " ")
    print(f"  Response ({len(text)} chars): {preview}...")

    # --- Quality gate ---
    passed = True

    # 1. Length bounds
    if len(text) < 50:
        print(f"  [FAIL] Response too short ({len(text)} chars, min 50)")
        passed = False
    elif len(text) > 1500:
        print(f"  [WARN] Response very long ({len(text)} chars) — may indicate runaway generation")

    # 2. Repetition detection (3-gram frequency)
    words = text.lower().split()
    if len(words) >= 6:
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        from collections import Counter
        trigram_counts = Counter(trigrams)
        worst_trigram, worst_count = trigram_counts.most_common(1)[0]
        if worst_count > 3:
            print(f"  [FAIL] Degenerate repetition detected: "
                  f"'{worst_trigram}' appears {worst_count} times")
            passed = False
        else:
            print(f"  [OK] No degenerate repetition (worst 3-gram: {worst_count}x)")

    # 3. Keyword check — response should mention something relevant
    keywords = ["anyloom", "swarm", "agent", "workspace", "qdrant",
                "llama", "docker", "loom", "dytopo", "rag", "vector",
                "inference", "model", "embedding"]
    text_lower = text.lower()
    found = [kw for kw in keywords if kw in text_lower]
    if found:
        print(f"  [OK] Relevant keywords found: {', '.join(found[:5])}")
    else:
        print(f"  [FAIL] No relevant keywords found in response")
        passed = False

    if passed:
        print(f"  [OK] Smoke test PASSED")
    else:
        print(f"  [FAIL] Smoke test FAILED — see details above")
    return passed


# ---------------------------------------------------------------------------
# MCP & Skills
# ---------------------------------------------------------------------------
CONTAINER_NAME = "anyloom-anythingllm"
MCP_CONFIG_PATH = Path(__file__).parent.parent / "config" / "anythingllm_mcp_servers.json"
CONTAINER_MCP_DEST = "/app/server/storage/plugins/anythingllm_mcp_servers.json"
SKILLS_SRC_DIR = Path(__file__).parent.parent / "skills"


def install_skills():
    """Copy custom agent skills from the repo's skills/ directory into the
    AnythingLLM container and install their npm dependencies.

    Each subdirectory of skills/ that contains both ``plugin.json`` and
    ``handler.js`` is treated as a valid skill and copied via ``docker cp``.
    After all skills are copied, the required npm packages are installed
    inside the container.

    Returns True on success, False on failure.
    """
    if not SKILLS_SRC_DIR.is_dir():
        print(f"  [SKIP] Skills source directory not found: {SKILLS_SRC_DIR}")
        return False

    skills_dest = "/app/server/storage/plugins/agent-skills"
    installed = 0

    # --- Copy each valid skill directory into the container ---
    for skill_path in sorted(SKILLS_SRC_DIR.iterdir()):
        if not skill_path.is_dir():
            continue
        if not (skill_path / "plugin.json").is_file():
            continue
        if not (skill_path / "handler.js").is_file():
            continue

        skill_name = skill_path.name
        print(f"  Installing {skill_name}...")
        try:
            # Remove existing skill dir first to avoid docker cp nesting
            subprocess.run(
                ["docker", "exec", CONTAINER_NAME, "rm", "-rf",
                 f"{skills_dest}/{skill_name}"],
                capture_output=True, text=True, timeout=15,
            )
            result = subprocess.run(
                ["docker", "cp", str(skill_path),
                 f"{CONTAINER_NAME}:{skills_dest}/{skill_name}"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                print(f"    [FAIL] docker cp: {result.stderr.strip()}")
                continue
            installed += 1
        except FileNotFoundError:
            print("  [FAIL] 'docker' command not found — is Docker installed?")
            return False
        except subprocess.TimeoutExpired:
            print(f"    [FAIL] docker cp timed out for {skill_name}")
            continue

    if installed == 0:
        print("  [SKIP] No valid skills found to install")
        return False

    # --- Install npm dependencies inside the container ---
    print("  Installing npm dependencies (this may take a moment)...")
    try:
        result = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "npm", "install",
             "--prefix", "/app/server", "--legacy-peer-deps",
             "yahoo-finance2@3", "defuddle", "jsdom",
             "@mozilla/readability", "turndown"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  [WARN] npm install returned non-zero: {result.stderr.strip()}")
        else:
            print("  npm dependencies updated")
    except FileNotFoundError:
        print("  [FAIL] 'docker' command not found — is Docker installed?")
        return False
    except subprocess.TimeoutExpired:
        print("  [WARN] npm install timed out after 120s — dependencies may be incomplete")

    print(f"  [OK] Installed {installed} skill(s), npm dependencies updated")
    return True


def configure_mcp():
    """Copy the local MCP server configuration into the AnythingLLM container.

    Reads config/anythingllm_mcp_servers.json from the repo and copies it to
    the container's plugin storage so AnythingLLM picks up the MCP servers.
    """
    # 1. Check local config file exists
    if not MCP_CONFIG_PATH.is_file():
        print(f"  [FAIL] MCP config not found: {MCP_CONFIG_PATH}")
        return False

    # Validate JSON before copying
    try:
        with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        server_count = len(config.get("mcpServers", {}))
        print(f"  Found {server_count} MCP server(s) in local config")
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [FAIL] Could not read MCP config: {e}")
        return False

    # 2. Copy into container via docker cp
    src = str(MCP_CONFIG_PATH)
    dest = f"{CONTAINER_NAME}:{CONTAINER_MCP_DEST}"
    try:
        result = subprocess.run(
            ["docker", "cp", src, dest],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            print(f"  [FAIL] docker cp failed: {stderr}")
            return False
    except FileNotFoundError:
        print("  [FAIL] 'docker' command not found — is Docker installed?")
        return False
    except subprocess.TimeoutExpired:
        print("  [FAIL] docker cp timed out")
        return False

    # 3. Verify by reading it back from the container
    try:
        result = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "cat", CONTAINER_MCP_DEST],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            print(f"  [FAIL] Verification failed: {result.stderr.strip()}")
            return False
        # Parse the content to confirm it's valid
        remote_config = json.loads(result.stdout)
        remote_count = len(remote_config.get("mcpServers", {}))
        if remote_count != server_count:
            print(f"  [FAIL] Server count mismatch: local={server_count}, "
                  f"container={remote_count}")
            return False
    except (json.JSONDecodeError, subprocess.TimeoutExpired) as e:
        print(f"  [FAIL] Verification error: {e}")
        return False

    server_names = ", ".join(config.get("mcpServers", {}).keys())
    print(f"  [OK] MCP config deployed to container ({server_names})")
    return True


def verify_skills():
    """Check which agent skills are installed in the AnythingLLM container.

    Lists the contents of the agent-skills plugin directory and reports
    which skills are available.
    """
    skills_dir = "/app/server/storage/plugins/agent-skills/"
    try:
        result = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "ls", skills_dir],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "No such file or directory" in stderr:
                print("  [WARN] Agent skills directory does not exist yet")
                print(f"         ({skills_dir})")
                return False
            print(f"  [FAIL] Could not list skills: {stderr}")
            return False

        entries = [e.strip() for e in result.stdout.strip().split("\n") if e.strip()]
        if not entries:
            print("  [INFO] No agent skills installed yet")
            return False

        print(f"  Found {len(entries)} agent skill(s):")
        for entry in entries:
            print(f"    - {entry}")
        return True

    except FileNotFoundError:
        print("  [FAIL] 'docker' command not found — is Docker installed?")
        return False
    except subprocess.TimeoutExpired:
        print("  [FAIL] docker exec timed out — is the container running?")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  AnyLoom — AnythingLLM Configuration")
    print("=" * 60)

    # 0 — Wait for services to be reachable
    print("\n[0/7] Waiting for services...")
    if not wait_for_service(f"{ANYTHINGLLM_UI}/api/system", "AnythingLLM", timeout=90):
        print("\n[FAIL] AnythingLLM not reachable. Is Docker running?")
        print("  Start with: docker compose up -d")
        sys.exit(1)

    # 1 — API key
    print("\n[1/7] API key...")
    api_key = get_api_key()

    # 2 — Service checks
    print("\n[2/7] Checking services...")
    llm_ok = check_llm()
    embed_ok, ctx_per_slot = check_embedding()
    check_qdrant()
    allm_ok = check_anythingllm(api_key)
    if not allm_ok:
        print("\n[FAIL] Cannot reach AnythingLLM API. Fix the issue above and retry.")
        sys.exit(1)

    # 3 — System defaults (LLM, vector DB, embedding, chunking, prompt)
    #     This effectively bypasses the setup wizard — all settings are
    #     pushed via API regardless of wizard state.
    print("\n[3/7] Setting system defaults...")
    if not configure_system_defaults(api_key):
        print("\n[FAIL] Could not apply system defaults.")
        sys.exit(1)

    # 4 — Workspace (create if needed)
    print("\n[4/7] Creating workspace...")
    slug = create_or_get_workspace(api_key)
    if not slug:
        print("\n[FAIL] Could not create workspace.")
        sys.exit(1)

    # 5 — Upload & embed RAG documents
    print("\n[5/7] Uploading RAG documents...")
    if embed_ok:
        upload_and_embed_rag_docs(api_key, slug, ctx_per_slot)
    else:
        print("  [SKIP] Embedding server not available — skipping RAG upload")
        print("  Fix the embedding server and re-run this script.")

    # 6 — Workspace settings (always push prompt + tuning)
    print("\n[6/7] Updating workspace settings...")
    if not update_workspace(api_key, slug):
        print("\n[FAIL] Could not update workspace settings.")
        sys.exit(1)

    # 7 — Smoke test
    print("\n[7/7] Smoke test...")
    if llm_ok:
        smoke_test(api_key, slug)
    else:
        print("  [SKIP] llama.cpp not running — start it and test manually")

    # MCP & Skills — optional, non-fatal
    print("\n--- MCP & Skills Configuration ---")
    try:
        install_skills()
    except Exception as e:
        print(f"  [WARN] Skill installation skipped: {e}")

    try:
        configure_mcp()
    except Exception as e:
        print(f"  [WARN] MCP configuration skipped: {e}")

    try:
        verify_skills()
    except Exception as e:
        print(f"  [WARN] Skill verification skipped: {e}")

    # Done
    print("\n" + "=" * 60)
    print("  Configuration complete!")
    print("=" * 60)
    print(f"\n  Workspace: {slug}")
    print(f"  UI:        {ANYTHINGLLM_UI}")
    if llm_ok:
        open_in_browser(ANYTHINGLLM_UI, "AnythingLLM workspace")
        print("\n  Ready to use!")
    else:
        print("\n  [WARN] Start llama.cpp before using the workspace:")
        print("    docker compose up -d llm")


if __name__ == "__main__":
    main()
