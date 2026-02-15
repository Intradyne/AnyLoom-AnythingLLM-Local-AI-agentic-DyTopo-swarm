"""CLI utility to view and update AnythingLLM and LM Studio settings.

Usage:
    # View current settings
    python src/settings_manager.py show anythingllm
    python src/settings_manager.py show lmstudio

    # Update AnythingLLM workspace settings
    python src/settings_manager.py set anythingllm -w c --topN 12
    python src/settings_manager.py set anythingllm -w c --temperature 0.2
    python src/settings_manager.py set anythingllm -w c --prompt "path/to/prompt.md"

    # Update LM Studio model settings
    python src/settings_manager.py set lmstudio --temperature 0.15
    python src/settings_manager.py set lmstudio --context-length 90000
    python src/settings_manager.py set lmstudio --max-tokens 4096

    # Bulk update from JSON
    python src/settings_manager.py apply anythingllm config.json

    # Export current settings to JSON
    python src/settings_manager.py export anythingllm settings_backup.json
"""
import sys, os, json, argparse, requests
from pathlib import Path

# AnythingLLM API
ANYTHINGLLM_BASE = "http://localhost:3001/api/v1"
ANYTHINGLLM_KEY = os.getenv("ANYTHINGLLM_API_KEY", "")

# LM Studio API
LMSTUDIO_BASE = "http://localhost:1234/v1"


def get_anythingllm_workspace(slug):
    """Fetch workspace settings from AnythingLLM."""
    r = requests.get(
        f"{ANYTHINGLLM_BASE}/workspace/{slug}",
        headers={"Authorization": f"Bearer {ANYTHINGLLM_KEY}"}
    )
    r.raise_for_status()
    return r.json()


def update_anythingllm_workspace(slug, updates):
    """Update workspace settings in AnythingLLM."""
    r = requests.post(
        f"{ANYTHINGLLM_BASE}/workspace/{slug}/update",
        headers={
            "Authorization": f"Bearer {ANYTHINGLLM_KEY}",
            "Content-Type": "application/json"
        },
        json=updates
    )
    r.raise_for_status()
    return r.json()


def get_lmstudio_models():
    """Fetch available models from LM Studio."""
    r = requests.get(f"{LMSTUDIO_BASE}/models")
    r.raise_for_status()
    return r.json()


def get_lmstudio_config():
    """Fetch current LM Studio server config (if available via API)."""
    # LM Studio doesn't expose server config via API, so we'll just show model info
    try:
        models = get_lmstudio_models()
        return {"models": models}
    except Exception as e:
        return {"error": str(e), "note": "LM Studio config is file-based, not API-exposed"}


def show_anythingllm(workspace):
    """Display AnythingLLM workspace settings."""
    data = get_anythingllm_workspace(workspace)
    ws = data.get("workspace", {})

    if isinstance(ws, list) and ws:
        ws = ws[0]

    print(f"\n{'=' * 60}")
    print(f"AnythingLLM Workspace: {workspace}")
    print(f"{'=' * 60}")
    print(f"  Name: {ws.get('name', 'N/A')}")
    print(f"  Slug: {ws.get('slug', 'N/A')}")
    print(f"  topN: {ws.get('topN', 'N/A')}")
    print(f"  Temperature: {ws.get('openAiTemp', 'N/A')}")
    print(f"  Chat model: {ws.get('chatModel', 'N/A')}")
    print(f"  Embedding model: {ws.get('embeddingModel', 'N/A')}")

    prompt = ws.get('openAiPrompt', '')
    if prompt:
        print(f"  System prompt: {len(prompt)} chars")
        print(f"    First 100 chars: {prompt[:100]}...")
    else:
        print(f"  System prompt: (not set)")

    print(f"{'=' * 60}\n")


def show_lmstudio():
    """Display LM Studio model info."""
    config = get_lmstudio_config()

    print(f"\n{'=' * 60}")
    print(f"LM Studio Server")
    print(f"{'=' * 60}")

    if "error" in config:
        print(f"  Error: {config['error']}")
        print(f"  Note: {config['note']}")
        print(f"\n  LM Studio settings are configured in:")
        print(f"    - Model settings UI (temperature, context, etc.)")
        print(f"    - ~/.lmstudio/config.json")
    else:
        models = config.get("models", {}).get("data", [])
        if models:
            print(f"  Loaded models:")
            for m in models:
                print(f"    - {m.get('id', 'unknown')}")
        else:
            print(f"  No models loaded")

    print(f"{'=' * 60}\n")


def set_anythingllm(workspace, **kwargs):
    """Update AnythingLLM workspace settings."""
    updates = {}

    if kwargs.get("topN") is not None:
        updates["topN"] = int(kwargs["topN"])

    if kwargs.get("temperature") is not None:
        updates["openAiTemp"] = float(kwargs["temperature"])

    if kwargs.get("prompt"):
        prompt_path = Path(kwargs["prompt"])
        if prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                updates["openAiPrompt"] = f.read()
            print(f"Loaded prompt from {prompt_path} ({len(updates['openAiPrompt'])} chars)")
        else:
            print(f"Warning: Prompt file not found: {prompt_path}")
            return

    if kwargs.get("chat_model"):
        updates["chatModel"] = kwargs["chat_model"]

    if not updates:
        print("No updates specified. Use --topN, --temperature, --prompt, or --chat-model")
        return

    print(f"Updating workspace '{workspace}' with: {list(updates.keys())}")
    result = update_anythingllm_workspace(workspace, updates)

    if result.get("workspace"):
        print("✓ Update successful")
        show_anythingllm(workspace)
    else:
        print(f"Update response: {result}")


def set_lmstudio(**kwargs):
    """Update LM Studio settings (note: most settings are UI-only)."""
    print("\nNote: LM Studio settings are primarily configured via the UI.")
    print("API-level settings (temperature, max_tokens) are per-request, not server-global.")
    print("\nTo change LM Studio settings:")
    print("  1. Open LM Studio UI")
    print("  2. Go to Local Server tab")
    print("  3. Adjust model settings (temperature, context length, etc.)")
    print("  4. Settings are saved automatically")

    if kwargs:
        print(f"\nRequested changes (not applied): {kwargs}")


def export_settings(target, workspace, output_file):
    """Export current settings to JSON file."""
    if target == "anythingllm":
        data = get_anythingllm_workspace(workspace)
    elif target == "lmstudio":
        data = get_lmstudio_config()
    else:
        print(f"Unknown target: {target}")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✓ Exported {target} settings to {output_file}")


def apply_settings(target, workspace, config_file):
    """Apply settings from JSON file."""
    with open(config_file, "r", encoding="utf-8") as f:
        updates = json.load(f)

    if target == "anythingllm":
        print(f"Applying settings to workspace '{workspace}'")
        result = update_anythingllm_workspace(workspace, updates)
        if result.get("workspace"):
            print("✓ Settings applied successfully")
            show_anythingllm(workspace)
    elif target == "lmstudio":
        print("LM Studio does not support bulk settings updates via API")
    else:
        print(f"Unknown target: {target}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage AnythingLLM and LM Studio settings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Global options
    parser.add_argument("--api-key", help="AnythingLLM API key (or set ANYTHINGLLM_API_KEY env var)")

    # Show command
    show_parser = subparsers.add_parser("show", help="Display current settings")
    show_parser.add_argument("target", choices=["anythingllm", "lmstudio"])
    show_parser.add_argument("-w", "--workspace", default="c",
                            help="Workspace slug (for AnythingLLM)")

    # Set command
    set_parser = subparsers.add_parser("set", help="Update settings")
    set_parser.add_argument("target", choices=["anythingllm", "lmstudio"])
    set_parser.add_argument("-w", "--workspace", default="c",
                           help="Workspace slug (for AnythingLLM)")
    set_parser.add_argument("--topN", type=int, help="Number of RAG context chunks")
    set_parser.add_argument("--temperature", type=float, help="Model temperature")
    set_parser.add_argument("--prompt", help="Path to system prompt file")
    set_parser.add_argument("--chat-model", help="Chat model name")
    set_parser.add_argument("--context-length", type=int, help="Context length (LM Studio)")
    set_parser.add_argument("--max-tokens", type=int, help="Max tokens (LM Studio)")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export settings to JSON")
    export_parser.add_argument("target", choices=["anythingllm", "lmstudio"])
    export_parser.add_argument("output", help="Output JSON file path")
    export_parser.add_argument("-w", "--workspace", default="c",
                              help="Workspace slug (for AnythingLLM)")

    # Apply command
    apply_parser = subparsers.add_parser("apply", help="Apply settings from JSON")
    apply_parser.add_argument("target", choices=["anythingllm", "lmstudio"])
    apply_parser.add_argument("config", help="Input JSON file path")
    apply_parser.add_argument("-w", "--workspace", default="c",
                             help="Workspace slug (for AnythingLLM)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Load API key from args, environment, or fallback
    global ANYTHINGLLM_KEY
    if args.api_key:
        ANYTHINGLLM_KEY = args.api_key
    elif not ANYTHINGLLM_KEY and args.target == "anythingllm":
        # Default to the key used in benchmarks
        ANYTHINGLLM_KEY = "92JHT3J-PMF4SGA-GT0X50Y-RMGKDT3"

    try:
        if args.command == "show":
            if args.target == "anythingllm":
                show_anythingllm(args.workspace)
            else:
                show_lmstudio()

        elif args.command == "set":
            if args.target == "anythingllm":
                set_anythingllm(
                    args.workspace,
                    topN=args.topN,
                    temperature=args.temperature,
                    prompt=args.prompt,
                    chat_model=args.chat_model
                )
            else:
                set_lmstudio(
                    temperature=args.temperature,
                    context_length=args.context_length,
                    max_tokens=args.max_tokens
                )

        elif args.command == "export":
            export_settings(args.target, args.workspace, args.output)

        elif args.command == "apply":
            apply_settings(args.target, args.workspace, args.config)

    except requests.exceptions.RequestException as e:
        print(f"\n✗ API Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
