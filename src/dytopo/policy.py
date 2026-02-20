"""
DyTopo Policy Enforcer (PCAS-Lite)
===================================

Lightweight tool-call policy enforcement for swarm agents.
Evaluates file, shell, and network operations against a JSON policy file
using deny-first evaluation with path traversal prevention.
"""

from __future__ import annotations

import fnmatch
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("dytopo.policy")

_DEFAULT_POLICY_PATH = Path(__file__).resolve().parent.parent.parent / "policy.json"


@dataclass
class PolicyDecision:
    """Result of a policy check."""
    allowed: bool
    reason: str = ""
    matched_rule: str = ""
    tool_name: str = ""
    params_summary: str = ""


class PolicyEnforcer:
    """Evaluate tool requests against a JSON policy file.

    Deny-first evaluation:
    1. Check deny rules first — if ANY deny matches, block immediately
    2. Check allow rules — if path/command matches an allow pattern, permit
    3. Default deny if no allow rule matches

    Path traversal prevention: all paths are resolved to absolute before matching.
    """

    def __init__(self, policy_path: str | Path | None = None):
        """Load policy from JSON file.

        Args:
            policy_path: Path to policy.json. Defaults to project root policy.json.
        """
        self.policy_path = Path(policy_path) if policy_path else _DEFAULT_POLICY_PATH
        self.rules: dict = {}
        self.enforcement: str = "strict"
        self.log_denials: bool = True

        if self.policy_path.exists():
            with open(self.policy_path, encoding="utf-8") as f:
                data = json.load(f)
            self.rules = data.get("rules", {})
            self.enforcement = data.get("enforcement", "strict")
            self.log_denials = data.get("log_denials", True)
            logger.info(f"Policy loaded from {self.policy_path} ({len(self.rules)} rule sets)")
        else:
            logger.warning(f"Policy file not found: {self.policy_path} — all requests allowed")

    def check_tool_request(self, tool_name: str, params: dict[str, Any]) -> PolicyDecision:
        """Evaluate a tool request against policy rules.

        Args:
            tool_name: One of "file_write", "file_read", "shell_exec", "network"
            params: Tool-specific parameters:
                file_write/file_read: {"path": str}
                shell_exec: {"command": str}
                network: {"host": str, "port": int}

        Returns:
            PolicyDecision with allowed/denied status and reason.
        """
        if not self.rules:
            return PolicyDecision(allowed=True, reason="No policy loaded", tool_name=tool_name)

        rule_set = self.rules.get(tool_name)
        if rule_set is None:
            return PolicyDecision(
                allowed=False,
                reason=f"Unknown tool type: {tool_name}",
                tool_name=tool_name,
                params_summary=str(params)[:200],
            )

        if tool_name in ("file_write", "file_read"):
            return self._check_file(tool_name, params, rule_set)
        elif tool_name == "shell_exec":
            return self._check_shell(params, rule_set)
        elif tool_name == "network":
            return self._check_network(params, rule_set)

        return PolicyDecision(
            allowed=False,
            reason=f"Unhandled tool type: {tool_name}",
            tool_name=tool_name,
        )

    def enforce(self, tool_name: str, params: dict[str, Any]) -> dict | None:
        """Convenience method: returns error dict if denied, None if allowed.

        Args:
            tool_name: Tool type to check
            params: Tool parameters

        Returns:
            None if allowed, or {"error": str, "tool": str, "reason": str} if denied.
        """
        decision = self.check_tool_request(tool_name, params)
        if decision.allowed:
            return None

        if self.log_denials:
            logger.warning(f"Policy denied {tool_name}: {decision.reason} (params: {decision.params_summary})")

        return {
            "error": "policy_denied",
            "tool": tool_name,
            "reason": decision.reason,
        }

    def _resolve_path(self, raw_path: str) -> Path:
        """Resolve a path to absolute, expanding ~ and preventing traversal."""
        p = Path(raw_path).expanduser()
        try:
            return p.resolve()
        except (OSError, ValueError):
            return p

    def _match_path(self, resolved: Path, pattern: str) -> bool:
        """Check if a resolved path matches a glob pattern.

        Expands ~ in patterns, resolves relative patterns against CWD,
        and uses fnmatch for glob matching with cross-platform support.
        """
        pattern_path = Path(pattern).expanduser()
        if not pattern_path.is_absolute():
            # Resolve relative patterns against CWD so they match absolute resolved paths
            expanded_pattern = str(Path.cwd() / pattern_path)
        else:
            expanded_pattern = str(pattern_path)
        resolved_str = str(resolved)
        # Try both Windows and Unix separators for cross-platform matching
        return fnmatch.fnmatch(resolved_str, expanded_pattern) or \
               fnmatch.fnmatch(resolved_str.replace("\\", "/"), expanded_pattern.replace("\\", "/"))

    def _check_file(self, tool_name: str, params: dict, rule_set: dict) -> PolicyDecision:
        """Check file read/write operations."""
        raw_path = params.get("path", "")
        if not raw_path:
            return PolicyDecision(
                allowed=False, reason="No path provided",
                tool_name=tool_name, params_summary=str(params)[:200],
            )

        resolved = self._resolve_path(raw_path)
        summary = str(resolved)[:200]

        # Deny-first: check deny_paths
        for pattern in rule_set.get("deny_paths", []):
            if self._match_path(resolved, pattern):
                return PolicyDecision(
                    allowed=False,
                    reason=f"Path matches deny rule: {pattern}",
                    matched_rule=pattern,
                    tool_name=tool_name,
                    params_summary=summary,
                )

        # Check allow_paths
        for pattern in rule_set.get("allow_paths", []):
            if self._match_path(resolved, pattern):
                return PolicyDecision(
                    allowed=True,
                    reason=f"Path matches allow rule: {pattern}",
                    matched_rule=pattern,
                    tool_name=tool_name,
                    params_summary=summary,
                )

        # Default deny
        return PolicyDecision(
            allowed=False,
            reason="Path not in any allow list",
            tool_name=tool_name,
            params_summary=summary,
        )

    def _check_shell(self, params: dict, rule_set: dict) -> PolicyDecision:
        """Check shell command execution."""
        command = params.get("command", "")
        if not command:
            return PolicyDecision(
                allowed=False, reason="No command provided",
                tool_name="shell_exec", params_summary="",
            )

        summary = command[:200]

        # Deny-first: check deny_commands
        for deny_cmd in rule_set.get("deny_commands", []):
            if deny_cmd in command:
                return PolicyDecision(
                    allowed=False,
                    reason=f"Command matches deny rule: {deny_cmd}",
                    matched_rule=deny_cmd,
                    tool_name="shell_exec",
                    params_summary=summary,
                )

        # Check deny_patterns (shell injection patterns)
        for pattern in rule_set.get("deny_patterns", []):
            if pattern in command:
                return PolicyDecision(
                    allowed=False,
                    reason=f"Command contains denied pattern: {pattern}",
                    matched_rule=pattern,
                    tool_name="shell_exec",
                    params_summary=summary,
                )

        # Check allow_commands (must start with allowed command)
        for allow_cmd in rule_set.get("allow_commands", []):
            if command.strip().startswith(allow_cmd):
                return PolicyDecision(
                    allowed=True,
                    reason=f"Command matches allow rule: {allow_cmd}",
                    matched_rule=allow_cmd,
                    tool_name="shell_exec",
                    params_summary=summary,
                )

        # Default deny
        return PolicyDecision(
            allowed=False,
            reason="Command not in any allow list",
            tool_name="shell_exec",
            params_summary=summary,
        )

    def _check_network(self, params: dict, rule_set: dict) -> PolicyDecision:
        """Check network access."""
        host = params.get("host", "")
        port = params.get("port", 0)
        summary = f"{host}:{port}"

        # Check if external access is denied
        if rule_set.get("deny_all_external", False):
            allowed_hosts = rule_set.get("allow_hosts", [])
            if host not in allowed_hosts:
                return PolicyDecision(
                    allowed=False,
                    reason=f"External host denied: {host}",
                    matched_rule="deny_all_external",
                    tool_name="network",
                    params_summary=summary,
                )

        # Check allowed ports
        allowed_ports = rule_set.get("allow_ports", [])
        if allowed_ports and port not in allowed_ports:
            return PolicyDecision(
                allowed=False,
                reason=f"Port {port} not in allowed ports: {allowed_ports}",
                matched_rule="allow_ports",
                tool_name="network",
                params_summary=summary,
            )

        return PolicyDecision(
            allowed=True,
            reason="Network access permitted",
            tool_name="network",
            params_summary=summary,
        )
