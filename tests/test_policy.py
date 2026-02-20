"""Tests for dytopo.policy module."""

import json
import pytest

from dytopo.policy import PolicyEnforcer, PolicyDecision


@pytest.fixture
def policy_file(tmp_path):
    """Create a test policy.json file."""
    policy = {
        "rules": {
            "file_write": {
                "allow_paths": ["./workspace/*", "./output/*", "~/dytopo-logs/*", "~/dytopo-checkpoints/*"],
                "deny_paths": ["./src/*", "./.git/*", "./config/*", "/etc/*"],
            },
            "file_read": {
                "allow_paths": ["./workspace/*", "./output/*", "./docs/*", "./rag-docs/*"],
                "deny_paths": ["./.env", "./.git/*", "~/.ssh/*"],
            },
            "shell_exec": {
                "allow_commands": ["python", "pip", "node", "npm", "git status", "git log", "git diff"],
                "deny_commands": ["rm -rf", "sudo", "chmod", "chown", "curl", "wget"],
                "deny_patterns": ["|", "&&", ">>", "$("],
            },
            "network": {
                "allow_hosts": ["localhost", "127.0.0.1"],
                "allow_ports": [8008, 8009, 6333],
                "deny_all_external": True,
            },
        },
        "enforcement": "strict",
        "log_denials": True,
        "on_denial": "return_error",
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    return path


@pytest.fixture
def enforcer(policy_file):
    return PolicyEnforcer(policy_path=policy_file)


class TestFileWrite:
    def test_allow_workspace_write(self, enforcer):
        d = enforcer.check_tool_request("file_write", {"path": "./workspace/output.txt"})
        assert d.allowed

    def test_deny_src_write(self, enforcer):
        d = enforcer.check_tool_request("file_write", {"path": "./src/hack.py"})
        assert not d.allowed
        assert "deny" in d.reason.lower()

    def test_deny_git_write(self, enforcer):
        d = enforcer.check_tool_request("file_write", {"path": "./.git/config"})
        assert not d.allowed

    def test_deny_etc_write(self, enforcer):
        d = enforcer.check_tool_request("file_write", {"path": "/etc/passwd"})
        assert not d.allowed


class TestFileRead:
    def test_allow_docs_read(self, enforcer):
        d = enforcer.check_tool_request("file_read", {"path": "./docs/readme.md"})
        assert d.allowed

    def test_deny_env_read(self, enforcer):
        d = enforcer.check_tool_request("file_read", {"path": "./.env"})
        assert not d.allowed

    def test_deny_ssh_read(self, enforcer):
        d = enforcer.check_tool_request("file_read", {"path": "~/.ssh/id_rsa"})
        assert not d.allowed


class TestShellExec:
    def test_allow_python(self, enforcer):
        d = enforcer.check_tool_request("shell_exec", {"command": "python script.py"})
        assert d.allowed

    def test_deny_rm_rf(self, enforcer):
        d = enforcer.check_tool_request("shell_exec", {"command": "rm -rf /"})
        assert not d.allowed

    def test_deny_sudo(self, enforcer):
        d = enforcer.check_tool_request("shell_exec", {"command": "sudo apt install foo"})
        assert not d.allowed

    def test_deny_pipe_injection(self, enforcer):
        d = enforcer.check_tool_request("shell_exec", {"command": "python script.py | cat /etc/passwd"})
        assert not d.allowed

    def test_deny_command_chaining(self, enforcer):
        d = enforcer.check_tool_request("shell_exec", {"command": "echo hello && rm -rf /"})
        assert not d.allowed

    def test_deny_subshell(self, enforcer):
        d = enforcer.check_tool_request("shell_exec", {"command": "python $(malicious)"})
        assert not d.allowed

    def test_allow_git_status(self, enforcer):
        d = enforcer.check_tool_request("shell_exec", {"command": "git status"})
        assert d.allowed


class TestNetwork:
    def test_allow_localhost(self, enforcer):
        d = enforcer.check_tool_request("network", {"host": "localhost", "port": 8008})
        assert d.allowed

    def test_deny_external_host(self, enforcer):
        d = enforcer.check_tool_request("network", {"host": "evil.com", "port": 80})
        assert not d.allowed

    def test_deny_wrong_port(self, enforcer):
        d = enforcer.check_tool_request("network", {"host": "localhost", "port": 9999})
        assert not d.allowed

    def test_allow_qdrant_port(self, enforcer):
        d = enforcer.check_tool_request("network", {"host": "127.0.0.1", "port": 6333})
        assert d.allowed


class TestEnforce:
    def test_enforce_returns_none_on_allow(self, enforcer):
        result = enforcer.enforce("shell_exec", {"command": "python test.py"})
        assert result is None

    def test_enforce_returns_error_on_deny(self, enforcer):
        result = enforcer.enforce("shell_exec", {"command": "sudo rm -rf /"})
        assert result is not None
        assert result["error"] == "policy_denied"


class TestNoPolicy:
    def test_no_policy_allows_all(self, tmp_path):
        enforcer = PolicyEnforcer(policy_path=tmp_path / "nonexistent.json")
        d = enforcer.check_tool_request("shell_exec", {"command": "sudo rm -rf /"})
        assert d.allowed
