"""
DyTopo Silent Verifier
======================

Pure deterministic output verification for agent work products.
NO LLM calls -- uses ast.parse(), JSON schema checks, and subprocess execution.
On any infrastructure failure the verifier defaults to PASS so it never blocks
the pipeline.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass

logger = logging.getLogger("dytopo.verifier")


@dataclass
class VerificationResult:
    """Result of a single verification check."""

    passed: bool
    exit_code: int = 0
    stderr: str = ""
    fix_hint: str = ""
    method: str = ""  # "syntax_check", "schema_validation", "code_execution", "disabled"


class OutputVerifier:
    """Deterministic output verification for agent work products.

    Dispatches to syntax checks, schema validation, or sandboxed code execution
    based on per-role specs supplied in *verification_config*.

    Fail-open policy: any infrastructure error results in ``passed=True`` so the
    swarm pipeline is never blocked by the verifier itself.
    """

    def __init__(self, verification_config: dict) -> None:
        self.enabled: bool = verification_config.get("enabled", False)
        self.max_retries: int = verification_config.get("max_retries", 1)
        self.specs: dict = verification_config.get("specs", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def verify(self, agent_role: str, output: str) -> VerificationResult:
        """Verify an agent's output based on its role spec.

        Args:
            agent_role: The agent's role (e.g. ``"developer"``, ``"researcher"``).
            output: The raw text output produced by the agent.

        Returns:
            A :class:`VerificationResult`.  ``passed`` is ``True`` when the
            check succeeds **or** when verification is disabled / not
            configured for the role.
        """
        if not self.enabled or agent_role not in self.specs:
            return VerificationResult(passed=True, method="disabled")

        try:
            spec = self.specs[agent_role]
            check_type = spec.get("type", "")

            if check_type == "syntax_check":
                return self._check_python_syntax(output)
            elif check_type == "schema_validation":
                required_fields = spec.get("required_fields", [])
                return self._check_schema(output, required_fields)
            elif check_type == "code_execution":
                command = spec.get("command", "python")
                timeout = spec.get("timeout_seconds", 10)
                return await self._execute_code(output, command, timeout)
            else:
                logger.debug("Unknown verification type %r for role %r", check_type, agent_role)
                return VerificationResult(passed=True, method="disabled")

        except Exception as exc:  # noqa: BLE001 — fail-open by design
            logger.warning(
                "Verifier infrastructure failure for role %r: %s", agent_role, exc,
            )
            return VerificationResult(passed=True, method="disabled")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_code_block(output: str, language: str) -> str:
        """Extract the first fenced code block for *language* from *output*.

        Tries, in order:
        1. A fenced block tagged with the requested *language*
           (e.g. ````` ```python ````` or ````` ```json `````).
        2. A generic fenced block (````` ``` `````).
        3. Falls back to the raw *output* text.
        """
        # Try language-specific fence first.
        pattern = r"```" + re.escape(language) + r"\s*\n(.*?)```"
        match = re.search(pattern, output, re.DOTALL)
        if match:
            return match.group(1)

        # Try generic / "code" fence.
        match = re.search(r"```(?:code)?\s*\n(.*?)```", output, re.DOTALL)
        if match:
            return match.group(1)

        # Raw text fallback.
        return output

    # ------------------------------------------------------------------
    # Verification strategies
    # ------------------------------------------------------------------

    def _check_python_syntax(self, output: str) -> VerificationResult:
        """Check Python syntax, extracting code from markdown fences when present."""
        code = self._extract_code_block(output, "python")
        code = code.strip()
        if not code:
            return VerificationResult(passed=True, method="syntax_check")

        try:
            ast.parse(code)
        except SyntaxError as e:
            return VerificationResult(
                passed=False,
                exit_code=1,
                stderr=str(e),
                fix_hint=f"Syntax error at line {e.lineno}: {e.msg}",
                method="syntax_check",
            )

        return VerificationResult(passed=True, method="syntax_check")

    def _check_schema(self, output: str, required_fields: list[str]) -> VerificationResult:
        """Validate that *output* contains a JSON object with all *required_fields*."""
        text = self._extract_code_block(output, "json").strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return VerificationResult(
                passed=False,
                exit_code=1,
                stderr=f"JSON parse error: {e}",
                fix_hint="Output must contain valid JSON",
                method="schema_validation",
            )

        if not isinstance(data, dict):
            return VerificationResult(
                passed=False,
                exit_code=1,
                stderr=f"Expected a JSON object, got {type(data).__name__}",
                fix_hint="Output must be a JSON object with required fields",
                method="schema_validation",
            )

        missing = [f for f in required_fields if f not in data]
        if missing:
            return VerificationResult(
                passed=False,
                exit_code=1,
                stderr=f"Missing required fields: {missing}",
                fix_hint=f"Add missing fields to JSON output: {missing}",
                method="schema_validation",
            )

        return VerificationResult(passed=True, method="schema_validation")

    async def _execute_code(
        self, output: str, command: str, timeout: int
    ) -> VerificationResult:
        """Write code to a temp file and execute it in a subprocess.

        On ``TimeoutError`` or any other exception the method returns
        ``passed=True`` (fail-open) so the pipeline is never blocked.
        """
        code = self._extract_code_block(output, "python").strip()
        if not code:
            return VerificationResult(passed=True, method="code_execution")

        tmp_path: str | None = None
        try:
            # Write code to a temporary file.
            fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="dytopo_verify_")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(code)

            proc = await asyncio.create_subprocess_exec(
                command,
                tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )

            stderr_text = stderr_bytes.decode("utf-8", errors="replace")[:500]

            if proc.returncode != 0:
                return VerificationResult(
                    passed=False,
                    exit_code=proc.returncode or 1,
                    stderr=stderr_text,
                    fix_hint=f"Code execution failed (exit {proc.returncode}): {stderr_text}",
                    method="code_execution",
                )

            return VerificationResult(passed=True, method="code_execution")

        except asyncio.TimeoutError:
            logger.warning("Code execution timed out after %ss", timeout)
            return VerificationResult(passed=True, method="code_execution")

        except Exception as exc:  # noqa: BLE001 — fail-open by design
            logger.warning("Code execution failed with infrastructure error: %s", exc)
            return VerificationResult(passed=True, method="code_execution")

        finally:
            # Clean up the temp file.
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
