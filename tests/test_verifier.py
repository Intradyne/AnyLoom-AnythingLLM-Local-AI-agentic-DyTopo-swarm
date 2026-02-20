"""Tests for dytopo.verifier module."""

import pytest

from dytopo.verifier import OutputVerifier, VerificationResult


@pytest.fixture
def enabled_verifier():
    return OutputVerifier({
        "enabled": True,
        "max_retries": 1,
        "specs": {
            "developer": {"type": "syntax_check", "timeout_seconds": 10},
            "researcher": {"type": "schema_validation", "required_fields": ["sources", "summary"]},
            "solver": {"type": "syntax_check", "timeout_seconds": 30},
        },
    })


@pytest.fixture
def disabled_verifier():
    return OutputVerifier({"enabled": False})


class TestSyntaxCheck:
    @pytest.mark.asyncio
    async def test_valid_python(self, enabled_verifier):
        output = '```python\ndef hello():\n    return "world"\n```'
        result = await enabled_verifier.verify("developer", output)
        assert result.passed
        assert result.method == "syntax_check"

    @pytest.mark.asyncio
    async def test_invalid_python(self, enabled_verifier):
        output = '```python\ndef hello(\n    return\n```'
        result = await enabled_verifier.verify("developer", output)
        assert not result.passed
        assert result.method == "syntax_check"
        assert result.fix_hint  # Should contain helpful message

    @pytest.mark.asyncio
    async def test_raw_valid_python(self, enabled_verifier):
        output = 'x = 1 + 2\nprint(x)'
        result = await enabled_verifier.verify("developer", output)
        assert result.passed

    @pytest.mark.asyncio
    async def test_empty_code_passes(self, enabled_verifier):
        output = ''
        result = await enabled_verifier.verify("developer", output)
        assert result.passed


class TestSchemaValidation:
    @pytest.mark.asyncio
    async def test_valid_schema(self, enabled_verifier):
        output = '```json\n{"sources": ["a", "b"], "summary": "test"}\n```'
        result = await enabled_verifier.verify("researcher", output)
        assert result.passed
        assert result.method == "schema_validation"

    @pytest.mark.asyncio
    async def test_missing_fields(self, enabled_verifier):
        output = '{"sources": ["a"]}'
        result = await enabled_verifier.verify("researcher", output)
        assert not result.passed
        assert "summary" in result.fix_hint

    @pytest.mark.asyncio
    async def test_invalid_json(self, enabled_verifier):
        output = 'not valid json at all'
        result = await enabled_verifier.verify("researcher", output)
        assert not result.passed
        assert result.method == "schema_validation"

    @pytest.mark.asyncio
    async def test_non_object_json(self, enabled_verifier):
        output = '["just", "an", "array"]'
        result = await enabled_verifier.verify("researcher", output)
        assert not result.passed


class TestDisabled:
    @pytest.mark.asyncio
    async def test_disabled_verifier_passes(self, disabled_verifier):
        result = await disabled_verifier.verify("developer", "invalid python {{{{")
        assert result.passed
        assert result.method == "disabled"

    @pytest.mark.asyncio
    async def test_unknown_role_passes(self, enabled_verifier):
        result = await enabled_verifier.verify("unknown_role", "anything")
        assert result.passed
        assert result.method == "disabled"


class TestFailOpen:
    @pytest.mark.asyncio
    async def test_infra_failure_defaults_pass(self):
        """Verify that infrastructure failures default to pass."""
        # Create verifier with a spec that has an unknown type
        verifier = OutputVerifier({
            "enabled": True,
            "specs": {"tester": {"type": "nonexistent_check"}},
        })
        result = await verifier.verify("tester", "test output")
        assert result.passed
