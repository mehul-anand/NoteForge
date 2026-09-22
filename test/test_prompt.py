from pathlib import Path

from src.nodes.react_node import (
    _BUNDLED_SYSTEM_PROMPT,
    _PROMPT_PATH,
    load_system_prompt,
)


def test_versioned_prompt_file_exists():
    assert _PROMPT_PATH.exists()
    assert "SECURITY" in load_system_prompt()


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("AGENT_SYSTEM_PROMPT", "custom prompt")
    assert load_system_prompt() == "custom prompt"


def test_bundled_fallback_when_file_missing(monkeypatch):
    monkeypatch.delenv("AGENT_SYSTEM_PROMPT", raising=False)
    monkeypatch.setattr(
        "src.nodes.react_node._PROMPT_PATH", Path("/nonexistent/agent_v1.md")
    )
    assert load_system_prompt() == _BUNDLED_SYSTEM_PROMPT