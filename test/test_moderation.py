import src.config.config as config


def test_moderation_fails_open_without_api_key(monkeypatch):
    """No key/broken API must not crash the gate — allow the request through."""
    monkeypatch.setattr(config.Config, "OPENAI_API_KEY", "")
    config.moderation.cache_clear()
    try:
        assert config.moderation("innocent text") is True
    finally:
        config.moderation.cache_clear()