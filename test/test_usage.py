"""Tests for the UsageGate rate limiter (faked clock, no network)."""

import pytest

from src.config.usage import UsageGate, get_usage_gate


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def test_llm_session_budget_is_per_session():
    clock = FakeClock()
    gate = UsageGate(
        clock=clock,
        llm_per_min=10_000,
        llm_per_hour=10_000,
        llm_per_day=10_000,
        llm_per_session=3,
    )
    assert gate.try_llm("s1") is True
    assert gate.try_llm("s1") is True
    assert gate.try_llm("s1") is True
    assert gate.try_llm("s1") is False
    assert gate.try_llm("s2") is True


def test_per_minute_window_rolls_over():
    clock = FakeClock()
    gate = UsageGate(
        clock=clock,
        llm_per_min=2,
        llm_per_hour=10_000,
        llm_per_day=10_000,
        llm_per_session=10_000,
    )
    assert gate.try_llm("s") is True
    assert gate.try_llm("s") is True
    assert gate.try_llm("s") is False
    clock.advance(61)
    assert gate.try_llm("s") is True


def test_global_daily_cap_binds_across_sessions():
    clock = FakeClock()
    gate = UsageGate(
        clock=clock,
        llm_per_min=10_000,
        llm_per_hour=10_000,
        llm_per_day=2,
        llm_per_session=10_000,
    )
    assert gate.try_llm("a") is True
    assert gate.try_llm("b") is True
    assert gate.try_llm("c") is False


def test_moderation_cap_rolls_over():
    clock = FakeClock()
    gate = UsageGate(clock=clock, moderation_per_min=1)
    assert gate.try_moderation() is True
    assert gate.try_moderation() is False
    clock.advance(61)
    assert gate.try_moderation() is True


def test_ingest_global_cap():
    clock = FakeClock()
    gate = UsageGate(clock=clock, ingest_per_day=1)
    assert gate.try_ingest() is True
    assert gate.try_ingest() is False


def test_singleton_is_cached():
    assert get_usage_gate() is get_usage_gate()


@pytest.mark.parametrize("max_calls", [0, -1])
def test_invalid_limit_rejected(max_calls):
    with pytest.raises(ValueError):
        UsageGate(clock=FakeClock(), llm_per_min=max_calls)