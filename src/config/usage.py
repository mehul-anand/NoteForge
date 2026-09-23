"""UsageGate — in-process rate limiting for the deployed demo.

Counters live in a module-level singleton (see get_usage_gate) so they
persist across Streamlit session reruns within one running process — the
global per-minute/hour/day caps are the actual anti-abuse ceiling on a
single hosted instance (Streamlit Community Cloud). Per-session budgets are
keyed by a caller-supplied session id.

All budgets are granted at the SUBMISSION level (one graph.run == one unit),
not per internal LLM call. Each graph.run consumes ~5-7 LLM calls, so the
caps here are scaled-down ceilings derived from the AGENTS.md budgets, not
exact LLM-call counts.

Values are overridable via constructor kwargs (used by tests and for tuning
a demo budget without touching constants).
"""

import time
from collections import OrderedDict, deque
from functools import lru_cache


class _Limit:
    """Rolling counter: fixed window in seconds, or lifetime (never evicts)."""

    __slots__ = ("max_calls", "window", "events")

    def __init__(self, max_calls: int, window: float | None):
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        self.max_calls = max_calls
        self.window = window
        self.events: deque[float] = deque()

    def _prune(self, now: float) -> None:
        if self.window is not None:
            cutoff = now - self.window
            while self.events and self.events[0] <= cutoff:
                self.events.popleft()

    def can_grant(self, now: float) -> bool:
        self._prune(now)
        return len(self.events) < self.max_calls

    def record(self, now: float) -> None:
        self.events.append(now)


class UsageGate:
    """Submission-level budgets. Designed for single-threaded apps."""

    def __init__(
        self,
        *,
        clock=time.monotonic,
        llm_per_min: int = 10,
        llm_per_session: int = 50,
        llm_per_hour: int = 200,
        llm_per_day: int = 600,
        moderation_per_min: int = 30,
        ingest_per_hour: int = 60,
        ingest_per_day: int = 300,
        max_sessions: int = 1000,
    ):
        self._clock = clock
        self._llm_per_min = _Limit(llm_per_min, 60)
        self._llm_per_hour = _Limit(llm_per_hour, 3600)
        self._llm_per_day = _Limit(llm_per_day, 86400)
        self._moderation_per_min = _Limit(moderation_per_min, 60)
        self._ingest_per_hour = _Limit(ingest_per_hour, 3600)
        self._ingest_per_day = _Limit(ingest_per_day, 86400)
        self._session_budget = llm_per_session
        self._sessions: OrderedDict[str, _Limit] = OrderedDict()
        self._max_sessions = max_sessions

    def _session_limit(self, session_id: str | None) -> _Limit | None:
        if session_id is None:
            return None
        limit = self._sessions.get(session_id)
        if limit is None:
            if len(self._sessions) >= self._max_sessions:
                self._sessions.popitem(last=False)
            limit = _Limit(self._session_budget, None)
            self._sessions[session_id] = limit
        else:
            self._sessions.move_to_end(session_id)
        return limit

    def _grant(self, *limits: _Limit) -> bool:
        """Check all limits, then record on success (single-threaded)."""
        now = self._clock()
        if not all(limit.can_grant(now) for limit in limits):
            return False
        for limit in limits:
            limit.record(now)
        return True

    def try_llm(self, session_id: str | None = None) -> bool:
        """Grant one graph.run submission (global + per-session budgets)."""
        limits = [self._llm_per_min, self._llm_per_hour, self._llm_per_day]
        session_limit = self._session_limit(session_id)
        if session_limit is not None:
            limits.append(session_limit)
        return self._grant(*limits)

    def try_moderation(self) -> bool:
        return self._grant(self._moderation_per_min)

    def try_ingest(self) -> bool:
        """Grant one document-process/embedding run (per-hour/day global)."""
        return self._grant(self._ingest_per_hour, self._ingest_per_day)


@lru_cache(maxsize=1)
def get_usage_gate() -> UsageGate:
    """Process-wide singleton so budgets survive Streamlit reruns."""
    return UsageGate()