from src.nodes.react_node import _clean_history, _history_context

NOISE = (
    "You've hit the demo's temporary usage limit — please wait a minute "
    "and try again."
)
ERROR = (
    "Sorry, something went wrong while generating that answer. Please try again."
)


def test_clean_history_drops_system_noise():
    history = [
        {"role": "user", "content": "who wrote paper A?"},
        {"role": "assistant", "content": "The author is Alice."},
        {"role": "user", "content": "another question"},
        {"role": "assistant", "content": NOISE},
        {"role": "assistant", "content": ERROR},
        {"role": "assistant", "content": "I'm sorry, I can't help with that request."},
        {"role": "assistant", "content": "No documents loaded. Upload PDFs or add a URL using the sidebar."},
        {"role": "system", "content": "ignored"},
    ]
    cleaned = _clean_history(history)
    assert cleaned == [
        {"role": "user", "content": "who wrote paper A?"},
        {"role": "assistant", "content": "The author is Alice."},
        {"role": "user", "content": "another question"},
    ]


def test_history_context_only_last_turns():
    turns = [{"role": "user", "content": f"q{i}"} for i in range(6)]
    ctx = _history_context(turns, turns=4)
    assert "q2" in ctx and "q5" in ctx
    assert "q0" not in ctx and "q1" not in ctx


def test_history_context_empty():
    assert _history_context([]) == ""
    assert _history_context([{"role": "assistant", "content": NOISE}]) == ""