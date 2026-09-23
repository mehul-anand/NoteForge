from streamlit_app import _stream_chunks


def test_empty_text_yields_single_chunk():
    assert _stream_chunks("") == [""]
    assert _stream_chunks(None) == [""]


def test_short_text_collapses_to_single_full_chunk():
    assert _stream_chunks("hi") == ["hi"]


def test_chunks_grow_monotonically_and_cover_text():
    text = "x" * 500
    prefixes = _stream_chunks(text)
    assert len(prefixes) <= 60
    assert len(prefixes) >= 2
    assert prefixes[-1] == text
    lengths = [len(p) for p in prefixes]
    assert lengths == sorted(lengths)
    assert lengths[0] >= 1


def test_max_updates_is_respected():
    text = "y" * 1000
    prefixes = _stream_chunks(text, max_updates=7)
    assert len(prefixes) <= 7
    assert prefixes[-1] == text
    assert all(isinstance(p, str) and text.startswith(p) for p in prefixes)