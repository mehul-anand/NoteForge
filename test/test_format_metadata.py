from src.nodes.react_node import Nodes


def test_format_metadata_accepts_dict():
    meta = {
        "title": "A Study",
        "year": 2023,
        "authors": ["Alice", "Bob"],
        "affiliations": [],
        "venue": "",
        "methods": [],
        "keywords": [],
        "contributions": [],
        "key_results": [],
    }
    out = Nodes._format_paper_metadata("study.pdf", meta)
    assert "study.pdf — A Study (2023)" in out
    assert "authors: Alice, Bob" in out


def test_format_metadata_truncates_long_fields():
    long_aff = ["X" * 500]
    meta = {
        "title": "T",
        "authors": [],
        "affiliations": long_aff,
        "venue": "",
        "methods": [],
        "keywords": [],
        "contributions": [],
        "key_results": [],
    }
    out = Nodes._format_paper_metadata("f.pdf", meta)
    assert "f.pdf — T;" in out
    assert len(out) < 2000