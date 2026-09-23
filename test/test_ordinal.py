from src.document_ingestion.document_processor import PaperMetadata, normalize_source_key
from src.nodes.react_node import Nodes

LITM_URL = "https://arxiv.org/html/2307.03172v3"
LITM_META = PaperMetadata(
    title="Lost in the Middle: How Language Models Use Long Contexts",
    authors=[
        "Nelson F. Liu", "Kevin Lin", "John Hewitt", "Ashwin Paranjape",
        "Michele Bevilacqua", "Fabio Petroni", "Percy Liang",
    ],
    year=2023,
    venue="arXiv",
    source=LITM_URL,
)


def _metadata_map():
    return {
        normalize_source_key(LITM_URL): LITM_META,
        normalize_source_key("CBF_UAV_2024.pdf"): PaperMetadata(
            title="CBF UAV Guidance",
            authors=["a", "b", "c", "d"],
            source="CBF_UAV_2024.pdf",
        ),
    }


def _sources():
    return sorted(_metadata_map())


def test_matches_url_title_acronym_and_plain_stem():
    """'LITM' (acronym of Lost in the Middle) binds to the uploaded arXiv URL,
    and a plain stem reference picks the right PDF. Spaced shorthand that is
    ambiguous ('CBF UAV' with both CBF_UAV and CBF_UGV present) must NOT be
    guessed — it returns None so the agent can't mis-attribute ordinals."""
    meta = _metadata_map()
    sources = _sources()
    assert Nodes._match_file("who is the 3rd author of LITM?", sources, meta) == normalize_source_key(LITM_URL)
    assert Nodes._match_file("4th author of CBF_UAV_2024.pdf", sources, meta) == normalize_source_key("CBF_UAV_2024.pdf")
    # Ambiguous: both UAV and UGV files contain 'CBF' — must not guess.
    assert Nodes._match_file("4th author of the CBF UAV paper", sources, meta) is None


def test_resolve_ordinal_facts_via_url_acronym():
    """No LLM: the 3rd/4th-ordinal clauses about LITM resolve to the exact
    authors from PAPER METADATA instead of being guessed by the agent."""
    meta = _metadata_map()
    sources = _sources()
    block = "\n".join(
        f"- {name}: {m.title}; authors: {', '.join(m.authors)}"
        for name, m in meta.items()
    )
    facts = Nodes._resolve_ordinal_facts(
        ["who is the 3rd author of LITM?", "who is the 4th author of LITM?"],
        sources,
        block,
        meta,
    )
    assert "John Hewitt" in facts
    assert "Ashwin Paranjape" in facts
