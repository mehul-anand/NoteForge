from langchain_core.documents import Document

from src.document_ingestion.document_processor import relabel_sources


def test_relabel_sources_overwrites_temp_path():
    docs = [
        Document(page_content="x", metadata={"source": "/tmp/tmpj_fmf38p.pdf"}),
        Document(page_content="y", metadata={"source": "/tmp/tmp24avjxp8.pdf"}),
    ]
    relabel_sources(docs, "HyDE_2025.pdf")
    assert [d.metadata["source"] for d in docs] == ["HyDE_2025.pdf", "HyDE_2025.pdf"]


def test_relabel_sources_handles_empty_list():
    relabel_sources([], "anything.pdf")  # must not raise