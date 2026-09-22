from types import SimpleNamespace

from src.document_ingestion.document_processor import (
    DocumentHandler,
    PaperMetadata,
)

VALID_FIELDS = {
    "source": "",
    "title": "Test Paper",
    "authors": [],
    "affiliations": [],
    "year": None,
    "venue": "",
    "methods": [],
    "contributions": [],
    "key_results": [],
    "keywords": [],
}


class RaisingStructured:
    def invoke(self, prompt):
        raise RuntimeError("structured parse failure")


class RetryStructured:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("first attempt fails")
        data = dict(VALID_FIELDS, title="Retried Paper")
        return SimpleNamespace(model_dump=lambda: data)


class FakeLLM:
    def __init__(self, text):
        self.text = text

    def invoke(self, prompt):
        return SimpleNamespace(content=self.text)


def test_structured_failure_degrades_to_summary():
    handler = DocumentHandler()
    llm = FakeLLM("one line summary")
    paper = handler._extract_metadata_for_file(
        "x.pdf", "content", llm, RaisingStructured()
    )
    assert paper.source == "x.pdf"
    assert "one line summary" in (paper.contributions or [])


def test_retry_succeeds_after_first_failure():
    handler = DocumentHandler()
    paper = handler._extract_metadata_for_file(
        "x.pdf", "content", None, RetryStructured()
    )
    assert paper.title == "Retried Paper"
    assert paper.source == "x.pdf"


def test_fill_year_from_filename():
    paper = PaperMetadata(source="CBF_UAV_2024.pdf")
    DocumentHandler._fill_year(paper, "CBF_UAV_2024.pdf")
    assert paper.year == 2024


def test_fill_year_respects_existing_year():
    paper = PaperMetadata(source="old.pdf", year=2019)
    DocumentHandler._fill_year(paper, "renamed_2024.pdf")
    assert paper.year == 2019