from types import SimpleNamespace

from src.document_ingestion.document_processor import PaperMetadata
from src.suggestions import Followups, deterministic_followups, generate_followups

META = {
    "A.pdf": PaperMetadata(
        title="Retrieval Paper A",
        authors=["Alice", "Bob"],
        methods=["MMR"],
    ),
    "B.pdf": PaperMetadata(
        title="Retrieval Paper B",
        authors=["Carol"],
        methods=["HyDE"],
    ),
}


class FakeLLM:
    def __init__(self, questions=None, raise_on_invoke=False):
        self._questions = questions or []
        self._raise = raise_on_invoke

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, prompt):
        if self._raise:
            raise RuntimeError("llm failure")
        return Followups(questions=self._questions)


def _result(**overrides):
    base = {
        "source_files": ["A.pdf", "B.pdf"],
        "paper_metadata": META,
        "retrieved_docs": [],
        "answer": "some answer",
        "task_type": "qa",
    }
    base.update(overrides)
    return base


def test_deterministic_qa_up_to_three():
    out = deterministic_followups(["A.pdf", "B.pdf"], META, "qa")
    assert len(out) <= 3
    assert any("key results" in q for q in out)
    assert any("authors" in q for q in out)
    assert all(q.endswith("?") for q in out)


def test_deterministic_compare_names_both_papers():
    out = deterministic_followups(["A.pdf", "B.pdf"], META, "compare")
    assert any(
        "Retrieval Paper A" in q and "Retrieval Paper B" in q for q in out
    )


def test_deterministic_gaps():
    out = deterministic_followups(["A.pdf", "B.pdf"], META, "gaps")
    assert any("research gap" in q for q in out)


def test_llm_path_returns_model_questions():
    llm = FakeLLM(questions=["Tell me more about MMR?", "How does HyDE work?"])
    out = generate_followups(_result(), llm)
    assert out == ["Tell me more about MMR?", "How does HyDE work?"]


def test_llm_caps_at_three():
    llm = FakeLLM(questions=["q1", "q2", "q3", "q4"])
    out = generate_followups(_result(), llm)
    assert len(out) == 3


def test_llm_failure_falls_back_to_deterministic():
    llm = FakeLLM(raise_on_invoke=True)
    out = generate_followups(_result(), llm)
    assert out  # non-empty fallback


def test_empty_metadata_yields_empty_deterministic():
    assert deterministic_followups([], {}, "qa") == []


def test_retrieved_excerpt_honors_source_names(tmp_path):
    from langchain_core.documents import Document

    doc = Document(
        page_content="x" * 2000, metadata={"source": tmp_path / "A.pdf"}
    )
    result = _result(retrieved_docs=[doc])
    out = generate_followups(result, FakeLLM(questions=["q1"]))
    assert out == ["q1"]


def test_answer_not_required_for_deterministic():
    result = _result(answer="")
    assert generate_followups(result, FakeLLM(raise_on_invoke=True))