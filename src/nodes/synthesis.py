"""Intent routing + review synthesis.

Two nodes plugged into the graph:

- `route_intent`: one cheap structured-output call classifies the user question
  into `qa | compare | review | gaps` (default `qa` on any failure).
- `synthesize`: for non-qa intents, one structured-output call per task type
  produces a deterministic artifact (ComparisonMatrix / RelatedWork / ResearchGaps),
  which is then rendered to markdown for the chat. The raw artifact is kept on
  `State.artifact` for eval scoring and later UI.

The Q&A path never touches this module — `qa` routes to the existing agent.
"""

from enum import StrEnum
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from src.nodes.react_node import Nodes
from src.state.state import State


class Intent(StrEnum):
    QA = "qa"
    COMPARE = "compare"
    REVIEW = "review"
    GAPS = "gaps"


class IntentOut(BaseModel):
    intent: Intent
    mixed: bool = False


class ComparisonRow(BaseModel):
    paper: str = ""
    task: str = ""
    method: str = ""
    key_results: str = ""
    limitations: str = ""


class ComparisonMatrix(BaseModel):
    overview: str = ""
    rows: List[ComparisonRow] = Field(default_factory=list)


class ThemeGroup(BaseModel):
    theme: str = ""
    text: str = ""


class RelatedWork(BaseModel):
    overview: str = ""
    theme_groups: List[ThemeGroup] = Field(default_factory=list)


class Gap(BaseModel):
    gap: str = ""
    evidence: str = ""


class ResearchGaps(BaseModel):
    gaps: List[Gap] = Field(default_factory=list)


_CLASSIFY_PROMPT = (
    "Classify the user's question into exactly one category:\n"
    "- qa: a direct factual lookup or question about the documents\n"
    "- compare: compares or contrasts two or more papers/methods/approaches, "
    "or asks what papers have in common / how they differ / what they share\n"
    "- review: writing a related-work or literature review narrative/draft\n"
    "- gaps: identifying research gaps, missing topics, or open problems across the papers\n"
    "Choose qa when uncertain or when the intent is a plain question.\n"
    "If the question combines parts with DIFFERENT intents (e.g. a comparison "
    "PLUS a factual lookup, or a general question joined to specific 'who is / "
    "what is / nth author' asks with 'also' or 'and'), set `mixed` to true AND "
    "set `intent` to the most complex part.\n"
    "Note: asking what papers have in common or share is compare, not gaps.\n"
    "User question: {question}"
)


class ReviewSynthesizer:
    def __init__(self, llm):
        self.llm = llm
        self._classifier = None
        self._structured = {}

    # ----- routing -----

    def route_intent(self, state: State) -> State:
        intent, mixed = self._classify(state.question)
        task_type = "qa" if mixed else intent.value
        return state.model_copy(update={"task_type": task_type})

    def _classify(self, question: str) -> tuple[Intent, bool]:
        try:
            if self._classifier is None:
                self._classifier = self.llm.with_structured_output(IntentOut)
            result = self._classifier.invoke(
                _CLASSIFY_PROMPT.format(question=question)
            )
            if isinstance(result, IntentOut) and result.intent is not None:
                return result.intent, bool(result.mixed)
            text = str(getattr(result, "intent", result)).strip().lower()
            for candidate in Intent:
                if candidate.value in text or candidate.name.lower() in text:
                    return candidate, False
            return Intent.QA, False
        except Exception:
            return Intent.QA, False

    # ----- synthesis -----

    def synthesize(self, state: State) -> State:
        task = state.task_type
        context = self._build_context(state)

        schema = {
            Intent.COMPARE.value: ComparisonMatrix,
            Intent.REVIEW.value: RelatedWork,
            Intent.GAPS.value: ResearchGaps,
        }.get(task)

        if schema is None:
            return state.model_copy(update={"answer": state.question, "artifact": None})

        prompt = _SYNTHESIS_PROMPTS.get(task, "")
        try:
            chain = self._structured_for(task, schema)
            for _ in range(2):  # one retry on failure
                try:
                    resp = chain.invoke(prompt.format(context=context, question=state.question))
                    artifact = resp.model_dump()
                    answer = self._render(task, resp)
                    return state.model_copy(update={"answer": answer, "artifact": artifact})
                except Exception:
                    continue
        except Exception:
            pass

        # Degraded fallback: ordinary prose answer, no artifact.
        fallback = self.llm.invoke(
            "Answer the user's question using the retrieved documents. "
            f"{prompt.format(context=context, question=state.question)}"
        )
        return state.model_copy(
            update={"answer": fallback.content, "artifact": None}
        )

    def _structured_for(self, task: str, schema):
        if task not in self._structured:
            self._structured[task] = self.llm.with_structured_output(schema)
        return self._structured[task]

    def _build_context(self, state: State) -> str:
        metadata_parts = []
        for fname in state.source_files:
            meta = state.paper_metadata.get(fname)
            if meta:
                metadata_parts.append(f"  {Nodes._format_paper_metadata(fname, meta)}")
        metadata_block = (
            "\n".join(metadata_parts)
            if metadata_parts
            else "  (no structured metadata available)"
        )

        context_parts = []
        for i, doc in enumerate(state.retrieved_docs, start=1):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(f"[{i}] Source: {Path(source).name}\n{doc.page_content}")
        docs_block = "\n\n".join(context_parts) or "(no retrieved documents)"

        return (
            f"=== PAPER METADATA ===\n{metadata_block}\n\n"
            f"=== RETRIEVED DOCUMENTS ===\n\n{docs_block}"
        )

    # ----- rendering (schema -> markdown) -----

    def _render(self, task: str, artifact) -> str:
        if task == Intent.COMPARE.value and isinstance(artifact, ComparisonMatrix):
            return _matrix_markdown(artifact)
        if task == Intent.REVIEW.value and isinstance(artifact, RelatedWork):
            return _review_markdown(artifact)
        if task == Intent.GAPS.value and isinstance(artifact, ResearchGaps):
            return _gaps_markdown(artifact)
        return artifact.model_dump_json(indent=2)


def _esc(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ")


def _matrix_markdown(m: ComparisonMatrix) -> str:
    lines = [m.overview, "", "| Paper | Task | Method | Key results | Limitations |",
             "|---|---|---|---|---|"]
    for row in m.rows:
        lines.append(
            f"| {_esc(row.paper)} | {_esc(row.task)} | {_esc(row.method)} | "
            f"{_esc(row.key_results)} | {_esc(row.limitations)} |"
        )
    return "\n".join(lines)


def _review_markdown(m: RelatedWork) -> str:
    parts = [m.overview, ""]
    for group in m.theme_groups:
        parts.append(f"### {group.theme}")
        parts.append(group.text)
        parts.append("")
    return "\n".join(parts).strip()


def _gaps_markdown(m: ResearchGaps) -> str:
    parts = ["Research gaps identified across the corpus:", ""]
    for gap in m.gaps:
        parts.append(f"- **{gap.gap}** — evidence: {gap.evidence}")
    return "\n".join(parts)


_SYNTHESIS_PROMPTS = {
    Intent.COMPARE.value: (
        "You are a literature-review copilot. Using the retrieved documents, "
        "build a comparison matrix: one row per paper with its task, method, "
        "key results, and limitations. Set `paper` to the source filename. "
        "Do not invent facts absent from the documents.\n\n"
        "{context}\n\nQuestion: {question}"
    ),
    Intent.REVIEW.value: (
        "You are a literature-review copilot. Write a draft related-work "
        "section based on the retrieved documents. Group the papers into "
        "themes; in each group's prose, refer to papers by their source "
        "filename. Start with a short overview. Do not invent facts.\n\n"
        "{context}\n\nQuestion: {question}"
    ),
    Intent.GAPS.value: (
        "You are a literature-review copilot. Identify research gaps, "
        "unaddressed questions, or open problems across the retrieved "
        "documents. For each gap, cite the evidence (gaps the documents "
        "acknowledge or are silent on). Do not invent gaps with no "
        "support.\n\n{context}\n\nQuestion: {question}"
    ),
}