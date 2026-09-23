"""Suggested follow-up questions, Perplexity-style.

`generate_followups` asks the LLM for a few short, grounded follow-up
questions a user is likely to ask next given the answer + retrieved context,
and degrades to `deterministic_followups` (templates over paper metadata +
task type) when the LLM call is unavailable, rate-limited, or malformed —
so suggestions never break the chat flow.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Followups(BaseModel):
    questions: List[str] = Field(default_factory=list)


_FOLLOWUP_PROMPT = (
    "The user just received the ANSWER below for their current document "
    "collection. Write exactly three short follow-up questions they might "
    "naturally ask next, based only on the answer, the retrieved document "
    "text, and the paper metadata. Questions must be answerable from the "
    "documents (or a web search) and must not re-ask what was just "
    "answered. One question per item, concise, no numbering, no preamble.\n\n"
    "=== METADATA ===\n{metadata}\n\n"
    "=== RETRIEVED TEXT (truncated) ===\n{retrieved}\n\n"
    "=== ANSWER ===\n{answer}\n\n"
    "Returns the questions only."
)


def _metadata_summary(source_files, paper_metadata) -> str:
    lines = []
    for fname in source_files or []:
        meta = (paper_metadata or {}).get(fname)
        if meta is None:
            for key in paper_metadata or {}:
                if key == fname:
                    meta = paper_metadata[key]
                    break
        if meta is None:
            continue
        data = meta.model_dump() if hasattr(meta, "model_dump") else meta
        title = data.get("title") or fname
        authors = ", ".join(data.get("authors") or []) or "unknown"
        year = data.get("year")
        year_str = f" ({year})" if year else ""
        methods = ", ".join(data.get("methods") or []) or ""
        bits = f"{fname} — {title}{year_str}; authors: {authors}"
        if methods:
            bits += f"; methods: {methods}"
        lines.append(bits)
    return "\n".join(lines) if lines else "(none)"


def _retrieved_excerpt(result: Dict[str, Any], limit: int = 1200) -> str:
    docs = result.get("retrieved_docs") or []
    parts = []
    used = 0
    for doc in docs:
        source = Path(doc.metadata.get("source", "unknown")).name
        text = doc.page_content.strip()
        if not text:
            continue
        parts.append(f"[{Path(source).name}] {text}")
        used += len(text)
        if used >= limit:
            break
    chunk = "\n\n".join(parts)
    return chunk[:limit] if chunk else "(no retrieved documents)"


def deterministic_followups(
    source_files, paper_metadata, task_type: str = "qa"
) -> List[str]:
    """Template follow-ups from paper metadata + task type (zero LLM calls)."""
    metas = []
    for fname in source_files or []:
        meta = (paper_metadata or {}).get(fname)
        if meta is not None:
            metas.append((fname, meta))

    def title_of(meta) -> str:
        data = meta.model_dump() if hasattr(meta, "model_dump") else meta
        return data.get("title") or "(untitled paper)"

    def method_of(meta) -> Optional[str]:
        data = meta.model_dump() if hasattr(meta, "model_dump") else meta
        methods = data.get("methods") or []
        return methods[0] if methods else None

    questions: List[str] = []
    titles = [title_of(m) for _, m in metas]

    if task_type == "compare" and len(titles) >= 2:
        m = method_of(metas[0][1]) or method_of(metas[1][1])
        questions.append(
            f"How do {titles[0]} and {titles[1]} differ"
            + (f" on {m}?" if m else "?")
        )
    elif task_type == "gaps":
        questions.append(
            "What evidence supports each identified research gap?"
        )

    for fname, meta in metas:
        title = title_of(meta)
        questions.append(f"What are the key results of {title}?")
        questions.append(f"Who are the authors of {title}?")
        if len(questions) >= 3:
            break

    return questions[:3]


def generate_followups(
    result: Dict[str, Any], llm, source_files=None, paper_metadata=None
) -> List[str]:
    """LLM-generated follow-ups, degrading to deterministic on any failure.

    `source_files`/`paper_metadata` override the values inside `result` when
    the caller holds them separately (keeps this usable from the UI which
    passes both styles)."""
    if source_files is None:
        source_files = result.get("source_files") or []
    if paper_metadata is None:
        paper_metadata = result.get("paper_metadata") or {}

    metadata = _metadata_summary(source_files, paper_metadata)
    retrieved = _retrieved_excerpt(result)
    answer = result.get("answer") or ""

    try:
        chain = llm.with_structured_output(Followups)
        resp = chain.invoke(
            _FOLLOWUP_PROMPT.format(
                metadata=metadata, retrieved=retrieved, answer=answer
            )
        )
        if isinstance(resp, Followups):
            questions = resp.questions
        else:
            text = str(getattr(resp, "questions", resp))
            questions = [q.strip() for q in text.split("\n") if q.strip()]
        clean = [q for q in questions if isinstance(q, str) and q.strip()]
        if clean:
            return clean[:3]
    except Exception:
        pass

    return deterministic_followups(
        source_files, paper_metadata, result.get("task_type", "qa")
    )