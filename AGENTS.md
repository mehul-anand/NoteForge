# AGENTS.md

Guidance for AI agents and humans working in this repo. Read `notes/system_map.md`
for the full architecture walkthrough; this file is the operational cheat-sheet.

## What this is

NoteForge — a RAG (retrieval-augmented generation) pipeline on LangGraph.
Upload PDFs (or URLs) → chunks + FAISS/MMR index + per-document structured
metadata → question → `route_intent → expand_query → rewrite_queries → retrieve
→ agent(ReAct + Tavily) | synthesize (matrix/related-work/gaps)`.
Product direction: Literature Review Copilot (see `notes/new_direction.md`).

## TODO (next)

- **Profile identity verification (`_verify_profile`)**: every Scholar/LinkedIn
  URL presented for a deterministically-resolved author must be gated — fetch
  the profile page, normalize its displayed principal name, and require equality
  with the resolved author (from PAPER METADATA). Mismatch → suppress that link,
  say "could not be verified; I don't link profiles that don't clearly identify
  the exact person" (rule 11: never substitute, incl. a co-author of the same
  paper — the LITM→Michele Bevilacqua Scholar substitution case). Name-equality
  only by default. ~2 Tavily calls/authors (verify fallback), plus the linked
  URL binding for these profiles must use the full URL as the source key.
- **B — per-intent fan-out + merge** (see `notes/new_direction.md` "Known
  future: B"): after `expand_query`, route each sub-question to its best node
  (agent / synthesize) and merge answers+artifacts. Option A (mixed question →
  agent) shipped as a stopgap; B is the split/merge graph that keeps structured
  artifacts for mixed questions.
- Phase 2.5: chat compaction (`compact_history`, >8 turns → LLM summary into
  `condensed`) + rate limiting (`UsageGate`: MAX_LLM_CALLS_PER_MIN=60,
  MAX_LLM_CALLS_PER_SESSION=300, MAX_MODERATIONS_PER_MIN=30, wrapped Tavily).

## Commands

```bash
# run the app (local UI)
streamlit run streamlit_app.py

# run the evaluation suite (baseline vs current, 12 questions, ~2min, needs .env keys)
.venv/bin/python3 evaluations/issue_five/run_eval.py

# lint (ruff is not in .venv; run via uv)
uv tool run ruff check src/ evaluations/
uv tool run ruff check --select F src/                # real-bug check only

# tests
.venv/bin/python3 -m pytest test/ -q

# deps
uv sync                      # after pyproject change
```

Python 3.11 (`.python-version`). Env keys in `.env` (copy `.env.example`,
never commit `.env`). LLM: `openai:gpt-4o-mini`.

## Architecture map

```
streamlit_app.py            UI: upload/URL ingest, chat loop, moderation gate
src/config/config.py        constants, get_llm() cached, moderation()
src/state/state.py          pydantic State flowing through graph
src/document_ingestion/     loaders (PyMuPDF/Text/WebBase), chunking,
                            PaperMetadata structured extraction
src/vector_store/store.py   FAISS + MMR retriever (k=15, fetch_k=30, λ=0.7)
src/nodes/react_node.py     the 4 graph nodes + agent system prompt
src/nodes/synthesis.py      intent router (qa/compare/review/gaps) + synthesis node
                            (Pydantic artifacts > rendered markdown)
src/graph_builder/graph.py  node wiring (route_intent → expand → rewrite →
                            retrieve → agent | synthesize)
prompts/agent_v1.md         versioned default system prompt (env-overridable)
evaluations/issue_five/     ground_truth.json (12 Qs) + run_eval.py
old_version/                pre-change snapshots (rollback copies; gitignored)
notes/                      design docs + interview notes (gitignored)
```

## Conventions

- **Change one thing at a time.** Test with one `graph.run()` call, then run the
  full eval if it matters. `notes/how_to.md` has the workflow.
- **Before editing** core files, copy them to `old_version/<change>/` for
  rollback (gitignored — they never pollute the repo).
- **Node state threading:** every node must pass through ALL State fields it
  receives (`question`, `source_files`, `paper_metadata`, `chat_history`, …).
  Forgetting one silently drops context.
- **Prompt changes** go in `prompts/` (versioned). Override chain:
  `AGENT_SYSTEM_PROMPT` env > `prompts/agent_v1.md` > bundled default string.
- **Keep production prompts private:** default lives in repo (showcase);
  deployed instance may override via env / `st.secrets` without touching code.
- **URLs** are treated as first-class sources alongside PDFs (WebBaseLoader).
- Add new evaluation questions to `ground_truth.json` + bump `run_eval.py`,
  and keep the LLM-judge metric (noisy ±0.1 — run 2-3x for stable numbers).

## GenAI specifics

- Embeddings `text-embedding` (OpenAI), retrieval is pure math (no LLM). LLM
  used only in decomposition/rewriting/agent/judge (~5-7 calls/question).
- **Structured output:** `PaperMetadata` via `.with_structured_output()`,
  single retry, then degraded 1-line-summary fallback.
- **LangSmith tracing:** set `LANGCHAIN_TRACING_V2=true`,
  `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` to trace graph runs. Off by default.
- **Moderation gate:** user input is screened before entering the pipeline;
  flagged → canned refusal. Documents are treated as untrusted data (the agent
  system prompt refuses instructions found inside uploaded content).