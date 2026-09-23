# Issue Five: Baseline vs Current — Quantitative Comparison

## Problem

How much has the RAG pipeline actually improved since the initial implementation? The pre-Claude version (`old_version/pre-claude/`) represents the "basic standard RAG" baseline — a simple retrieval-augmented generation pipeline with no agent loop, no query decomposition, no MMR, no doc summaries, and no source file awareness. Measure the improvement across retrieval and generation metrics.

## Baseline Architecture

```
retriever → responder → END
```

- **Retriever**: FAISS similarity search, k=8 (no MMR)
- **Responder**: single LLM call with context, no agent, no tools
- **No** query decomposition, rewriting, doc summaries, source file injection, or chat history

## Current Architecture

```
route_intent → expand_query → rewrite_queries → retrieve (×sub-queries)
    → agent_node (ReAct + Tavily) | synthesize (matrix/related-work/gaps) → END
```

- **Retriever**: FAISS MMR, k=15, fetch_k=30, lambda_mult=0.7
- **Agent**: ReAct loop with Tavily web search fallback
- **Synthesize**: intent router (`qa`/`compare`/`review`/`gaps`) sends non-qa
  questions to a structured-artifact synthesis node instead of the agent
- **Includes**: intent routing, query decomposition, file-aware sub-queries,
  abbreviation rewriting, structured per-paper metadata (PaperMetadata), source
  file ground-truth injection

## Results

Both pipelines evaluated on the same 12 questions from `ground_truth.json`,
using k=8 for both to ensure fair metric comparison.

Numbers below are from the committed run in `comparison_results.json`
(2026-09-23). The LLM judge is noisy (±0.1) — run 2–3× and report the band;
a re-run moved current factual from 1.0 to 0.958.

| Metric | Baseline | Current | Change |
|---|---|---|---|
| Precision@8 | 0.812 | 0.688 | **-15.3%** |
| Recall@8 | 0.771 | 0.875 | **+13.5%** |
| MRR | 0.917 | 0.958 | **+4.5%** |
| Factual Accuracy | 0.542 | 0.958 | **+77%** |

### Per-question breakdown

| Question | Baseline Factual | Current Factual | Delta |
|---|---|---|---|
| hyde_authors | 1.0 | 1.0 | — |
| crag_authors | 1.0 | 1.0 | — |
| cbf_uav_authors | 1.0 | 1.0 | — |
| cbf_ugv_authors | 0.5 | 1.0 | +0.5 |
| hyde_title | 1.0 | 1.0 | — |
| crag_title | **0.0** | 1.0 | +1.0 |
| cbf_uav_year | 1.0 | 1.0 | — |
| cbf_ugv_year | 1.0 | 1.0 | — |
| institutes_involved | **0.0** | 0.5 | +0.5 |
| how_many_papers | **0.0** | 1.0 | +1.0 |
| ugv_abstract | **0.0** | 1.0 | +1.0 |
| all_papers_compound | **0.0** | 1.0 | +1.0 |

## Analysis

### What improved (and why)

**Factual accuracy (+77%)** — The baseline scored 0.0 on 5 of 12 questions; the
current pipeline fixes all but one fully (institutes_involved is partial, 0.5):

- **ugv_abstract (0.0 → 1.0)**: Baseline had precision@8 = 0.0 — zero retrieved chunks from the UGV paper. The current system's structured paper metadata injection provides title/abstract/authors independently of retrieval quality.
- **all_papers_compound (0.0 → 1.0)**: Baseline returned hallucinated references (Llama 2, GPT-4, etc.) from reference sections. The current system's file-aware decomposition generates per-file sub-queries, and the agent uses the uploaded files list as ground truth.
- **how_many_papers (0.0 → 1.0)**: Baseline had no concept of the file list — it could only answer from retrieved chunks. Source file injection gives the current system ground-truth knowledge.
- **crag_title (0.0 → 1.0)**: The CRAG paper title was not in the top-8 chunks for the baseline. The current system benefits from MMR diversity and structured metadata.
- **cbf_ugv_authors (0.5 → 1.0)**: The baseline's 1-sentence summary truncated the 9-author list. Structured PaperMetadata captures the full author list in order.
- **institutes_involved (0.0 → 0.5)**: Baseline had no institute information at all. Current system gets IISc coverage through metadata + Tavily fallback, but still misses the CRAG (USTC) and HyDE (Queen's/RMCC) institutes.

### What regressed (and why)

**Precision@8 (-15.3%)** — The baseline's strict similarity search returns only the most similar chunks, so its precision is higher. MMR intentionally diversifies results (lambda_mult=0.7), bringing in chunks from different sources at the cost of some precision. This is a known tradeoff: MMR sacrifices precision for recall.

### What stayed the same

Everything baseline could answer (1.0), current still answers (1.0) — no
regressions on generation quality. The only imperfection left is
**institutes_involved (0.5)**: the current run covers IISc but the CRAG and HyDE
institutes were not recovered (the retrieval coverage safety net + Tavily lookups
did not resolve them this run).

## Future Work

1. **institutes_involved (0.5) and judge noise** — the remaining imperfect score
   needs either better institute extraction in PaperMetadata (affiliations
   already captured) or a more robust Tavily / retrieval coverage fallback.
   Re-run the suite 2–3× and report the band, since current factual has
   measured 1.0 and 0.958 across runs (±0.1 judge noise).
2. **Chunk-level metrics** — current precision/recall are document-level (is the chunk from the right file?). True chunk-level evaluation would need per-question chunk ID labels in ground truth.
3. **Precision tuning** — lambda_mult could be increased (closer to 1.0) to reduce MMR diversity if precision matters more than recall for the target use case.
