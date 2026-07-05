# eval: issue_four

## Idea

Before this, every change was validated by eyeballing it. "Looks right" or "sounds better" — no numbers. We couldn't tell if a change actually improved things or just felt like it did.

### Ground truth dataset (`ground_truth.json`)

12 questions across 4 groups:

| Category             | What it tests                                                       |
| -------------------- | ------------------------------------------------------------------- |
| Simple lookups       | Can the retriever find authors for a single paper?                  |
| Abbreviation queries | Does CBF get expanded to Control Barrier Function before retrieval? |
| Metadata lookups     | Can it extract title and year correctly?                            |
| Cross-document       | Can it pull info from multiple papers at once?                      |

Each question has a known correct answer and a list of which files should contribute chunks.

### Metrics

| Metric           | Range | What it actually measures                                         |
| ---------------- | ----- | ----------------------------------------------------------------- |
| Precision@15     | 0-1   | Out of 15 chunks we grabbed, how many came from the right papers? |
| Recall@15        | 0-1   | Out of all papers that matter, how many showed up in those 15?    |
| MRR              | 0-1   | How early did the first good chunk appear?                        |
| Factual accuracy | 0-1   | LLM checks if our answer matches ground truth                     |

Note: precision and recall are document-level, not chunk-level.

#### Evaluation script (`run_eval.py`)

## Results

### Aggregate

| Metric               | Score |
| -------------------- | ----- |
| Avg Precision@15     | 0.694 |
| Avg Recall@15        | 0.875 |
| Avg MRR              | 0.958 |
| Avg Factual Accuracy | 0.875 |

### Per question

| Question            | P@15  | R@15 | MRR | Factual | Notes                                                                     |
| ------------------- | ----- | ---- | --- | ------- | ------------------------------------------------------------------------- |
| hyde_authors        | 0.733 | 1.0  | 1.0 | 1.0     | Correct                                                                   |
| crag_authors        | 0.733 | 1.0  | 1.0 | 1.0     | Correct                                                                   |
| cbf_uav_authors     | 0.733 | 1.0  | 1.0 | 1.0     | Correct                                                                   |
| cbf_ugv_authors     | 0.467 | 1.0  | 1.0 | 0.5     | Partial — returned Phani Thontepu and colleagues instead of all 9 authors |
| hyde_title          | 0.800 | 1.0  | 1.0 | 1.0     | Correct                                                                   |
| crag_title          | 0.733 | 1.0  | 1.0 | 1.0     | Correct                                                                   |
| cbf_uav_year        | 0.667 | 1.0  | 1.0 | 1.0     | Correct                                                                   |
| cbf_ugv_year        | 0.333 | 1.0  | 1.0 | 1.0     | Correct despite low precision                                             |
| institutes_involved | 1.000 | 0.5  | 1.0 | 0.5     | Partial — missed CBF_UAV institute, guessed some CRAG affiliations        |
| how_many_papers     | 1.000 | 0.5  | 1.0 | 1.0     | Correct via doc summaries                                                 |
| ugv_abstract        | 0.133 | 1.0  | 0.5 | 1.0     | Correct despite very low precision                                        |
| all_papers_compound | 1.000 | 0.5  | 1.0 | 0.5     | Partial — CBF_UGV title wrong, authors truncated                          |

### What this tells us

1. **MRR (0.958)** — When a relevant chunk exists, it shows up early. Rank 1 or 2 almost always.

2. **Recall (0.875)** — Single-paper queries always find their target. Cross-doc queries (4 papers) usually find 2 of 4 in top-15.

3. **Precision (0.694)** — About 30% of what we retrieve is noise from unrelated papers. Expected with MMR diversity and k=15.

4. **Doc summaries are carrying the system** — Look at `ugv_abstract`: precision was 0.133 (nearly useless retrieval) but factual accuracy was 1.0. The doc summary injected the correct info directly.

5. **The bottleneck is still doc summaries** — All 3 partial answers trace back to the same problem: the one-sentence summary truncates author lists and paraphrases titles. CBF_UGV has 9 authors — a single sentence can't fit them all.

### Historical comparison

| Stage                   | hyde_authors                  | crag_authors                   | cbf_ugv_authors      | all_papers  |
| ----------------------- | ----------------------------- | ------------------------------ | -------------------- | ----------- |
| Pre-fix (baseline)      | Not specified                 | Not specified                  | Not specified        | All missing |
| Post-architecture fix   | Trovato et al. (hallucinated) | Sewon Min et al. (wrong paper) | Not specified        | Wrong       |
| Post decomp + summaries | Correct                       | Correct                        | Phani and colleagues | Partial     |
| Current eval snapshot   | 1.0 factual                   | 1.0 factual                    | 0.5 factual          | 0.5 factual |

## Next

- **Doc summary prompt** — too aggressive for 9+ authors. Fixing this directly addresses all 3 partial-scoring questions.
- **Precision tuning** — lowering lambda_mult or k could reduce noise at the cost of some recall.
- **Better metric formulas** — chunk-level precision/recall would be more accurate but needs per-query chunk labels, which is extra work.
