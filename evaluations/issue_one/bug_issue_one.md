# bug: retrieval_one

## Question Asked

> Give me a brief intro about all the papers I gave, also how many papers did I give ? can you also tell me the institutes involved in each paper and the authors there and finally you give me the year of establishment of each of these institutions

## Papers Actually Uploaded (4)

1. Control Barrier Functions in Dynamic UAVs for Kinematic Obstacle Avoidance: A Collision Cone Approach
2. Collision Cone Control Barrier Functions for Kinematic Obstacle Avoidance in UGVs
3. Corrective Retrieval Augmented Generation (CRAG)
4. Never Come Up Empty: Adaptive HyDE Retrieval for Improving LLM Developer Support

## Output Observed (wrong)

Returned only 3 papers, none matching the uploaded ones:

- _Factscore: Fine-grained atomic evaluation of factual precision in long form text generation_ — NOT uploaded
- _I Know What You Are Searching For: Code Snippet Recommendation from Stack Overflow Posts_ — NOT uploaded
- _Never Come Up Empty: Adaptive HyDE Retrieval..._ — correct title, but **wrong authors** (listed Paul Barham et al. instead of Fangjian Lei et al.)

Institution data was also hallucinated (e.g., "Association for Computational Linguistics" appearing for multiple papers).

## Root Cause Analysis

1. **Agent bypassed retriever** — `gpt-4o-mini` called `tavily_search` instead of the `retriever` tool. The system prompt says "try retriever first" but the model treated it as optional.
2. **k=4 too low** — default retriever returns 4 chunks; with 4 papers chunked at 500 chars, chunks from some papers may not be returned.
3. **No retrieval guardrails** — Tavily results (real papers from the web) were accepted as ground truth with no cross-check against actual documents.
4. **No traceability** — `agent_node` doesn't populate `retrieved_docs` in State, making it impossible to audit what was actually retrieved.
