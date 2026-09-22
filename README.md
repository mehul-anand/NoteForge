# NoteForge

- after running some experiments in my private RAG repos I am making this one from scratch in public, showcasing all the steps I am taking
- I will also work on optimisng the outputs and reducing the hallucinations

- this file will serve as my personal documentation

### Todos:

#### Todos for v1:

- [done] Initialise a project

- [done] Project Structure

- [done] Document pre-processing

- [done] Vectorstore

- [done] State

- [done] Nodes and Graph structure
  - designing the whole workflow

- [done] deploy the basic version

#### Todos for v2:

- [done] Tavily
  - replaced Wikipedia

- [done] Evaluation
  - [done] Using Notebooklm to test the outputs for now
  - [done] Tool selection issue found and fixed (agent bypassed retriever for Tavily)
  - [done] Created Evaluations/ with bug → plan → fix workflow

- [done] Corrective strategies
  - [done] Implement MMR
    - in `./src/vector_store/store.py` : `fetch_k = 30` for fetching 30 candidates and `lambda_mult = 0.7` to balance relevancy v/s diversity (0 -> pure diversity, 1 -> pure similarity)
  - [done] Chat history — agent remembers conversation context across turns
  - [done] Query decomposition — compound questions broken into sub-queries
    for full document coverage
  - [done] Source file injection — ground-truth file list fed to agent
    for accurate doc counting
  - [done] Tavily fallback — agent tries web search before giving up
  - [done] File-aware decomposition — per-document sub-queries
    using uploaded file names
  - [done] Per-document LLM summaries — one-shot extraction of title/authors/topic
    from first page of each PDF, injected into agent context directly
    (bypasses embedding retrieval for metadata queries)
  - [done] Query rewriting — new node between decomposition and retrieval
    that expands abbreviations (CBF -> Control Barrier Function) and domain terms
    to improve embedding similarity
  - [done] User document upload — st.file_uploader replaces hardcoded data/ scan;
    users upload PDFs via sidebar, process on demand, temp files cleaned after
  - [done] Evaluation framework — ground_truth.json (12 questions) + run_eval.py
    computes Precision@15, Recall@15, MRR, factual accuracy.
    Baseline: 0.958 MRR, 0.875 recall, 0.875 factual accuracy

- [next] Richer metric formulas — chunk-level precision/recall instead of document-level
  (needs per-query chunk labels in ground truth)
- [next] Doc summary prompt improvement — current 1-sentence summary truncates 9+ author lists
- [next] Conditional safety net — only run retrieve_docs safety net (lines 95-102) when there are multiple sub-queries; skip for simple queries to avoid pulling irrelevant chunks
