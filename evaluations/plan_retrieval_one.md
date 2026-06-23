# plan: retrieval_one

## Approach: Force Retrieval First (Option B)

Retrieval becomes mandatory — the agent always retrieves documents before any Tavily call. Tavily is only used for information clearly missing from the documents (e.g., institution establishment years).

## Changes

### 1. Increase retriever breadth
- **chunk_size**: 500 → 1500 (capture full metadata per chunk)
- **k**: 4 → 10 (cover more chunks across all papers)

### 2. Restructure agent workflow
Replace the flat ReAct agent with a two-step process:
1. **Retrieve** → always fetch documents first (standalone step)
2. **Generate** → ReAct agent receives retrieved context in its prompt, can call Tavily only for *gaps* (like establishment years)

### 3. Stronger system prompt
- Explicitly state that retriever results are "the user's uploaded documents" and are ground truth
- Tavily is only for supplementary info not found in docs

### 4. Track retrieved_docs in state
- `agent_node` populates `retrieved_docs` so output is auditable

## Files to modify
- `src/document_ingestion/document_processor.py` — update chunk_size
- `src/vector_store/store.py` — update k in `as_retriever()`
- `src/nodes/react_node.py` — restructure agent flow, update prompt
- `src/graph_builder/graph.py` — may need to separate retrieve → agent nodes

## Verification
- Re-ingest the same 4 papers
- Ask the same question
- Expected: correct paper count (4), correct titles, correct authors, institutions with establishment years
