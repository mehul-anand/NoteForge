# fix: issue_one

## Problem

Agent was bypassing the retriever tool and calling `tavily_search` directly, returning web search results (real but irrelevant papers) instead of the user's actual uploaded documents.

**Result**: Returned 3 papers, none matching the uploaded ones — fabricated titles, wrong authors, hallucinated institutions.

## Root Cause

1. **Agent bypassed retriever** — `gpt-4o-mini` called `tavily_search` instead of the `retriever` tool. The system prompt said "try retriever first" but the model treated it as optional.
2. **k=4 too low** — default retriever returns 4 chunks; with 4 papers chunked at 500 chars, chunks from some papers may be missed entirely.
3. **No retrieval guardrails** — Tavily results were accepted as ground truth with no cross-check against actual documents.
4. **No traceability** — `agent_node` didn't populate `retrieved_docs` in State, making it impossible to audit what was retrieved.

## Plan

**Approach: Force Retrieval First** — retrieval becomes mandatory before any Tavily call. Tavily is only used for information clearly missing from the documents (e.g., institution establishment years).

## Implementation

### 1. Document chunking — larger context per chunk

- **File**: `src/document_ingestion/document_processor.py`
- `chunk_size`: 500 → **1500**
- `chunk_overlap`: 50 → **150**
- *Why*: Larger chunks capture full paper metadata (authors, institute, title) in a single chunk, reducing fragmentation.

### 2. Retriever breadth — more results

- **File**: `src/vector_store/store.py`
- `as_retriever()` → `as_retriever(search_kwargs={"k": 10})`
- *Why*: k=10 ensures chunks from all 4 papers are returned, vs. k=4 which could miss entire papers.

### 3. Architecture — forced retrieval before agent

- **File**: `src/graph_builder/graph.py`
- **Before**: `START → agent (had retriever + tavily tools) → END`
- **After**: `START → retrieve (always runs) → agent (only tavily tool) → END`
- *Why*: Retrieval is now mandatory — no agent decision involved. The agent receives documents pre-fetched.

### 4. Agent toolset — retriever removed from agent

- **File**: `src/nodes/react_node.py`
- Removed the `retriever` tool from `_build_tools()` — agent only has `tavily_search`
- Added `retrieve_docs()` method for the graph's retrieve node
- `agent_node()` now receives `retrieved_docs` from state, passes them as `RETRIEVED DOCUMENTS` in the HumanMessage
- System prompt explicitly states: retrieved documents are ground truth; Tavily is for supplementary info only

### 5. Traceability — retrieved_docs preserved in output

- `agent_node` now returns `State(..., retrieved_docs=state.retrieved_docs, ...)`
- Makes output auditable

## Verification

- Re-ingest the same 4 papers
- Ask: *"Give me a brief intro about all the papers I gave, also how many papers did I give?..."*
- Expected: correct paper count (4), correct titles, correct authors, institutions with establishment years
