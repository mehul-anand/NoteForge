# fix: retrieval_one

## Root Cause

Agent was bypassing the retriever tool and calling `tavily_search` directly, returning web search results (real but irrelevant papers) instead of the user's actual uploaded documents.

## Changes Made

### 1. Document chunking — larger context per chunk

- **File**: `src/document_ingestion/document_processor.py`
- `chunk_size`: 500 → **1500**
- `chunk_overlap`: 50 → **150**
- _Why_: Larger chunks capture full paper metadata (authors, institute, title) in a single chunk, reducing fragmentation.

### 2. Retriever breadth — more results

- **File**: `src/vector_store/store.py`
- `as_retriever()` → `as_retriever(search_kwargs={"k": 10})`
- _Why_: k=10 ensures chunks from all 4 papers are returned, vs. k=4 which could miss entire papers.

### 3. Architecture — forced retrieval before agent

- **File**: `src/graph_builder/graph.py`
- **Before**: `START → agent (had retriever + tavily tools) → END`
- **After**: `START → retrieve (always runs) → agent (only tavily tool) → END`
- _Why_: Retrieval is now mandatory — no agent decision involved. The agent receives documents pre-fetched.

### 4. Agent toolset — retriever removed from agent

- **File**: `src/nodes/react_node.py`
- Removed the `retriever` tool from `_build_tools()` — agent only has `tavily_search`
- Added `retrieve_docs()` method for the graph's retrieve node
- `agent_node()` now receives `retrieved_docs` from state, passes them as `RETRIEVED DOCUMENTS` in the HumanMessage
- System prompt explicitly states: retrieved documents are ground truth; Tavily is for supplementary info only

### 5. Traceability — retrieved_docs preserved in output

- `agent_node` now returns `State(..., retrieved_docs=state.retrieved_docs, ...)`
- Makes output auditable

## Expected Behavior After Fix

1. Documents are always retrieved (k=10, 1500-char chunks)
2. Agent receives full document context in its prompt
3. Agent only calls Tavily for information clearly missing from docs (e.g., institution establishment years)
4. No more fabricated paper titles or wrong authors
