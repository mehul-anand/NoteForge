# fix: issue_two

## Problem

Agent answers each question in isolation — no memory of prior turns. Follow-ups like "hehe info" (a trigger set in a previous turn) failed because the agent had no conversation context.

Additionally, compound questions like "intro + authors + institutes + year for all papers" returned uneven retrieval — chunks from some documents were missed entirely.

## Root Cause

- `streamlit_app.py` stored chat history in session state for UI display but never passed it to the graph/LLM.
- `State` had no `chat_history` field.
- `agent_node` sent only the current question to the LLM via ReAct agent.
- A single retrieval call with a compound query produced diluted embeddings — some papers were well-covered, others had no matching chunks.

## Implementation

### 1. State — added chat_history field

- **File**: `src/state/state.py`
- Added `chat_history: List[Dict[str, str]] = []` to State
- Stores previous Q&A turns as simple `{"role": "..." , "content": "..."}` dicts

### 2. Graph — thread chat_history through run()

- **File**: `src/graph_builder/graph.py`
- `run()` now accepts `chat_history` param and passes it to initial State

### 3. Agent — prepend history to LLM calls

- **File**: `src/nodes/react_node.py`
- Converts `state.chat_history` dicts → LangChain `HumanMessage`/`AIMessage`
- Prepends them before the current question in `agent_node`
- Carries `chat_history` through `retrieve_docs` return (was being dropped)
- Carries `chat_history` through `agent_node` return (was being dropped)

### 4. Streamlit — pass history to graph

- **File**: `streamlit_app.py`
- Passes `st.session_state.messages[:-1]` as `chat_history` to `graph.run()`
- Excludes the current user prompt (it's already in `state.question`)

### 5. System prompt — guide agent on history use

- **File**: `src/nodes/react_node.py`
- Added rule 6: use chat history for follow-ups, attribute sources honestly
- Softened rule 1 to avoid claiming Tavily-sourced info was "in the documents"

### 6. Query decomposition — targeted retrieval for complex questions

- **File**: `src/nodes/react_node.py`, `src/state/state.py`, `src/graph_builder/graph.py`
- Added `expand_query` node: LLM decomposes compound questions into focused sub-queries
- `retrieve_docs` now iterates over sub-queries and deduplicates results
- Architecture: `START → expand_query → retrieve → agent → END`
- Simple questions pass through unchanged (single sub-query)

### 7. Removed reference chunk filter — generalized bot behavior

- **File**: `src/document_ingestion/document_processor.py`
- Removed `_is_reference_chunk` that silently dropped chunks where >50% of
  lines started with `[` (academic bibliography format)
- All content now indexed as-is; LLM naturally handles reference sections

### 8. Tavily fallback chain — prevent "don't know" before trying web search

- **File**: `src/nodes/react_node.py`
- Added rule 0: forced fallback chain — try Tavily for any external/real-time
  information before saying "I don't know"
- Prevents the agent from stopping at "not in documents → give up" for queries
  like weather, time, and current events

## Verification

- **Chat history test:**
  - Q1: "okay whenever I say hehe info, you give me the name of the authors of the HyDE paper"
  - A: Correctly understood and agreed
  - Q2 (2 turns later): "hehe info"
  - A: Correctly recalled the trigger and returned HyDE authors (Haowei Lin, Zihao Wang, Yitong Lu, Jianwei Yin, Yang Xu)
  - Expected: history flows through → agent remembers prior turn instructions across multiple intervening exchanges

- **Query decomposition test:**
  - Q: "Give me a brief intro about all the papers I gave, also how many papers did I give ? can you also tell me the institutes involved in each paper and the authors there and finally you give me the year of establishment of each of these institutions"
  - A: Correctly returned all 4 papers. Paper 1 had full authors + Institute (IISc 1909). Papers 2-3 had partial authors. Paper 4 authors missing. Institute names + establishment years for IISc (1909), UC Berkeley (1868), and Allen AI (2014) retrieved via Tavily.
  - Expected: sub-queries improved coverage vs single-query retrieval, but some author info still missed — decomposition prompt should be made aware of source_files for per-document sub-query generation

- **Tavily fallback test:**
  - Q: "what is the time and weather now?"
  - Before fix: "I don't have that information in the retrieved documents"
  - After fix: Tavily returns current weather/time
