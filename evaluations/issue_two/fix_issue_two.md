# fix: issue_two

## Problem

Agent answers each question in isolation — no memory of prior turns. Follow-ups like "hehe info" (a trigger set in a previous turn) failed because the agent had no conversation context.

## Root Cause

- `streamlit_app.py` stored chat history in session state for UI display but never passed it to the graph/LLM.
- `State` had no `chat_history` field.
- `agent_node` sent only the current question to the LLM via ReAct agent.

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

### 4. Streamlit — pass history to graph

- **File**: `streamlit_app.py`
- Passes `st.session_state.messages[:-1]` as `chat_history` to `graph.run()`
- Excludes the current user prompt (it's already in `state.question`)

### 5. System prompt — guide agent on history use

- **File**: `src/nodes/react_node.py`
- Added rule 6: use chat history for follow-ups, attribute sources honestly
- Softened rule 1 to avoid claiming Tavily-sourced info was "in the documents"

## Verification

- Ingest papers, set up a multi-turn test:
  - Q1: "okay whenever I say hehe info, you give me the name of the authors of the HyDE paper"
  - A: Correctly understood and agreed
  - Q2 (2 turns later): "hehe info"
  - A: Correctly recalled the trigger and returned HyDE authors (Haowei Lin, Zihao Wang, Yitong Lu, Jianwei Yin, Yang Xu)
  - Expected: history flows through → agent remembers prior turn instructions across multiple intervening exchanges
