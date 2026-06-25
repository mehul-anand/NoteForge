"""RAG Workflow Nodes"""

from pathlib import Path
from typing import List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent

from src.state.state import State


class Nodes:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None

    def retrieve_docs(self, state: State) -> State:
        """
        Dedicated retrieval node — always runs before the agent.
        The agent never decides whether to retrieve; it always receives docs.
        """
        docs = self.retriever.invoke(state.question)
        retrieved_docs_arr = []
        for doc in docs:
            retrieved_docs_arr.append(Path(doc.metadata.get("source", " ")).name)
        retrieved_sources_set = set(retrieved_docs_arr)

        for filename in state.source_files:
            if filename not in retrieved_sources_set:
                stem = Path(filename).stem
                extra = self.retriever.invoke(stem)
                if extra:
                    docs.extend(extra[:2])  # taking out top 2 chunks

        return State(
            question=state.question,
            retrieved_docs=docs,
            source_files=state.source_files,  # pass through unchanged
            chat_history=state.chat_history,
        )

    def _build_tools(self) -> List[Tool]:
        """
        Agent only gets Tavily — for supplementary info not in uploaded docs
        (e.g. institution founding years, external facts).
        The retriever is NOT a tool here; it runs as a forced graph node instead.
        """
        tavily_tool = TavilySearch(
            max_results=3,
            description=(
                "Search the web for supplementary information clearly NOT present "
                "in the retrieved documents (e.g. year an institution was founded). "
                "Do NOT use this for anything already covered by the retrieved documents."
            ),
        )
        return [tavily_tool]

    def _build_agent(self):
        tools = self._build_tools()
        system_prompt = (
            "You are a RAG assistant. The user's uploaded documents have already been "
            "retrieved and are provided to you as 'RETRIEVED DOCUMENTS' below.\n\n"
            "Rules:\n"
            "1. The retrieved documents are ground truth — always answer from them first.\n"
            "2. The EXACT list of uploaded files is provided under 'UPLOADED FILES'. "
            "Use this as the definitive source for counting and listing documents — "
            "do NOT count references or citations mentioned within the documents.\n"
            "3. Use tavily_search ONLY for supplementary facts clearly absent from the "
            "retrieved documents (e.g. year an institution was founded, current stock price).\n"
            "4. Do NOT use tavily_search for anything already present in the documents.\n"
            "5. If the retrieved documents lack enough information, say so — do not invent facts."
        )
        self._agent = create_react_agent(self.llm, tools=tools, prompt=system_prompt)

    def agent_node(self, state: State) -> State:
        """
        Agent node — receives pre-retrieved docs from state,
        formats them as context, and calls the LLM.
        """
        if self._agent is None:
            self._build_agent()

        context_parts = []
        for i, doc in enumerate(state.retrieved_docs, start=1):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(
                f"[{i}] Source: {Path(source).name}\n{doc.page_content}"
            )
        context = "\n\n".join(context_parts)

        # Inject the ground-truth file list so the agent always knows what was
        # uploaded, even if retrieval didn't return chunks from every document.
        uploaded_files_str = (
            ", ".join(state.source_files)
            if state.source_files
            else "unknown (not provided)"
        )

        message = (
            f"=== UPLOADED FILES ===\n"
            f"The user uploaded exactly these {len(state.source_files)} file(s): "
            f"{uploaded_files_str}\n\n"
            f"=== RETRIEVED DOCUMENTS ===\n\n"
            f"{context}\n\n"
            f"=== QUESTION ===\n\n"
            f"{state.question}"
        )

        history_messages = []
        for msg in state.chat_history:
            if msg["role"] == "user":
                history_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history_messages.append(AIMessage(content=msg["content"]))

        result = self._agent.invoke(
            {"messages": [*history_messages, HumanMessage(content=message)]}
        )
        messages = result.get("messages", [])

        answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                answer = msg.content
                break

        return State(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            source_files=state.source_files,
            answer=answer,
        )
