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

    def expand_query(self, state: State) -> State:
        files_str = ", ".join(state.source_files) if state.source_files else "unknown"
        prompt = (
            "You are a query decomposition assistant. Break the user's "
            "question into simple, self-contained sub-questions — one per "
            "line. Each sub-query should target a single piece of information "
            "that can be retrieved from a document.\n\n"
            "The user has uploaded these files:\n"
            "{files}\n\n"
            "If the question mentions specific documents or topics, generate "
            "one sub-query per file that is likely relevant. Always include "
            "the filename explicitly in the sub-query where applicable.\n\n"
            "If the question is already simple and focused, return it unchanged "
            "(one line).\n\n"
            "User question: {question}"
        )
        response = self.llm.invoke(
            prompt.format(files=files_str, question=state.question)
        )
        sub_queries = [
            q.strip() for q in response.content.strip().split("\n") if q.strip()
        ]
        return State(
            question=state.question,
            sub_queries=sub_queries,
            source_files=state.source_files,
            doc_summaries=state.doc_summaries,
            chat_history=state.chat_history,
        )

    def retrieve_docs(self, state: State) -> State:
        """
        Dedicated retrieval node — always runs after expand_query.
        Iterates over sub_queries for comprehensive document coverage.
        """
        all_docs = []
        seen = set()
        for query in state.sub_queries:
            docs = self.retriever.invoke(query)
            for doc in docs:
                sig = doc.page_content[:200]
                if sig not in seen:
                    seen.add(sig)
                    all_docs.append(doc)

        retrieved_sources = set()
        for doc in all_docs:
            retrieved_sources.add(Path(doc.metadata.get("source", " ")).name)
        for filename in state.source_files:
            if filename not in retrieved_sources:
                extra = self.retriever.invoke(Path(filename).stem)
                if extra:
                    all_docs.extend(extra[:2])

        return State(
            question=state.question,
            retrieved_docs=all_docs,
            source_files=state.source_files,
            doc_summaries=state.doc_summaries,
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
            "0. Fallback chain: always try tavily_search for external or real-time "
            "information BEFORE saying you don't know. If the question is about weather, "
            "time, current events, or any external fact not in the documents, search the "
            "web first — only say 'I don't know' if Tavily also returns nothing.\n"
            "1. Answer primarily from the retrieved documents — they are your primary "
            "source of truth.\n"
            "2. The EXACT list of uploaded files is provided under 'UPLOADED FILES'. "
            "Use this as the definitive source for counting and listing documents — "
            "do NOT count references or citations mentioned within the documents.\n"
            "3. Use tavily_search for ANY real-time, external, or world knowledge not present "
            "in the retrieved documents — including weather, time, current events, "
            "institution founding years, stock prices, etc. If the question is completely "
            "unrelated to the documents (e.g. weather, news, general trivia), skip retrieval "
            "and use tavily_search.\n"
            "4. Do NOT use tavily_search for anything already present in the documents.\n"
            "5. If the retrieved documents lack enough information, say so — do not invent facts.\n"
            "6. Chat history contains previous Q&A turns — use it for conversational "
            "follow-ups and context. When referencing information from past answers, be "
            "honest about its source: if it came from Tavily (web search), do NOT claim "
            "it was in the documents — state that it was obtained via web search."
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

        # Inject doc summaries so the agent always sees metadata (authors,
        # titles, doc type) for every file regardless of retrieval quality.
        summaries_parts = []
        for fname in state.source_files:
            summary = state.doc_summaries.get(fname, "")
            if summary:
                summaries_parts.append(f"  {fname}: {summary}")
        summaries_block = (
            "\n".join(summaries_parts)
            if summaries_parts
            else "  (no summaries available)"
        )

        uploaded_files_str = (
            ", ".join(state.source_files)
            if state.source_files
            else "unknown (not provided)"
        )

        message = (
            f"=== DOCUMENT SUMMARIES ===\n"
            f"{summaries_block}\n\n"
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
            doc_summaries=state.doc_summaries,
            chat_history=state.chat_history,
            answer=answer,
        )
