"""RAG Workflow with a ReAct Agent"""

from typing import List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent

from src.state.state import State


class Nodes:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None

    def _build_tools(self) -> List[Tool]:
        def retriever_tool_func(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found"
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title} \n {d.page_content}")
            return "\n\n".join(merged)

        retrieval_tool = Tool(
            name="retriever",
            description="Fetch passages from uploaded documents (PDFs, web pages, text files). Use this FIRST for questions about ingested content.",
            func=retriever_tool_func,
        )
        tavily_tool = TavilySearchResults(
            max_results=3,
            description="Search the web for current or external information. Use this ONLY if the retriever finds nothing relevant, or if the question asks about live/up-to-date info.",
        )
        return [retrieval_tool, tavily_tool]

    def _build_agent(self):
        tools = self._build_tools()
        system_prompt = (
            "You are a RAG agent. Always try the 'retriever' tool first for questions about the user's documents. "
            "If the retriever returns nothing relevant, or if the question asks for current/live information, "
            "fall back to 'tavily_search' to search the web. "
            "Once you have enough information, provide a clear answer. "
            "If neither tool yields useful information, say so."
        )
        self._agent = create_react_agent(self.llm, tools=tools, prompt=system_prompt)

    def agent_node(self, state: State) -> State:
        if self._agent is None:
            self._build_agent()
        result = self._agent.invoke(
            {"messages": [HumanMessage(content=state.question)]}
        )
        messages = result.get("messages", [])
        answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                answer = msg.content
                break
        return State(question=state.question, answer=answer)
