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

    def retrieve_docs(self, state: State) -> State:
        docs = self.retriever.invoke(state.question)
        return State(question=state.question, retrieved_docs=docs)

    def _build_tools(self) -> List[Tool]:
        tavily_tool = TavilySearchResults(
            max_results=3,
            description="Search the web for current or external information. Use this for supplementary info not found in the user's documents (e.g., establishment year of an institution).",
        )
        return [tavily_tool]

    def _build_agent(self):
        tools = self._build_tools()
        system_prompt = (
            "You are a RAG agent. The user's uploaded documents have already been retrieved "
            "and are provided to you as 'RETRIEVED DOCUMENTS' below.\n\n"
            "Rules:\n"
            "1. The 'RETRIEVED DOCUMENTS' are from the user's actual uploaded files — they are ground truth.\n"
            "2. Always answer based on these documents first.\n"
            "3. Use 'tavily_search' ONLY for supplementary information clearly not in the retrieved documents "
            "(e.g., establishment year of an institution).\n"
            "4. Do NOT use tavily_search for information already present in the retrieved documents.\n"
            "5. Count the number of papers by examining distinct document sources in the retrieved passages.\n"
            "6. If the retrieved documents don't contain enough information, say so rather than making things up."
        )
        self._agent = create_react_agent(self.llm, tools=tools, prompt=system_prompt)

    def agent_node(self, state: State) -> State:
        if self._agent is None:
            self._build_agent()

        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
        message = (
            f"=== RETRIEVED DOCUMENTS ===\n\n"
            f"{context}\n\n"
            f"=== QUESTION ===\n\n"
            f"{state.question}"
        )

        result = self._agent.invoke({"messages": [HumanMessage(content=message)]})
        messages = result.get("messages", [])
        answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                answer = msg.content
                break
        return State(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer,
        )
