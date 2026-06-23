"""RAG Workflow with a ReAct Agent"""

from typing import List, Optional

from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent

from src.state.state import State


class Nodes:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # lazy initialisation
        # _agent is protected (use of _)

    def retrieve_docs(self, state: State) -> State:
        docs = self.retriever.invoke(state.question)
        return State(question=state.question, retrieved_docs=docs)

    # build tools
    def _build_tools(self) -> List[Tool]:
        def retriever_tool_func(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found"
            else:
                merged = []
                for i, d in enumerate(docs[:8], start=1):
                    meta = d.metadata if hasattr(d, "metadata") else {}
                    title = meta.get("title") or meta.get("source") or f"doc_{i}"
                    merged.append(f"[{i}] {title} \n {d.page_content}")
                return "\n\n".join(merged)

        retrieval_tool = Tool(
            name="retriever",
            description="Fetch passages from a indexed vectorstore",
            func=retriever_tool_func,
        )
        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )
        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general knowledge",
            func=wiki.run,
        )
        return [retrieval_tool, wikipedia_tool]

    # build agent

    def _build_agent(self):
        "ReAct agent with tools"
        tools = []
        system_prompt = "You are a RAG agent and your job is to help the user with answers based on the docs [], refer to 'retriever' to get something out of the documents and 'wikipedia' for general questions that are not there in the docs, do mention that what the user asked does not exist in the documents"
        self._agent = create_react_agent(self.llm, tools=tools, prompt=system_prompt)
