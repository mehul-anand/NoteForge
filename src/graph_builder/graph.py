"""Graph Builder"""

from typing import Dict, List

from langgraph.graph import END, START, StateGraph

from src.nodes.react_node import Nodes
from src.state.state import State


class GraphBuilder:
    """Builds and runs the LangGraph RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.nodes = Nodes(retriever=self.retriever, llm=self.llm)
        self.graph = None

    def build(self):
        """
        Architecture: START → expand_query → retrieve → agent → END

        'expand_query' decomposes complex questions into focused sub-queries.
        'retrieve' iterates over each sub-query for comprehensive coverage.
        'agent' receives pre-fetched docs from state and only has Tavily
        for supplementary lookups.
        """
        builder = StateGraph(State)
        builder.add_node("expand_query", self.nodes.expand_query)
        builder.add_node("retrieve", self.nodes.retrieve_docs)
        builder.add_node("agent", self.nodes.agent_node)
        builder.set_entry_point("expand_query")
        builder.add_edge("expand_query", "retrieve")
        builder.add_edge("retrieve", "agent")
        builder.add_edge("agent", END)

        self.graph = builder.compile()
        return self.graph

    def run(
        self,
        question: str,
        source_files: List[str] = [],
        chat_history: List[Dict[str, str]] = [],
    ) -> dict:
        if self.graph is None:
            self.build()
        initial_state = State(
            question=question, source_files=source_files, chat_history=chat_history
        )
        return self.graph.invoke(initial_state)
