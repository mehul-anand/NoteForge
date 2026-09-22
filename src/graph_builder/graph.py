"""Graph Builder"""

from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

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
        Architecture: START → expand_query → rewrite_queries → retrieve → agent → END

        'expand_query' decomposes complex questions into focused sub-queries.
        'rewrite_queries' rewrites each sub-query to improve embedding similarity.
        'retrieve' iterates over rewritten queries for comprehensive coverage.
        'agent' receives pre-fetched docs from state and only has Tavily
        for supplementary lookups.
        """
        builder = StateGraph(State)
        builder.add_node("expand_query", self.nodes.expand_query)
        builder.add_node("rewrite_queries", self.nodes.rewrite_queries)
        builder.add_node("retrieve", self.nodes.retrieve_docs)
        builder.add_node("agent", self.nodes.agent_node)
        builder.set_entry_point("expand_query")
        builder.add_edge("expand_query", "rewrite_queries")
        builder.add_edge("rewrite_queries", "retrieve")
        builder.add_edge("retrieve", "agent")
        builder.add_edge("agent", END)

        self.graph = builder.compile()
        return self.graph

    def run(
        self,
        question: str,
        source_files: List[str] = [],
        doc_summaries: Dict[str, str] = {},
        paper_metadata: Dict[str, Any] = {},
        chat_history: List[Dict[str, str]] = [],
    ) -> dict:
        if self.graph is None:
            self.build()
        initial_state = State(
            question=question,
            source_files=source_files,
            doc_summaries=doc_summaries,
            paper_metadata=paper_metadata,
            chat_history=chat_history,
        )
        return self.graph.invoke(initial_state)
