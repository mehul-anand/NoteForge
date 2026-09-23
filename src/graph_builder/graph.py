"""Graph Builder"""

from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

from src.nodes.react_node import Nodes
from src.nodes.synthesis import ReviewSynthesizer
from src.state.state import State


class GraphBuilder:
    """Builds and runs the LangGraph RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.nodes = Nodes(retriever=self.retriever, llm=self.llm)
        self.synthesizer = ReviewSynthesizer(llm=self.llm)
        self.graph = None

    def build(self):
        """
        Architecture:
          route_intent → expand_query → rewrite_queries → retrieve
              → agent (qa) | synthesize (compare/review/gaps) → END

        'route_intent' classifies each question into qa / compare / review /
        gaps. Every path shares the decomposition + retrieval spine; only the
        terminal node differs — the ReAct agent for qa, a structured
        synthesis node for the review intents.
        """
        builder = StateGraph(State)
        builder.add_node("route_intent", self.synthesizer.route_intent)
        builder.add_node("expand_query", self.nodes.expand_query)
        builder.add_node("rewrite_queries", self.nodes.rewrite_queries)
        builder.add_node("retrieve", self.nodes.retrieve_docs)
        builder.add_node("agent", self.nodes.agent_node)
        builder.add_node("synthesize", self.synthesizer.synthesize)
        builder.set_entry_point("route_intent")
        builder.add_edge("route_intent", "expand_query")
        builder.add_edge("expand_query", "rewrite_queries")
        builder.add_edge("rewrite_queries", "retrieve")
        builder.add_conditional_edges(
            "retrieve",
            lambda state: "agent" if state.task_type == "qa" else "synthesize",
            {"agent": "agent", "synthesize": "synthesize"},
        )
        builder.add_edge("agent", END)
        builder.add_edge("synthesize", END)

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
