"""Graph Builder"""

from langgraph.graph import END, START, StateGraph

from src.nodes.nodes import Nodes
from src.state.state import State


class GraphBuilder:
    """For the langgraph workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.nodes = Nodes(retriever=self.retriever, llm=self.llm)
        self.graph = None

    def build(self):
        # create the graph
        builder = StateGraph(State)
        # nodes
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)
        # entry point
        builder.set_entry_point("retriever")
        # edges
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)
        # compile the graph
        self.graph = builder.compile()
        return self.graph

    def run(self, question: str) -> dict:
        if self.graph is None:
            self.build()
        initial_state = State(question=question)
        return self.graph.invoke(initial_state)
