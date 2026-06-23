"""Graph Builder"""

from langgraph.graph import END, START, StateGraph

from src.nodes.react_node import Nodes
from src.state.state import State


class GraphBuilder:
    """For the langgraph workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.nodes = Nodes(retriever=self.retriever, llm=self.llm)
        self.graph = None

    def build(self):
        builder = StateGraph(State)
        builder.add_node("agent", self.nodes.agent_node)
        builder.set_entry_point("agent")
        builder.add_edge("agent", END)
        self.graph = builder.compile()
        return self.graph

    def run(self, question: str) -> dict:
        if self.graph is None:
            self.build()
        initial_state = State(question=question)
        return self.graph.invoke(initial_state)
