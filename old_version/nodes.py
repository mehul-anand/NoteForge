from src.state.state import State


class Nodes:
    "Contains node functions"

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self, state: State) -> State:
        docs = self.retriever.invoke(state.question)
        return State(question=state.question, retrieved_docs=docs)

    def generate_answer(self, state: State) -> State:
        # Combine retrieval with the context
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
        # Create a prompt
        prompt = f"""Answer the question based on the context.
        Context:
                {context}
        Question:
                {state.question}
        """
        # generate a response
        response = self.llm.invoke(prompt)
        return State(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content,
        )
