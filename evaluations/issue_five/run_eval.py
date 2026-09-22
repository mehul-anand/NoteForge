"""Comparison eval: Baseline (pre-Claude) vs Current (optimised) RAG pipeline.
Runs both on the same 12 questions from ground_truth.json and reports
improvement percentage for each metric.

Usage: .venv/bin/python3 evaluations/issue_five/run_eval.py
"""

import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentHandler
from src.graph_builder.graph import GraphBuilder
from src.state.state import State
from src.vector_store.store import VectorStore


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def load_ground_truth(path: str) -> list:
    with open(path) as f:
        return json.load(f)["test_questions"]


def load_and_chunk_documents(data_dir: str) -> tuple:
    """Load all PDFs/TXTs from data_dir, split into chunks.
    Returns (chunks, filenames, all_docs)."""
    handler = DocumentHandler()
    data_path = Path(data_dir)
    pdfs = sorted(data_path.glob("*.pdf")) + sorted(data_path.glob("*.txt"))

    all_docs: List[Document] = []
    for pdf in pdfs:
        if pdf.suffix == ".pdf":
            loaded = handler.pdf_loader(str(pdf))
        else:
            loaded = handler.text_loader(str(pdf))
        all_docs.extend(loaded)

    chunks = handler.doc_splitter(all_docs)
    filenames = [pdf.name for pdf in pdfs]
    return chunks, filenames, all_docs


def compute_precision_at_k(retrieved_chunks, relevant_sources, k=8):
    """% of top-k chunks from relevant sources."""
    if k == 0:
        return 0.0
    top_k = retrieved_chunks[:k]
    relevant = sum(
        1 for c in top_k if Path(c.metadata.get("source", "")).name in relevant_sources
    )
    return relevant / k


def compute_recall_at_k(retrieved_chunks, relevant_sources, k=8):
    """% of relevant sources that appear in top-k chunks."""
    if not relevant_sources:
        return 0.0
    top_k = retrieved_chunks[:k]
    retrieved_sources = {Path(c.metadata.get("source", "")).name for c in top_k}
    hits = len(retrieved_sources & set(relevant_sources))
    return hits / len(relevant_sources)


def compute_mrr(retrieved_chunks, relevant_sources):
    """1 / rank of first chunk from a relevant source. 0 if none."""
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if Path(chunk.metadata.get("source", "")).name in relevant_sources:
            return 1.0 / rank
    return 0.0


def compute_factual_accuracy(answer, ground_truth, llm):
    """LLM-as-judge: score 0.0 to 1.0."""
    prompt = (
        "You are an evaluation judge. Rate whether the ANSWER below matches "
        "the GROUND TRUTH. Score from 0.0 to 1.0 where:\n"
        "- 1.0 = completely correct, all key facts match\n"
        "- 0.5 = partially correct, some facts match but some are missing/wrong\n"
        "- 0.0 = completely wrong or no relevant information\n\n"
        f"GROUND TRUTH: {ground_truth}\n\n"
        f"ANSWER: {answer}\n\n"
        "Return ONLY a single number between 0.0 and 1.0 — no other text."
    )
    try:
        resp = llm.invoke(prompt)
        return max(0.0, min(1.0, float(resp.content.strip())))
    except Exception:
        return 0.0


def print_results(title, agg, results):
    print(f"\n  {'=' * 50}")
    print(f"  {title}")
    print(f"  {'=' * 50}")
    print(f"  avg_precision@{K}:  {agg['avg_precision']}")
    print(f"  avg_recall@{K}:     {agg['avg_recall']}")
    print(f"  avg_mrr:            {agg['avg_mrr']}")
    print(f"  avg_factual_acc:    {agg['avg_factual']}")
    for r in results:
        factual = r["generation"]["factual_accuracy"]
        flag = "  <<<" if factual < 1.0 else ""
        print(
            f"    {r['id']:25s}  prec={r['retrieval']['precision']:.3f}  "
            f"recall={r['retrieval']['recall']:.3f}  "
            f"mrr={r['retrieval']['mrr']:.3f}  "
            f"fact={factual:.1f}{flag}"
        )


# ---------------------------------------------------------------------------
# Baseline pipeline
# ---------------------------------------------------------------------------

BASELINE_K = 8


def build_baseline_pipeline(chunks, llm):
    """Build a simple retrieve → generate graph (pre-Claude style).
    No MMR, no agent, no decomposition, no summaries, k=8.
    """
    # plain similarity retriever, k=8
    embeddings = OpenAIEmbeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": BASELINE_K},
    )

    # nodes
    def retrieve_docs(state: State) -> State:
        docs = retriever.invoke(state.question)
        return State(question=state.question, retrieved_docs=docs)

    def generate_answer(state: State) -> State:
        context = "\n\n".join(
            f"[{i}] Source: {Path(d.metadata.get('source', 'unknown')).name}\n{d.page_content}"
            for i, d in enumerate(state.retrieved_docs, start=1)
        )
        prompt = (
            "Answer the question based on the context below. If the context "
            "lacks enough information, say so — do not invent facts.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {state.question}"
        )
        try:
            resp = llm.invoke(prompt)
            answer = resp.content.strip()
        except Exception:
            answer = "Error generating answer."
        return State(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer,
        )

    # graph
    builder = StateGraph(State)
    builder.add_node("retriever", retrieve_docs)
    builder.add_node("responder", generate_answer)
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "responder")
    builder.add_edge("responder", END)
    graph = builder.compile()

    return graph


# ---------------------------------------------------------------------------
# Current pipeline
# ---------------------------------------------------------------------------

CURRENT_K = 15


def build_current_pipeline(chunks, filenames, all_docs, llm):
    """Build the full optimised pipeline (current src/ code).
    MMR retriever, ReAct agent, query decomposition, doc summaries, etc.
    """
    # current vectorstore with MMR
    vs = VectorStore()
    vs.create_retriever(chunks)
    retriever = vs.get_retriever()

    # structured paper metadata
    handler = DocumentHandler()
    paper_metadata = handler.extract_metadata(all_docs, llm)

    # full graph
    builder = GraphBuilder(retriever, llm)
    builder.build()

    return builder, paper_metadata


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

K = 8  # shared k for fair comparison


def run_evaluation():
    gt_path = Path(__file__).parent / "ground_truth.json"
    questions = load_ground_truth(str(gt_path))

    print("=" * 60)
    print("Loading and chunking documents …")
    print("=" * 60)
    chunks, filenames, all_docs = load_and_chunk_documents("data")
    print(f"  {len(chunks)} chunks from {len(filenames)} files")

    llm = Config.get_llm()

    # ---- Build both pipelines ----
    print("\nBuilding baseline pipeline (pre-Claude: similarity k=8, no agent) …")
    baseline_graph = build_baseline_pipeline(chunks, llm)

    print("Building current pipeline (MMR k=15, agent, metadata, decomposition) …")
    current_graph, paper_metadata = build_current_pipeline(
        chunks, filenames, all_docs, llm
    )

    # ---- Run both ----
    baseline_results = []
    current_results = []

    for q in questions:
        qid = q["id"]
        query = q["query"]
        ground_truth = q["ground_truth_answer"]
        relevant_sources = q["relevant_sources"]

        # --- Baseline ---
        b_result = baseline_graph.invoke(State(question=query))
        b_docs = b_result.get("retrieved_docs", [])
        b_answer = b_result.get("answer", "")

        b_precision = compute_precision_at_k(b_docs, relevant_sources, k=K)
        b_recall = compute_recall_at_k(b_docs, relevant_sources, k=K)
        b_mrr = compute_mrr(b_docs, relevant_sources)
        b_factual = compute_factual_accuracy(b_answer, ground_truth, llm)

        baseline_results.append(
            {
                "id": qid,
                "retrieval": {
                    "precision": b_precision,
                    "recall": b_recall,
                    "mrr": b_mrr,
                },
                "generation": {"factual_accuracy": b_factual, "answer": b_answer},
            }
        )

        # --- Current ---
        c_result = current_graph.run(
            query,
            source_files=filenames,
            paper_metadata=paper_metadata,
            chat_history=[],
        )
        c_answer = c_result.get("answer", "")

        # For current, we get the retrieval metrics by calling the retriever directly
        # (the graph's internal state has retrieved_docs but we can't access it
        #  directly from graph.run() — so we re-invoke the retriever)
        vs = VectorStore()
        vs.create_retriever(chunks)
        c_docs = vs.get_retriever().invoke(query)

        c_precision = compute_precision_at_k(c_docs, relevant_sources, k=K)
        c_recall = compute_recall_at_k(c_docs, relevant_sources, k=K)
        c_mrr = compute_mrr(c_docs, relevant_sources)
        c_factual = compute_factual_accuracy(c_answer, ground_truth, llm)

        current_results.append(
            {
                "id": qid,
                "retrieval": {
                    "precision": c_precision,
                    "recall": c_recall,
                    "mrr": c_mrr,
                },
                "generation": {"factual_accuracy": c_factual, "answer": c_answer},
            }
        )

    # ---- Aggregate ----
    def aggregate(results):
        n = len(results)
        return {
            "avg_precision": round(
                sum(r["retrieval"]["precision"] for r in results) / n, 3
            ),
            "avg_recall": round(sum(r["retrieval"]["recall"] for r in results) / n, 3),
            "avg_mrr": round(sum(r["retrieval"]["mrr"] for r in results) / n, 3),
            "avg_factual": round(
                sum(r["generation"]["factual_accuracy"] for r in results) / n, 3
            ),
        }

    baseline_agg = aggregate(baseline_results)
    current_agg = aggregate(current_results)

    # ---- Print ----
    print_results(
        "BASELINE (pre-Claude: similarity k=8, no agent, no summaries)",
        baseline_agg,
        baseline_results,
    )
    print_results(
        "CURRENT (MMR k=15, agent, metadata, decomposition, rewriting)",
        current_agg,
        current_results,
    )

    # ---- Comparison table ----
    print(f"\n\n  {'=' * 55}")
    print(f"  COMPARISON (k={K} for both pipelines)")
    print(f"  {'=' * 55}")
    print(f"  {'Metric':<25s}  {'Baseline':<10s}  {'Current':<10s}  {'Change':<10s}")
    print(f"  {'-' * 25}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    for key, label in [
        ("avg_precision", f"Precision@{K}"),
        ("avg_recall", f"Recall@{K}"),
        ("avg_mrr", "MRR"),
        ("avg_factual", "Factual Accuracy"),
    ]:
        b_val = baseline_agg[key]
        c_val = current_agg[key]
        diff = c_val - b_val
        pct = (diff / b_val * 100) if b_val != 0 else float("inf")
        print(f"  {label:<25s}  {b_val:<10.3f}  {c_val:<10.3f}  {pct:>+6.1f}%")

    # ---- Save ----
    output = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "config": {
            "comparison_k": K,
            "baseline": {"retriever": "similarity", "k": BASELINE_K},
            "current": {
                "retriever": "MMR",
                "k": CURRENT_K,
                "fetch_k": 30,
                "lambda_mult": 0.7,
            },
        },
        "baseline": {"aggregate": baseline_agg, "results": baseline_results},
        "current": {"aggregate": current_agg, "results": current_results},
    }

    out_path = Path(__file__).parent / "comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Full results saved to {out_path}")


if __name__ == "__main__":
    run_evaluation()
