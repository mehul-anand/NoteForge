"""Standalone evaluation script for NoteForge RAG pipeline.
Computes retrieval metrics (precision@k, recall@k, MRR, coverage)
and generation metrics (factual accuracy) against ground truth.

Usage: .venv/bin/python3 evaluations/issue_four/run_eval.py
"""

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections import Counter

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentHandler
from src.graph_builder.graph import GraphBuilder
from src.vector_store.store import VectorStore


def load_ground_truth(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    return data["test_questions"]


def build_pipeline():
    """Re-ingest documents from data/ and return retriever + graph."""
    handler = DocumentHandler()
    data_dir = Path("data")
    pdfs = sorted(data_dir.glob("*.pdf")) + sorted(data_dir.glob("*.txt"))

    all_docs = []
    for pdf in pdfs:
        loaded = (
            handler.pdf_loader(str(pdf))
            if pdf.suffix == ".pdf"
            else handler.text_loader(str(pdf))
        )
        all_docs.extend(loaded)

    chunks = handler.doc_splitter(all_docs)
    source_counts = Counter(doc.metadata.get("source", "unknown") for doc in chunks)

    vs = VectorStore()
    vs.create_retriever(chunks)
    retriever = vs.get_retriever()

    llm = Config.get_llm()
    doc_summaries = handler.extract_summaries(all_docs, llm)

    builder = GraphBuilder(retriever, llm)
    builder.build()

    filenames = [pdf.name for pdf in pdfs]

    return retriever, builder, doc_summaries, filenames, source_counts


def compute_precision_at_k(retrieved_chunks, relevant_sources, k=15):
    """% of top-k chunks from relevant sources."""
    if k == 0:
        return 0.0
    top_k = retrieved_chunks[:k]
    relevant = sum(
        1 for c in top_k if Path(c.metadata.get("source", "")).name in relevant_sources
    )
    return relevant / k


def compute_recall_at_k(retrieved_chunks, relevant_sources, k=15):
    """% of relevant sources that appear in top-k chunks."""
    if not relevant_sources:
        return 0.0
    top_k = retrieved_chunks[:k]
    retrieved_sources = {Path(c.metadata.get("source", "")).name for c in top_k}
    hits = len(retrieved_sources & set(relevant_sources))
    return hits / len(relevant_sources)


def compute_mrr(retrieved_chunks, relevant_sources):
    """1 / rank of first chunk from a relevant source. 0 if none found."""
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if Path(chunk.metadata.get("source", "")).name in relevant_sources:
            return 1.0 / rank
    return 0.0


def compute_factual_accuracy(answer, ground_truth, llm):
    """LLM-as-judge: score 0.0 to 1.0 for how well answer matches ground truth."""
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
        score = float(resp.content.strip())
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


def run_evaluation():
    gt_path = Path(__file__).parent / "ground_truth.json"
    questions = load_ground_truth(str(gt_path))

    print("=" * 60)
    print("NoteForge Evaluation — Building Pipeline")
    print("=" * 60)
    retriever, graph_builder, doc_summaries, filenames, _ = build_pipeline()

    llm = Config.get_llm()

    results = []
    for q in questions:
        qid = q["id"]
        query = q["query"]
        ground_truth = q["ground_truth_answer"]
        relevant_sources = q["relevant_sources"]

        print(f"\n--- Evaluating: {qid} ---")
        print(f"Query: {query}")

        # --- Retrieval Metrics ---
        retrieved = retriever.invoke(query)
        precision = compute_precision_at_k(retrieved, relevant_sources)
        recall = compute_recall_at_k(retrieved, relevant_sources)
        mrr = compute_mrr(retrieved, relevant_sources)

        retrieval_metrics = {
            "precision_at_15": round(precision, 3),
            "recall_at_15": round(recall, 3),
            "mrr": round(mrr, 3),
        }
        print(f"  Precision@15: {retrieval_metrics['precision_at_15']}")
        print(f"  Recall@15:    {retrieval_metrics['recall_at_15']}")
        print(f"  MRR:          {retrieval_metrics['mrr']}")

        # --- Generation Metrics ---
        result = graph_builder.run(
            query,
            source_files=filenames,
            doc_summaries=doc_summaries,
            chat_history=[],
        )
        answer = result.get("answer", "")

        # Truncate for display
        answer_preview = answer[:120].replace("\n", " ")
        print(f"  Answer:       {answer_preview}...")

        factual = compute_factual_accuracy(answer, ground_truth, llm)
        generation_metrics = {
            "factual_accuracy": round(factual, 3),
            "answer": answer,
        }
        print(f"  Factual:      {generation_metrics['factual_accuracy']}")

        results.append(
            {
                "id": qid,
                "query": query,
                "retrieval_metrics": retrieval_metrics,
                "generation_metrics": generation_metrics,
            }
        )

    # --- Aggregate ---
    n = len(results)
    agg = {
        "avg_precision_at_15": round(
            sum(r["retrieval_metrics"]["precision_at_15"] for r in results) / n, 3
        ),
        "avg_recall_at_15": round(
            sum(r["retrieval_metrics"]["recall_at_15"] for r in results) / n, 3
        ),
        "avg_mrr": round(sum(r["retrieval_metrics"]["mrr"] for r in results) / n, 3),
        "avg_factual_accuracy": round(
            sum(r["generation_metrics"]["factual_accuracy"] for r in results) / n, 3
        ),
    }

    output = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "config": {
            "k": 15,
            "chunk_size": 2500,
            "chunk_overlap": 200,
            "model": "gpt-4o-mini",
            "retriever": "FAISS + MMR (lambda_mult=0.7, fetch_k=30)",
        },
        "aggregate": agg,
        "results": results,
    }

    # Save
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    for key, val in agg.items():
        print(f"  {key}: {val}")
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    run_evaluation()
