"""
Ablation helpers: compare retrieval with vs without cross-encoder reranking.

Run:  python -m backend.eval.ablation_study
"""

from __future__ import annotations

from typing import Any

from backend.app.retriever import AdvancedRetriever
from backend.eval.evaluation_harness import compare_reranker_ablation


class AblationStudy:
    def __init__(self):
        self.retriever = AdvancedRetriever()

    def ablate_reranker(self, query: str, k: int = 5) -> dict[str, Any]:
        with_r = self.retriever.retrieve_and_rerank(query, use_reranker=True)
        without_r = self.retriever.retrieve_and_rerank(
            query, use_reranker=False, apply_score_threshold=False
        )
        return {
            "query": query,
            "chunks_with_reranker": len(with_r),
            "chunks_without_reranker": len(without_r),
            "reranker_removed": max(0, len(without_r) - len(with_r)),
        }

    def ablate_hybrid_search(self, query: str, k: int = 5) -> dict[str, Any]:
        semantic_only = len(self.retriever._semantic_search(query, k))
        keyword_only = len(self.retriever._keyword_search(query, k))
        hybrid = self.retriever.retrieve_and_rerank(query)
        return {
            "query": query,
            "faiss_only": semantic_only,
            "bm25_only": keyword_only,
            "hybrid_after_gate": len(hybrid),
        }


ABLATION_QUERIES = [
    "What is transformer attention?",
    "Explain reinforcement learning updates",
    "How does contrastive learning work?",
]


if __name__ == "__main__":
    study = AblationStudy()
    print("\n" + "=" * 70)
    print("ABLATION 1: chunk counts with vs without reranker (same RRF pool)")
    print("=" * 70 + "\n")
    for q in ABLATION_QUERIES:
        r = study.ablate_reranker(q)
        print(f"Query: {r['query']}")
        print(f"  Chunks w/ rerank+threshold: {r['chunks_with_reranker']}")
        print(f"  Chunks w/o reranker:        {r['chunks_without_reranker']}")
        print()

    print("\n" + "=" * 70)
    print("ABLATION 2: eval harness pass rate with vs without reranker")
    print("=" * 70 + "\n")
    summary = compare_reranker_ablation()
    print(f"Pass rate w/ reranker:    {summary['with_reranker_pass_rate']:.1%}")
    print(f"Pass rate w/o reranker:   {summary['without_reranker_pass_rate']:.1%}")
    print(f"Delta (with - without):   {summary['delta']:+.1%}")

    print("\n" + "=" * 70)
    print("ABLATION 3: hybrid vs single-channel recall@k (index counts)")
    print("=" * 70 + "\n")
    for q in ABLATION_QUERIES:
        r = study.ablate_hybrid_search(q)
        print(f"Query: {r['query']}")
        print(
            f"  FAISS@k: {r['faiss_only']}  BM25@k: {r['bm25_only']}  Hybrid final: {r['hybrid_after_gate']}"
        )
        print()
