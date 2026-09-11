"""三类 Retriever 的对比评测流程。"""

from .benchmark_data import BENCHMARKS
from .config import BM25_TOP_N, DENSE_TOP_N, HYBRID_TOP_K
from .retrieval_metrics import (
    calculate_hit_at_k,
    calculate_reciprocal_rank,
)


def run_benchmark(retriever):
    """打印逐题 Top-1，并汇总 Hit@1、Hit@3 和 MRR。"""
    retriever_groups = {
        "BM25": "bm25_results",
        "Dense": "dense_results",
        "Hybrid": "hybrid_results",
    }
    metric_totals = {
        name: {"hit_at_1": 0.0, "hit_at_3": 0.0, "rr": 0.0}
        for name in retriever_groups
    }

    print("=== Hybrid Retriever Benchmark ===")
    print(f"BM25 Top-N: {BM25_TOP_N}")
    print(f"Dense Top-N: {DENSE_TOP_N}")
    print(f"Hybrid Top-K: {HYBRID_TOP_K}")

    for number, benchmark in enumerate(BENCHMARKS, start=1):
        all_results = retriever.retrieve(benchmark["query"])
        relevant_ids = benchmark["relevant_document_ids"]

        print(f"\n--- Case {number}: {benchmark['query_type']} ---")
        print(f"Query: {benchmark['query']}")
        print(f"Relevant Documents: {', '.join(relevant_ids)}")
        print(f"Expected Evidence: {benchmark['expected_evidence']}")

        for retriever_name, result_key in retriever_groups.items():
            results = all_results[result_key]
            top_1 = results[0] if results else None
            top_1_text = (
                f"{top_1['document_id']} - {top_1['title']}"
                if top_1
                else "无结果"
            )
            hit_at_1 = calculate_hit_at_k(results, relevant_ids, 1)
            hit_at_3 = calculate_hit_at_k(results, relevant_ids, 3)
            reciprocal_rank = calculate_reciprocal_rank(results, relevant_ids)

            metric_totals[retriever_name]["hit_at_1"] += hit_at_1
            metric_totals[retriever_name]["hit_at_3"] += hit_at_3
            metric_totals[retriever_name]["rr"] += reciprocal_rank

            print(
                f"{retriever_name:<6} Top-1: {top_1_text} | "
                f"Hit@1: {hit_at_1:.0f} | "
                f"Hit@3: {hit_at_3:.0f} | "
                f"RR: {reciprocal_rank:.4f}"
            )

    case_count = len(BENCHMARKS)
    print("\n=== Final Comparison ===")
    print(f"{'Retriever':<10} {'Hit@1':>8} {'Hit@3':>8} {'MRR':>8}")
    for retriever_name, totals in metric_totals.items():
        print(
            f"{retriever_name:<10} "
            f"{totals['hit_at_1'] / case_count:>8.2%} "
            f"{totals['hit_at_3'] / case_count:>8.2%} "
            f"{totals['rr'] / case_count:>8.4f}"
        )
