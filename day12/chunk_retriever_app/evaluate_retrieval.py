"""Benchmark 评测流程。"""

from .benchmarks import BENCHMARKS
from .calculate_recall import calculate_recall
from .config import EVALUATION_KS
from .search import search


def evaluate_retrieval(model, index, chunks):
    """运行 Benchmark 并打印各项及宏平均 Evidence Recall@K。"""
    recall_totals = {k: 0.0 for k in EVALUATION_KS}

    print("\n=== Retrieval Evaluation ===")
    for benchmark_number, benchmark in enumerate(BENCHMARKS, start=1):
        results = search(
            benchmark["query"],
            model,
            index,
            chunks,
            top_k=max(EVALUATION_KS),
        )

        print(
            f"\nBenchmark {benchmark_number} "
            f"[{benchmark['question_type']}]: {benchmark['query']}"
        )
        for k in EVALUATION_KS:
            recall = calculate_recall(
                results,
                benchmark["expected_evidences"],
                k,
            )
            recall_totals[k] += recall
            print(f"Evidence Recall@{k}: {recall:.2%}")

    benchmark_count = len(BENCHMARKS)
    print("\n=== Final Evaluation ===")
    for k in EVALUATION_KS:
        average_recall = recall_totals[k] / benchmark_count
        print(f"Evidence Recall@{k}: {average_recall:.2%}")
