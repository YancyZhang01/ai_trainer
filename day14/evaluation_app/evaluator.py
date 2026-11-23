"""完整 Benchmark 的运行、打印与指标汇总。"""

from .benchmark_data import BENCHMARKS
from .evaluate_case import evaluate_case


def evaluate_benchmarks(model, index, api_key):
    """运行所有案例，打印 RAG 记录及宏平均指标。"""
    evaluated_cases = []

    print("=== RAG Evaluation ===")
    for case_number, benchmark in enumerate(BENCHMARKS, start=1):
        result = evaluate_case(benchmark, model, index, api_key)
        evaluated_cases.append(result)

        print(f"\n{'=' * 70}")
        print(f"Case {case_number}: {result['question']}")
        print(f"Case Type: {result['case_type']}")
        print(f"Answerable: {result['answerable']}")
        print(f"Ground Truth: {result['ground_truth_answer']}")
        print(f"Expected Keywords: {result['expected_keywords']}")
        print(f"Retrieved IDs: {result['retrieved_ids']}")

        print("Context:")
        for context in result["contexts"]:
            print(
                f"- [{context['document_id']}] "
                f"score={context['score']:.4f} | "
                f"{context['source']}"
            )
            print(f"  {context['text']}")

        print(f"Answer: {result['answer']}")
        print("Source:")
        for source in result["sources"]:
            print(f"- [{source['document_id']}] {source['source']}")

        retrieval_text = (
            "N/A"
            if result["retrieval_hit_at_3"] is None
            else f"{result['retrieval_hit_at_3']:.0f}"
        )
        reciprocal_rank_text = (
            "N/A"
            if result["reciprocal_rank"] is None
            else f"{result['reciprocal_rank']:.4f}"
        )
        evidence_recall_text = (
            "N/A"
            if result["evidence_recall_at_3"] is None
            else f"{result['evidence_recall_at_3']:.2%}"
        )
        correctness_text = (
            "N/A"
            if result["answer_correctness"] is None
            else f"{result['answer_correctness']:.0f}"
        )

        print(f"Retrieval Hit@3: {retrieval_text}")
        print(f"Evidence Recall@3: {evidence_recall_text}")
        print(f"MRR Contribution: {reciprocal_rank_text}")
        print(f"Answer Correctness: {correctness_text}")
        print(f"Abstention Accuracy: {result['abstention_accuracy']:.0f}")

    answerable_cases = [
        result
        for result in evaluated_cases
        if result["answerable"]
    ]
    answerable_count = len(answerable_cases)
    total_count = len(evaluated_cases)

    retrieval_hit_at_3 = sum(
        result["retrieval_hit_at_3"]
        for result in answerable_cases
    ) / answerable_count
    mrr = sum(
        result["reciprocal_rank"]
        for result in answerable_cases
    ) / answerable_count
    evidence_recall_at_3 = sum(
        result["evidence_recall_at_3"]
        for result in answerable_cases
    ) / answerable_count
    answer_correctness = sum(
        result["answer_correctness"]
        for result in answerable_cases
    ) / answerable_count
    abstention_accuracy = sum(
        result["abstention_accuracy"]
        for result in evaluated_cases
    ) / total_count

    print(f"\n{'=' * 70}")
    print("Final Evaluation")
    print(f"Retrieval Hit@3: {retrieval_hit_at_3:.2%}")
    print(f"Evidence Recall@3: {evidence_recall_at_3:.2%}")
    print(f"MRR: {mrr:.4f}")
    print(f"Answer Correctness: {answer_correctness:.2%}")
    print(f"Abstention Accuracy: {abstention_accuracy:.2%}")
