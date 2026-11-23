"""单条 Benchmark 评测。"""

from .answer_metrics import (
    calculate_abstention_accuracy,
    calculate_answer_correctness,
)
from .config import RETRIEVAL_TOP_K
from .rag_runner import run_rag
from .retrieval_metrics import (
    calculate_evidence_recall_at_k,
    calculate_hit_at_k,
    calculate_reciprocal_rank,
)


def evaluate_case(benchmark, model, index, api_key):
    """运行一次 RAG，并为其补充全部 Evaluation 指标。"""
    rag_record = run_rag(
        benchmark["question"],
        model,
        index,
        api_key,
    )

    if benchmark["answerable"]:
        retrieval_hit = calculate_hit_at_k(
            rag_record["retrieved_ids"],
            benchmark["relevant_document_ids"],
            RETRIEVAL_TOP_K,
        )
        reciprocal_rank = calculate_reciprocal_rank(
            rag_record["retrieved_ids"],
            benchmark["relevant_document_ids"],
        )
        evidence_recall = calculate_evidence_recall_at_k(
            rag_record["retrieved_ids"],
            benchmark["relevant_document_ids"],
            RETRIEVAL_TOP_K,
        )
    else:
        retrieval_hit = None
        reciprocal_rank = None
        evidence_recall = None

    return {
        **benchmark,
        **rag_record,
        "retrieval_hit_at_3": retrieval_hit,
        "evidence_recall_at_3": evidence_recall,
        "reciprocal_rank": reciprocal_rank,
        "answer_correctness": calculate_answer_correctness(
            rag_record["answer"],
            benchmark,
        ),
        "abstention_accuracy": (
            calculate_abstention_accuracy(
                rag_record["abstained"],
                benchmark["answerable"],
            )
            if rag_record["generation_succeeded"]
            else 0.0
        ),
    }
