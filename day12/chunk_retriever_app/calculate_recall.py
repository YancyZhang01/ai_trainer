"""证据级 Recall 计算。"""

from .is_evidence_covered import is_evidence_covered


def calculate_recall(results, expected_evidences, k):
    """计算前 K 个结果完整覆盖了多少条 Expected Evidence。"""
    if not expected_evidences:
        return 0.0

    top_k_results = results[:k]
    hit_count = sum(
        is_evidence_covered(evidence, top_k_results)
        for evidence in expected_evidences
    )
    return hit_count / len(expected_evidences)
