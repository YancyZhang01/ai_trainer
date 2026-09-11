"""Document 级检索指标。"""


def calculate_hit_at_k(results, relevant_document_ids, k):
    """只要前 K 名包含任意相关 Document，Hit@K 就等于 1。"""
    retrieved_ids = {
        result["document_id"]
        for result in results[:k]
    }
    return float(bool(retrieved_ids.intersection(relevant_document_ids)))


def calculate_reciprocal_rank(results, relevant_document_ids):
    """返回第一个相关 Document 排名的倒数，未命中返回 0。"""
    for rank, result in enumerate(results, start=1):
        if result["document_id"] in relevant_document_ids:
            return 1.0 / rank
    return 0.0
