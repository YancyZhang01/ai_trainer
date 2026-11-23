"""Document 级检索指标。"""


def calculate_hit_at_k(retrieved_ids, relevant_ids, k):
    """前 K 个结果包含任意相关文档时返回 1。"""
    return float(bool(set(retrieved_ids[:k]).intersection(relevant_ids)))


def calculate_reciprocal_rank(retrieved_ids, relevant_ids):
    """返回第一个相关文档排名的倒数。"""
    for rank, document_id in enumerate(retrieved_ids, start=1):
        if document_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def calculate_evidence_recall_at_k(retrieved_ids, relevant_ids, k):
    """返回前 K 个结果覆盖的相关文档比例，适合评估多证据问题。"""
    if not relevant_ids:
        return None

    retrieved_at_k = set(retrieved_ids[:k])
    return len(retrieved_at_k.intersection(relevant_ids)) / len(set(relevant_ids))


def calculate_context_precision_at_k(retrieved_ids, relevant_ids, k):
    """前 K 条结果中相关文档所占比例。"""
    retrieved_at_k = retrieved_ids[:k]
    if not retrieved_at_k:
        return 0.0
    relevant_set = set(relevant_ids)
    relevant_count = sum(
        document_id in relevant_set for document_id in retrieved_at_k
    )
    return relevant_count / len(retrieved_at_k)


def calculate_context_recall_at_k(retrieved_ids, relevant_ids, k):
    """全部人工标注相关文档中，被前 K 条结果找回的比例。"""
    if not relevant_ids:
        return None
    retrieved_at_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_at_k.intersection(relevant_set)) / len(relevant_set)
