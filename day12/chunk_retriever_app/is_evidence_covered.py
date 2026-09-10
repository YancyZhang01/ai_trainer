"""判断检索结果是否覆盖证据区间。"""


def is_evidence_covered(evidence, results):
    """判断是否有同一文档的结果完整覆盖 Expected Evidence。"""
    expected_span = evidence["evidence_span"]

    return any(
        result["document_id"] == evidence["document_id"]
        and result["start_char"] <= expected_span["start_char"]
        and result["end_char"] >= expected_span["end_char"]
        for result in results
    )
