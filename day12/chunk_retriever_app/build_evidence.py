"""创建稳定的 Benchmark 证据标注。"""

from .documents import DOCUMENTS_BY_ID


def build_evidence(document_id, expected_evidence):
    """根据证据原文计算它在原始文档中的字符区间。"""
    document_text = DOCUMENTS_BY_ID[document_id]["text"]
    start_char = document_text.find(expected_evidence)
    if start_char == -1:
        raise ValueError(f"证据不在文档 {document_id} 中：{expected_evidence}")

    return {
        "document_id": document_id,
        "evidence_span": {
            "start_char": start_char,
            "end_char": start_char + len(expected_evidence),
        },
        "expected_evidence": expected_evidence,
    }
