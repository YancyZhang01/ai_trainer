"""Reciprocal Rank Fusion。"""

from .config import HYBRID_TOP_K, RRF_K


def reciprocal_rank_fusion(
    bm25_results,
    dense_results,
    top_k=HYBRID_TOP_K,
    rrf_k=RRF_K,
):
    """用排名而非原始分数融合 BM25 和 Dense 结果。"""
    fused_documents = {}

    for retriever_name, results in (
        ("bm25", bm25_results),
        ("dense", dense_results),
    ):
        for result in results:
            document_id = result["document_id"]
            if document_id not in fused_documents:
                fused_documents[document_id] = {
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "dense_rank": None,
                    "document": {
                        key: value
                        for key, value in result.items()
                        if key not in {"rank", "score"}
                    },
                }

            fused_documents[document_id]["rrf_score"] += (
                1 / (rrf_k + result["rank"])
            )
            fused_documents[document_id][f"{retriever_name}_rank"] = result[
                "rank"
            ]

    ranked_documents = sorted(
        fused_documents.values(),
        key=lambda item: (-item["rrf_score"], item["document"]["document_id"]),
    )

    return [
        {
            "rank": rank,
            "rrf_score": item["rrf_score"],
            "bm25_rank": item["bm25_rank"],
            "dense_rank": item["dense_rank"],
            **item["document"],
        }
        for rank, item in enumerate(ranked_documents[:top_k], start=1)
    ]
