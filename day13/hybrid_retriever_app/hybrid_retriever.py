"""两路检索与融合编排。"""

from .bm25_retriever import BM25Retriever
from .config import BM25_TOP_N, DENSE_TOP_N, HYBRID_TOP_K
from .dense_retriever import DenseRetriever
from .reciprocal_rank_fusion import reciprocal_rank_fusion


class HybridRetriever:
    """同时运行 BM25 与 Dense Retriever，并通过 RRF 融合。"""

    def __init__(self, documents):
        self.bm25_retriever = BM25Retriever(documents)
        self.dense_retriever = DenseRetriever(documents)

    def retrieve(
        self,
        query,
        bm25_top_n=BM25_TOP_N,
        dense_top_n=DENSE_TOP_N,
        hybrid_top_k=HYBRID_TOP_K,
    ):
        bm25_results = self.bm25_retriever.search(query, bm25_top_n)
        dense_results = self.dense_retriever.search(query, dense_top_n)
        hybrid_results = reciprocal_rank_fusion(
            bm25_results,
            dense_results,
            top_k=hybrid_top_k,
        )

        return {
            "bm25_results": bm25_results,
            "dense_results": dense_results,
            "hybrid_results": hybrid_results,
        }
