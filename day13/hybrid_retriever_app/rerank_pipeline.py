"""粗召回和精排的两阶段检索流程。"""

from .config import RERANK_CANDIDATE_COUNT, RERANK_TOP_K
from .hybrid_retriever import HybridRetriever
from .reranker import Reranker


class RerankPipeline:
    """先用 Hybrid Retriever 粗召回，再用 Cross-Encoder 精排。"""

    def __init__(self, documents):
        self.hybrid_retriever = HybridRetriever(documents)
        self.reranker = Reranker()

    def retrieve_and_rerank(self, query):
        """对同一 Query 生成至少 5 个候选，并输出两阶段结果。"""
        retrieval_results = self.hybrid_retriever.retrieve(
            query,
            bm25_top_n=RERANK_CANDIDATE_COUNT,
            dense_top_n=RERANK_CANDIDATE_COUNT,
            hybrid_top_k=RERANK_CANDIDATE_COUNT,
        )
        coarse_candidates = retrieval_results["hybrid_results"]
        reranked_results = self.reranker.rerank(
            query,
            coarse_candidates,
            top_k=RERANK_TOP_K,
        )

        return {
            "query": query,
            "candidate_count": len(coarse_candidates),
            "coarse_results": coarse_candidates,
            "reranked_results": reranked_results,
        }
