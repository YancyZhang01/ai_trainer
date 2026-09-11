"""统一管理检索参数。"""

BM25_TOP_N = 5
DENSE_TOP_N = 5
HYBRID_TOP_K = 3
RRF_K = 60
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# Reranker 先接收较多粗召回候选，再输出更精确的 Top-K。
RERANK_CANDIDATE_COUNT = 5
RERANK_TOP_K = 3
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
RERANKER_MAX_LENGTH = 512
