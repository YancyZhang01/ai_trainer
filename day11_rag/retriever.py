"""食品文档的向量索引和检索逻辑。"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


TOP_K = 2
MIN_RELEVANCE_SCORE = 0.5
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def build_index(documents):
    """加载 Embedding 模型，并为食品文档建立 FAISS 索引。"""
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    texts = [document["text"] for document in documents]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return model, index


def retrieve(query, model, index, documents, top_k=TOP_K):
    """检索与 query 最相关的 Top-K 文档。"""
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    result_count = min(top_k, len(documents))
    scores, indices = index.search(query_embedding, result_count)

    results = []
    for rank, (document_index, score) in enumerate(
        zip(indices[0], scores[0]),
        start=1,
    ):
        results.append(
            {
                "rank": rank,
                "score": float(score),
                **documents[document_index],
            }
        )
    return results


def has_relevant_evidence(results):
    """使用最高相似度判断知识库中是否存在可用证据。"""
    return bool(results) and results[0]["score"] >= MIN_RELEVANCE_SCORE


def print_contexts(results, top_k=TOP_K):
    """打印本次检索得到的上下文及其来源。"""
    print(f"\n=== Retrieved Top {top_k} Context ===")
    for result in results:
        print(f"\n[{result['rank']}] {result['document_id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Context: {result['text']}")
