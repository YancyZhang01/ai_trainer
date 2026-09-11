"""SentenceTransformer + FAISS 向量检索。"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import DENSE_TOP_N, MODEL_NAME


class DenseRetriever:
    """把文档编码为归一化向量，并使用 FAISS 内积检索。"""

    def __init__(self, documents):
        self.documents = documents
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        document_embeddings = self.model.encode(
            [document["text"] for document in documents],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        self.index = faiss.IndexFlatIP(document_embeddings.shape[1])
        self.index.add(document_embeddings)

    def search(self, query, top_n=DENSE_TOP_N):
        """返回与 Query 语义最接近的前 Top-N 个文档。"""
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        result_count = min(top_n, len(self.documents))
        scores, positions = self.index.search(query_embedding, result_count)

        return [
            {
                "rank": rank,
                "score": float(score),
                **self.documents[position],
            }
            for rank, (position, score) in enumerate(
                zip(positions[0], scores[0]),
                start=1,
            )
        ]
