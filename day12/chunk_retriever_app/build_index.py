"""Embedding 与 FAISS 索引构建。"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import MODEL_NAME


def build_index(chunks):
    """生成 Chunk 向量并建立 FAISS 内积索引。"""
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        chunk_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return model, index
