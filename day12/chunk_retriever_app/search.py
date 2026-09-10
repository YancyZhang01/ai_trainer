"""语义检索。"""

import numpy as np

from .config import TOP_K


def search(query, model, index, chunks, top_k=TOP_K):
    """检索与用户问题语义最接近的 Top-K 个 Chunk。"""
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    result_count = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, result_count)

    results = []
    for rank, (chunk_position, score) in enumerate(
        zip(indices[0], scores[0]),
        start=1,
    ):
        results.append(
            {
                "rank": rank,
                "score": float(score),
                **chunks[chunk_position],
            }
        )
    return results
