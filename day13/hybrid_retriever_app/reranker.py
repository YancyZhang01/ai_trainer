"""Cross-Encoder 候选重排。"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import RERANK_TOP_K, RERANKER_MAX_LENGTH, RERANKER_MODEL_NAME


class Reranker:
    """使用 Query-Document 交叉编码分数重新排列候选文档。"""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL_NAME
        )
        self.model.eval()

    def rerank(self, query, candidates, top_k=RERANK_TOP_K):
        """为同一 Query 的候选逐一评分，并返回重排后的 Top-K。"""
        if not candidates:
            return []

        pairs = [[query, candidate["text"]] for candidate in candidates]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=RERANKER_MAX_LENGTH,
        )

        with torch.no_grad():
            scores = self.model(**inputs, return_dict=True).logits.view(-1)

        scored_candidates = []
        for candidate, score in zip(candidates, scores.tolist()):
            scored_candidates.append(
                {
                    **candidate,
                    "coarse_rank": candidate["rank"],
                    "coarse_score": candidate["rrf_score"],
                    "reranker_score": float(score),
                }
            )

        scored_candidates.sort(
            key=lambda item: (
                -item["reranker_score"],
                item["coarse_rank"],
            )
        )

        return [
            {**candidate, "rank": rank}
            for rank, candidate in enumerate(
                scored_candidates[:top_k],
                start=1,
            )
        ]
