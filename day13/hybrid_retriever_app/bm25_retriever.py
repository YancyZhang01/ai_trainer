"""BM25 稀疏关键词检索。"""

import math
from collections import Counter

from .config import BM25_TOP_N
from .tokenizer import tokenize


class BM25Retriever:
    """不依赖外部 BM25 包的最小实现。"""

    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.document_tokens = [tokenize(doc["text"]) for doc in documents]
        self.document_lengths = [len(tokens) for tokens in self.document_tokens]
        self.average_length = sum(self.document_lengths) / len(documents)

        document_frequency = Counter()
        for tokens in self.document_tokens:
            document_frequency.update(set(tokens))

        document_count = len(documents)
        self.idf = {
            token: math.log(
                1 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for token, frequency in document_frequency.items()
        }

    def search(self, query, top_n=BM25_TOP_N):
        """返回 BM25 分数大于零的前 Top-N 个文档。"""
        query_tokens = tokenize(query)
        scored_documents = []

        for document, tokens, document_length in zip(
            self.documents,
            self.document_tokens,
            self.document_lengths,
        ):
            token_frequency = Counter(tokens)
            score = 0.0

            for token in query_tokens:
                frequency = token_frequency.get(token, 0)
                if frequency == 0:
                    continue

                numerator = frequency * (self.k1 + 1)
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * document_length / self.average_length
                )
                score += self.idf.get(token, 0.0) * numerator / denominator

            if score > 0:
                scored_documents.append((score, document))

        scored_documents.sort(
            key=lambda item: (-item[0], item[1]["document_id"])
        )
        return [
            {
                "rank": rank,
                "score": score,
                **document,
            }
            for rank, (score, document) in enumerate(
                scored_documents[:top_n],
                start=1,
            )
        ]
