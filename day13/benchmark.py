"""Hybrid Retriever Benchmark 入口。"""

from hybrid_retriever_app.benchmark_runner import run_benchmark
from hybrid_retriever_app.documents import DOCUMENTS
from hybrid_retriever_app.hybrid_retriever import HybridRetriever


def main():
    retriever = HybridRetriever(DOCUMENTS)
    run_benchmark(retriever)


if __name__ == "__main__":
    main()
