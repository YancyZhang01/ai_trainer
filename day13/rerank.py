"""Hybrid Retrieval + Reranker 程序入口。"""

from hybrid_retriever_app.documents import DOCUMENTS
from hybrid_retriever_app.print_rerank_results import print_rerank_results
from hybrid_retriever_app.rerank_pipeline import RerankPipeline


def main():
    query = input("请输入检索问题：").strip()
    if not query:
        print("输入无效：问题不能为空。")
        return

    pipeline = RerankPipeline(DOCUMENTS)
    results = pipeline.retrieve_and_rerank(query)
    print_rerank_results(results)


if __name__ == "__main__":
    main()
