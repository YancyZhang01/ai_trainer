"""Hybrid Retriever 程序入口。"""

from hybrid_retriever_app.documents import DOCUMENTS
from hybrid_retriever_app.hybrid_retriever import HybridRetriever
from hybrid_retriever_app.print_results import print_results


def main():
    query = input("请输入检索问题：").strip()
    if not query:
        print("输入无效：问题不能为空。")
        return

    retriever = HybridRetriever(DOCUMENTS)
    results = retriever.retrieve(query)
    print_results(results)

    print("\n为什么 Hybrid Retrieval 更稳定：")
    print("BM25擅长匹配明确关键词，Dense擅长理解同义表达和语义相近的问题。")
    print("RRF根据两路结果的排名进行融合，降低单一路线漏检或误排带来的影响。")


if __name__ == "__main__":
    main()
