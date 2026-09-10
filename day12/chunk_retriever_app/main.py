"""长文本语义检索程序入口。"""

from .build_chunks import build_chunks
from .build_index import build_index
from .documents import DOCUMENTS
from .evaluate_retrieval import evaluate_retrieval
from .print_results import print_results
from .search import search


def main():
    """构建索引、处理用户查询并执行 Benchmark。"""
    chunks = build_chunks(DOCUMENTS)
    model, index = build_index(chunks)

    print("Document 数量:", len(DOCUMENTS))
    print("Chunk 数量:", len(chunks))

    query = input("请输入检索问题：").strip()
    if not query:
        print("输入无效：问题不能为空。")
        return

    results = search(query, model, index, chunks)
    print_results(results)
    evaluate_retrieval(model, index, chunks)
