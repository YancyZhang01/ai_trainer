from retriever import search


# ============================================================
# 1. Benchmark
# 每条 query 对应一个强相关 document_id
# ============================================================

benchmarks = [
    {
        "query": "什么工具可以进行高效的向量相似性搜索？",
        "relevant_ids": ["doc_006"]
    },
    {
        "query": "什么是根据文本含义进行搜索，而不是只匹配关键词？",
        "relevant_ids": ["doc_009"]
    },
    {
        "query": "人工智能开发中常用的编程语言是什么？",
        "relevant_ids": ["doc_005"]
    },
    {
        "query": "如何让计算机理解图片和视频？",
        "relevant_ids": ["doc_019"]
    },
    {
        "query": "苹果手机也在受ai的影响",
        "relevant_ids": ["doc_001"]
    }
]


# ============================================================
# 2. Recall 计算函数
# ============================================================

def calculate_recall(search_results, relevant_ids):
    """
    Recall = 检索到的相关文档数量 / 所有相关文档数量
    """

    retrieved_ids = [
        result["document_id"]
        for result in search_results
    ]

    hit_count = sum(
        1
        for relevant_id in relevant_ids
        if relevant_id in retrieved_ids
    )

    recall = hit_count / len(relevant_ids)

    return recall, hit_count


# ============================================================
# 3. Retrieval Evaluation
# ============================================================

def evaluate_retrieval():

    total_hit = 0
    total_relevant = 0

    print("=" * 60)
    print("Retrieval Evaluation")
    print("=" * 60)

    for benchmark_index, benchmark in enumerate(
        benchmarks,
        start=1
    ):

        query = benchmark["query"]
        relevant_ids = benchmark["relevant_ids"]

        # ----------------------------------------
        # 调用 search()
        # ----------------------------------------

        results = search(
            query=query,
            top_k=1
        )

        # ----------------------------------------
        # 计算 Recall@3
        # ----------------------------------------

        recall, hit_count = calculate_recall(
            results,
            relevant_ids
        )

        total_hit += hit_count
        total_relevant += len(relevant_ids)

        # ----------------------------------------
        # 打印当前 Query
        # ----------------------------------------

        print(f"\nBenchmark {benchmark_index}")
        print("-" * 60)

        print(f"Query: {query}")
        print(f"Relevant IDs: {relevant_ids}")

        print("\nTop-3 Results:")

        # ----------------------------------------
        # 打印 Top-3
        # ----------------------------------------

        for result in results:

            print(
                f"\nRank: {result['rank']}"
            )

            print(
                f"Score: {result['score']:.4f}"
            )

            print(
                f"Text: {result['text']}"
            )

        # ----------------------------------------
        # 当前 Query Recall
        # ----------------------------------------

        print(
            f"\nRecall@3: {recall:.2%}"
        )

        print("=" * 60)


    # ========================================================
    # 4. 总 Recall
    # ========================================================

    total_recall = (
        total_hit / total_relevant
        if total_relevant > 0
        else 0
    )

    print("\n")
    print("=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    print(
        f"Total Relevant Documents: {total_relevant}"
    )

    print(
        f"Total Retrieved Relevant Documents: {total_hit}"
    )

    print(
        f"Total Recall@3: {total_recall:.2%}"
    )


# ============================================================
# 5. 程序入口
# ============================================================

if __name__ == "__main__":
    evaluate_retrieval()