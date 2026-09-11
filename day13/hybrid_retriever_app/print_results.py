"""三组检索结果输出。"""


def print_results(results):
    """分别打印 BM25、Dense 和 Hybrid 排名。"""
    result_groups = (
        ("BM25 Top-N", results["bm25_results"], "score"),
        ("Dense Top-N", results["dense_results"], "score"),
        ("Hybrid Top-K", results["hybrid_results"], "rrf_score"),
    )

    for title, group, score_name in result_groups:
        print(f"\n=== {title} ===")
        if not group:
            print("没有匹配结果。")
            continue

        for result in group:
            print(f"\nRank: {result['rank']}")
            print(f"Document: {result['document_id']} - {result['title']}")
            print(f"Score: {result[score_name]:.6f}")
            print(f"Category: {result['category']}")
            print(f"Source: {result['source']}")
            if score_name == "rrf_score":
                print(f"BM25 Rank: {result['bm25_rank']}")
                print(f"Dense Rank: {result['dense_rank']}")
            print(f"Text: {result['text']}")
