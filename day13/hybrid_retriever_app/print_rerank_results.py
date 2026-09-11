"""粗召回和 Reranker 结果输出。"""


def print_rerank_results(results):
    """打印粗召回 Top-3 和 Reranker Top-3。"""
    print(f"\nQuery: {results['query']}")
    print(f"Candidate Count: {results['candidate_count']}")

    print("\n=== 粗召回 Top-3 ===")
    for result in results["coarse_results"][:3]:
        print(
            f"Rank {result['rank']}: "
            f"{result['document_id']} - {result['title']} | "
            f"RRF Score: {result['rrf_score']:.6f}"
        )
        print(f"Text: {result['text']}")

    print("\n=== Reranker Top-3 ===")
    for result in results["reranked_results"]:
        print(
            f"Rank {result['rank']}: "
            f"{result['document_id']} - {result['title']} | "
            f"Reranker Score: {result['reranker_score']:.6f} | "
            f"Coarse Rank: {result['coarse_rank']}"
        )
        print(f"Text: {result['text']}")
