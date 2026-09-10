"""检索结果输出。"""


def print_results(results):
    """输出检索结果及完整来源信息。"""
    print(f"\n=== Top {len(results)} Chunks ===")

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Document: {result['document_id']}")
        print(f"Chunk: {result['chunk_index']}")
        print(f"Title: {result['title']}")
        print(f"Category: {result['category']}")
        print(f"Source: {result['source']}")
        print(f"Start Char: {result['start_char']}")
        print(f"End Char: {result['end_char']}")
        print(f"Text: {result['text']}")
