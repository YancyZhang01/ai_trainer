"""食品 RAG 的 Benchmark 数据和简单评测逻辑。"""

from documents import FOOD_DOCUMENTS
from llm import NO_EVIDENCE_MESSAGE, generate_answer, load_api_key
from retriever import TOP_K, build_index, has_relevant_evidence, print_contexts, retrieve


BENCHMARKS = [
    {
        "question": "酸奶是如何制成的？",
        "relevant_id": "food_003",
        "expected_keyword": "乳酸菌",
    },
    {
        "question": "豆腐的主要原料是什么？",
        "relevant_id": "food_004",
        "expected_keyword": "大豆",
    },
    {
        "question": "马吉利是干什么的？",
    "relevant_id": None,
    "expected_keyword": "没有查找到",
    },
]


def evaluate_case(benchmark, model, index, api_key):
    """评估单个问题的检索命中和答案关键词命中。"""
    results = retrieve(
        benchmark["question"],
        model,
        index,
        FOOD_DOCUMENTS,
    )
    print_contexts(results)

    retrieved_ids = [result["document_id"] for result in results]
    retrieval_hit = benchmark["relevant_id"] in retrieved_ids

    if has_relevant_evidence(results):
        answer = generate_answer(
            benchmark["question"],
            results,
            api_key,
        )
    else:
        answer = NO_EVIDENCE_MESSAGE

    answer = answer or ""
    keyword_hit = benchmark["expected_keyword"] in answer

    return {
        "answer": answer,
        "retrieval_hit": retrieval_hit,
        "keyword_hit": keyword_hit,
        "passed": retrieval_hit and keyword_hit,
    }


def evaluate():
    """运行全部 Benchmark，并输出简单汇总指标。"""
    api_key = load_api_key()
    if not api_key:
        print("未读取到 GEMINI_API_KEY，请在项目根目录的 .env 文件中设置。")
        return

    model, index = build_index(FOOD_DOCUMENTS)
    retrieval_hits = 0
    keyword_hits = 0
    passed_cases = 0

    for case_number, benchmark in enumerate(BENCHMARKS, start=1):
        print(f"\n{'=' * 60}")
        print(f"Benchmark {case_number}: {benchmark['question']}")

        result = evaluate_case(benchmark, model, index, api_key)
        retrieval_hits += int(result["retrieval_hit"])
        keyword_hits += int(result["keyword_hit"])
        passed_cases += int(result["passed"])

        print(f"Answer: {result['answer']}")
        print(f"Relevant ID Hit: {result['retrieval_hit']}")
        print(f"Expected Keyword Hit: {result['keyword_hit']}")
        print(f"Accepted: {result['passed']}")

    total = len(BENCHMARKS)
    print(f"\n{'=' * 60}")
    print("Final Evaluation")
    print(f"Recall@{TOP_K}: {retrieval_hits / total:.2%}")
    print(f"Keyword Accuracy: {keyword_hits / total:.2%}")
    print(f"Acceptance Rate: {passed_cases / total:.2%}")


if __name__ == "__main__":
    evaluate()
