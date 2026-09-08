"""最小食品知识 RAG 主流程。"""

from documents import FOOD_DOCUMENTS
from llm import NO_EVIDENCE_MESSAGE, generate_answer, load_api_key
from retriever import build_index, has_relevant_evidence, print_contexts, retrieve


def run_rag(query, model, index, api_key):
    """执行一次检索、生成答案并标注来源。"""
    results = retrieve(query, model, index, FOOD_DOCUMENTS)
    print_contexts(results)

    if not has_relevant_evidence(results):
        print(f"\n{NO_EVIDENCE_MESSAGE}")
        return

    answer = generate_answer(query, results, api_key)
    if answer is None:
        return
    if not answer or answer == NO_EVIDENCE_MESSAGE:
        print(f"\n{NO_EVIDENCE_MESSAGE}")
        return

    print(f"\nAnswer: {answer}")
    print("Source:")
    for result in results:
        print(f"- [{result['document_id']}] {result['source']}")


def main():
    query = input("请输入食品相关问题：").strip()
    if not query:
        print("输入无效：问题不能为空。")
        return

    api_key = load_api_key()
    if not api_key:
        print("未读取到 GEMINI_API_KEY，请在项目根目录的 .env 文件中设置。")
        return

    model, index = build_index(FOOD_DOCUMENTS)
    run_rag(query, model, index, api_key)


if __name__ == "__main__":
    main()
