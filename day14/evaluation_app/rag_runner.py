"""复用 day11_rag 并记录一次完整运行。"""

from day11_rag.documents import FOOD_DOCUMENTS
from day11_rag.llm import NO_EVIDENCE_MESSAGE, generate_answer
from day11_rag.retriever import has_relevant_evidence, retrieve

from .config import RETRIEVAL_TOP_K


def run_rag(question, model, index, api_key):
    """返回 Retrieved Context、Answer、Source 和运行状态。"""
    retrieved_results = retrieve(
        question,
        model,
        index,
        FOOD_DOCUMENTS,
        top_k=RETRIEVAL_TOP_K,
    )

    generation_succeeded = True
    if has_relevant_evidence(retrieved_results):
        answer = generate_answer(
            question,
            retrieved_results,
            api_key,
        )
        if answer is None:
            generation_succeeded = False
    else:
        answer = NO_EVIDENCE_MESSAGE

    abstained = answer == NO_EVIDENCE_MESSAGE
    return {
        "retrieved_ids": [
            result["document_id"]
            for result in retrieved_results
        ],
        "contexts": [
            {
                "document_id": result["document_id"],
                "score": result["score"],
                "source": result["source"],
                "text": result["text"],
            }
            for result in retrieved_results
        ],
        "answer": answer,
        "sources": [
            {
                "document_id": result["document_id"],
                "source": result["source"],
            }
            for result in retrieved_results
        ],
        "abstained": abstained,
        "generation_succeeded": generation_succeeded,
    }
