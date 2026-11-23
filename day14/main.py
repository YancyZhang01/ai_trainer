"""最小 RAG Evaluation 系统入口。"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from day11_rag.documents import FOOD_DOCUMENTS
from day11_rag.llm import get_config_error, load_api_key
from day11_rag.retriever import build_index
from day14.evaluation_app.evaluator import evaluate_benchmarks


def main():
    api_key = load_api_key()
    config_error = get_config_error(api_key)
    if config_error:
        print(config_error)
        return

    model, index = build_index(FOOD_DOCUMENTS)
    evaluate_benchmarks(model, index, api_key)


if __name__ == "__main__":
    main()
