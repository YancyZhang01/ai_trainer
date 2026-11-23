"""运行 50 条 RAG Benchmark 并生成完整模型评测表。"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from day11_rag.documents import FOOD_DOCUMENTS
from day11_rag.llm import get_config_error, load_api_key
from day11_rag.retriever import build_index
from day14.evaluation_app.benchmark_data import BENCHMARKS, validate_benchmarks
from day14.evaluation_app.config import (
    EVALUATION_CSV_FILE,
    EVALUATION_JSON_FILE,
    RAG_RECORDS_FILE,
    SUMMARY_FILE,
)
from day14.evaluation_app.full_evaluator import (
    collect_rag_records,
    run_evaluation,
)
from day14.evaluation_app.report_writer import (
    build_summary,
    load_json,
    merge_by_case_id,
    print_evaluation_table,
    save_csv,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="运行完整 RAG Evaluation")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅运行前 N 条，适合先做小规模验证",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="只运行 RAG 和手工检索指标，不调用 Ragas Judge",
    )
    parser.add_argument(
        "--force-rag",
        action="store_true",
        help="忽略已有 rag_records.json，重新调用 RAG",
    )
    parser.add_argument(
        "--force-ragas",
        action="store_true",
        help="忽略已有自动评测分数，重新调用 Ragas Judge",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    validate_benchmarks()
    benchmarks = BENCHMARKS[:args.limit] if args.limit else BENCHMARKS
    if not benchmarks:
        print("没有可评测的 Benchmark。")
        return

    api_key = load_api_key()
    config_error = get_config_error(api_key)
    if config_error:
        print(config_error)
        return

    print(f"Benchmark 数量：{len(benchmarks)}")
    model, index = build_index(FOOD_DOCUMENTS)
    rag_records = collect_rag_records(
        benchmarks,
        model,
        index,
        api_key,
        RAG_RECORDS_FILE,
        force=args.force_rag,
    )
    rows = run_evaluation(
        benchmarks,
        rag_records,
        api_key,
        model,
        use_ragas=not args.skip_ragas,
        checkpoint_path=EVALUATION_JSON_FILE,
        force_ragas=args.force_ragas,
    )
    existing_rows = load_json(EVALUATION_JSON_FILE, [])
    case_order = [benchmark["case_id"] for benchmark in BENCHMARKS]
    all_rows = merge_by_case_id(existing_rows, rows, case_order)
    summary = build_summary(all_rows)

    save_json(EVALUATION_JSON_FILE, all_rows)
    save_csv(EVALUATION_CSV_FILE, all_rows)
    save_json(SUMMARY_FILE, summary)
    print_evaluation_table(rows)

    print("\n评测文件：")
    print(f"- RAG 原始记录：{RAG_RECORDS_FILE}")
    print(f"- 完整 JSON 表：{EVALUATION_JSON_FILE}")
    print(f"- 完整 CSV 表：{EVALUATION_CSV_FILE}")
    print(f"- 汇总指标：{SUMMARY_FILE}")


if __name__ == "__main__":
    main()
