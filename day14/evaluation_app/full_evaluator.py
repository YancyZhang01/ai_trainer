"""50 条 Benchmark 的 RAG 执行、手工指标和 Ragas 指标编排。"""

import asyncio
import hashlib
import json

from day10.llm_client import MODEL_NAME

from .answer_metrics import calculate_abstention_accuracy
from .config import (
    RAGAS_JUDGE_MODEL_NAME,
    RAGAS_SCORE_CACHE_FILE,
    RETRIEVAL_TOP_K,
)
from .rag_runner import run_rag
from .ragas_evaluator import create_ragas_metrics, evaluate_with_ragas
from .report_writer import load_json, merge_by_case_id, save_json
from .retrieval_metrics import (
    calculate_context_precision_at_k,
    calculate_context_recall_at_k,
    calculate_hit_at_k,
    calculate_reciprocal_rank,
)


RAGAS_METRIC_NAMES = (
    "answer_correctness",
    "faithfulness",
    "answer_relevance",
)


def collect_rag_records(benchmarks, model, index, api_key, output_path,
                        force=False):
    """运行 RAG 并逐条保存 response/context，支持从已有记录续跑。"""
    existing = load_json(output_path, [])
    records_by_id = {record["case_id"]: record for record in existing}
    records = []
    original_order = [record["case_id"] for record in existing]

    for number, benchmark in enumerate(benchmarks, start=1):
        cached = None if force else records_by_id.get(benchmark["case_id"])
        if (
            cached
            and cached.get("question") == benchmark["question"]
            and cached.get("llm_model_name", MODEL_NAME) == MODEL_NAME
            and cached.get("generation_succeeded")
            and cached.get("answer") is not None
        ):
            record = cached
            print(
                f"[{number}/{len(benchmarks)}] 复用 {benchmark['case_id']}",
                flush=True,
            )
        else:
            print(
                f"[{number}/{len(benchmarks)}] 运行 {benchmark['case_id']}",
                flush=True,
            )
            rag_record = run_rag(
                benchmark["question"], model, index, api_key
            )
            record = {
                "case_id": benchmark["case_id"],
                "question": benchmark["question"],
                "llm_model_name": MODEL_NAME,
                **rag_record,
            }
        records.append(record)
        save_json(
            output_path,
            merge_by_case_id(existing, records, original_order),
        )

    return records


def _manual_scores(benchmark, rag_record):
    if benchmark["answerable"]:
        hit_at_3 = calculate_hit_at_k(
            rag_record["retrieved_ids"],
            benchmark["relevant_document_ids"],
            RETRIEVAL_TOP_K,
        )
        mrr = calculate_reciprocal_rank(
            rag_record["retrieved_ids"],
            benchmark["relevant_document_ids"],
        )
        context_precision = calculate_context_precision_at_k(
            rag_record["retrieved_ids"],
            benchmark["relevant_document_ids"],
            RETRIEVAL_TOP_K,
        )
        context_recall = calculate_context_recall_at_k(
            rag_record["retrieved_ids"],
            benchmark["relevant_document_ids"],
            RETRIEVAL_TOP_K,
        )
    else:
        hit_at_3 = None
        mrr = None
        context_precision = None
        context_recall = None

    abstention = (
        calculate_abstention_accuracy(
            rag_record["abstained"], benchmark["answerable"]
        )
        if rag_record["generation_succeeded"]
        else 0.0
    )
    return {
        "retrieval_hit_at_3": hit_at_3,
        "mrr": mrr,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "abstention_accuracy": abstention,
    }


def _ragas_fingerprint(benchmark, rag_record):
    """问题、回答、上下文或 Judge 模型变化时自动使旧分数失效。"""
    payload = {
        "question": benchmark["question"],
        "reference": benchmark["ground_truth_answer"],
        "response": rag_record["answer"],
        "contexts": [item["text"] for item in rag_record["contexts"]],
        "judge_model": RAGAS_JUDGE_MODEL_NAME,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


async def evaluate_records(benchmarks, rag_records, api_key, embedding_model,
                           use_ragas=True, checkpoint_path=None,
                           force_ragas=False):
    """合并 Ground Truth、运行记录、手工指标和 Ragas 自动指标。"""
    metrics = (
        create_ragas_metrics(
            api_key,
            embedding_model,
            use_prompt_cache=not force_ragas,
        )
        if use_ragas
        else None
    )
    existing = load_json(checkpoint_path, []) if checkpoint_path else []
    existing_by_id = {row["case_id"]: row for row in existing}
    existing_order = [row["case_id"] for row in existing]
    score_cache = (
        load_json(RAGAS_SCORE_CACHE_FILE, {})
        if use_ragas
        else {}
    )
    rows = []
    ragas_available = use_ragas

    for number, (benchmark, rag_record) in enumerate(
        zip(benchmarks, rag_records), start=1
    ):
        print(
            f"Ragas [{number}/{len(benchmarks)}] {benchmark['case_id']}",
            flush=True,
        )
        fingerprint = _ragas_fingerprint(benchmark, rag_record)
        cached_row = existing_by_id.get(benchmark["case_id"])
        cached_entry = score_cache.get(benchmark["case_id"], {})
        ragas_scores = {name: None for name in RAGAS_METRIC_NAMES}

        # 兼容升级前已写入 evaluation_results.json 的有效分数。
        if (
            use_ragas
            and not force_ragas
            and cached_row
            and cached_row.get("question") == benchmark["question"]
            and cached_row.get("response") == rag_record["answer"]
        ):
            for name in RAGAS_METRIC_NAMES:
                ragas_scores[name] = cached_row.get(name)

        # 新缓存带 fingerprint，能防止问题、回答、Context 或模型变化后误用。
        if cached_entry.get("fingerprint") == fingerprint:
            for name in RAGAS_METRIC_NAMES:
                if cached_entry.get("scores", {}).get(name) is not None:
                    ragas_scores[name] = cached_entry["scores"][name]

        errors = []

        def save_metric_checkpoint(scores, metric_errors, quota_exhausted):
            score_cache[benchmark["case_id"]] = {
                "fingerprint": fingerprint,
                "scores": scores,
                "errors": metric_errors,
                "quota_exhausted": quota_exhausted,
            }
            save_json(RAGAS_SCORE_CACHE_FILE, score_cache)

        if metrics is not None and ragas_available:
            ragas_scores, errors, quota_exhausted = await evaluate_with_ragas(
                benchmark,
                rag_record,
                metrics,
                cached_scores=ragas_scores,
                on_update=save_metric_checkpoint,
            )
            if quota_exhausted:
                ragas_available = False
        elif use_ragas and any(
            score is None for score in ragas_scores.values()
        ):
            errors.append("本次运行检测到 API 额度不足，未完成的 Ragas 指标已跳过。")

        if use_ragas:
            save_metric_checkpoint(ragas_scores, errors, not ragas_available)

        rows.append({
            "case_id": benchmark["case_id"],
            "case_type": benchmark["case_type"],
            "question": benchmark["question"],
            "ground_truth_answer": benchmark["ground_truth_answer"],
            "answerable": benchmark["answerable"],
            "relevant_document_ids": benchmark["relevant_document_ids"],
            "expected_keywords": benchmark["expected_keywords"],
            "retrieved_ids": rag_record["retrieved_ids"],
            "retrieved_contexts": rag_record["contexts"],
            "response": rag_record["answer"],
            "sources": rag_record["sources"],
            **ragas_scores,
            **_manual_scores(benchmark, rag_record),
            "evaluation_errors": errors,
        })
        if checkpoint_path:
            save_json(
                checkpoint_path,
                merge_by_case_id(existing, rows, existing_order),
            )

    return rows


def run_evaluation(benchmarks, rag_records, api_key, embedding_model,
                   use_ragas=True, checkpoint_path=None,
                   force_ragas=False):
    return asyncio.run(
        evaluate_records(
            benchmarks,
            rag_records,
            api_key,
            embedding_model,
            use_ragas=use_ragas,
            checkpoint_path=checkpoint_path,
            force_ragas=force_ragas,
        )
    )
