"""Evaluation 配置。"""

import os
from pathlib import Path

from day10.llm_client import (
    BASE_URL,
    CLIENT_MAX_RETRIES,
    MODEL_NAME,
    REQUEST_TIMEOUT,
)

RETRIEVAL_TOP_K = 3
RAGAS_JUDGE_MODEL_NAME = os.getenv("RAGAS_MODEL_NAME") or MODEL_NAME
RAGAS_OPENAI_BASE_URL = os.getenv("RAGAS_BASE_URL") or BASE_URL
RAGAS_REQUEST_TIMEOUT = REQUEST_TIMEOUT
RAGAS_CLIENT_MAX_RETRIES = CLIENT_MAX_RETRIES
RAGAS_METRIC_TIMEOUT = float(
    os.getenv("RAGAS_METRIC_TIMEOUT_SECONDS", "120")
)


def get_ragas_api_key(default_api_key):
    """允许 Judge 使用独立 Key；未配置时复用 RAG 的 Key。"""
    return os.getenv("RAGAS_API_KEY") or default_api_key

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RAG_RECORDS_FILE = RESULTS_DIR / "rag_records.json"
EVALUATION_JSON_FILE = RESULTS_DIR / "evaluation_results.json"
EVALUATION_CSV_FILE = RESULTS_DIR / "evaluation_results.csv"
SUMMARY_FILE = RESULTS_DIR / "evaluation_summary.json"
RAGAS_SCORE_CACHE_FILE = RESULTS_DIR / "ragas_score_cache.json"
RAGAS_PROMPT_CACHE_DIR = RESULTS_DIR / "ragas_prompt_cache"
