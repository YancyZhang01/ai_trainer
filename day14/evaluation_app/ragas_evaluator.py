"""使用 Ragas 0.4.x 对一次 RAG 回答进行自动评测。"""

import asyncio
import sys
import types


def _install_ragas_compatibility_shim():
    """兼容 Ragas 0.4.3 与新版 langchain-community 的导入差异。"""
    module_name = "langchain_community.chat_models.vertexai"
    try:
        __import__(module_name)
    except ModuleNotFoundError:
        compatibility_module = types.ModuleType(module_name)
        compatibility_module.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[module_name] = compatibility_module


_install_ragas_compatibility_shim()

from openai import AsyncOpenAI
from ragas.cache import DiskCacheBackend
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    Faithfulness,
)

from .config import (
    RAGAS_CLIENT_MAX_RETRIES,
    RAGAS_JUDGE_MODEL_NAME,
    RAGAS_METRIC_TIMEOUT,
    RAGAS_OPENAI_BASE_URL,
    RAGAS_PROMPT_CACHE_DIR,
    RAGAS_REQUEST_TIMEOUT,
    get_ragas_api_key,
)


class ExistingEmbeddingModel(BaseRagasEmbedding):
    """让 Ragas 复用检索阶段已经加载的 SentenceTransformer。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def embed_text(self, text, **kwargs):
        return self.model.encode(text, normalize_embeddings=True).tolist()

    async def aembed_text(self, text, **kwargs):
        return self.embed_text(text, **kwargs)

    def embed_texts(self, texts, **kwargs):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    async def aembed_texts(self, texts, **kwargs):
        return self.embed_texts(texts, **kwargs)


def create_ragas_metrics(api_key, embedding_model, use_prompt_cache=True):
    """创建共享的 Ragas Judge、Embedding 与三个 LLM 指标。"""
    client = AsyncOpenAI(
        api_key=get_ragas_api_key(api_key),
        base_url=RAGAS_OPENAI_BASE_URL,
        timeout=RAGAS_REQUEST_TIMEOUT,
        max_retries=RAGAS_CLIENT_MAX_RETRIES,
    )
    prompt_cache = (
        DiskCacheBackend(str(RAGAS_PROMPT_CACHE_DIR))
        if use_prompt_cache
        else None
    )
    judge_llm = llm_factory(
        RAGAS_JUDGE_MODEL_NAME,
        provider="openai",
        client=client,
        adapter="instructor",
        temperature=0.0,
        cache=prompt_cache,
    )
    embeddings = ExistingEmbeddingModel(embedding_model)

    return {
        # 先运行调用次数较少的指标，尽快确认 Judge 通路是否正常。
        "answer_relevance": AnswerRelevancy(
            llm=judge_llm,
            embeddings=embeddings,
            strictness=1,
        ),
        "faithfulness": Faithfulness(llm=judge_llm),
        "answer_correctness": AnswerCorrectness(
            llm=judge_llm,
            embeddings=embeddings,
        ),
    }


def _number_or_none(metric_result):
    """把 MetricResult 转成可写入 JSON/CSV 的普通浮点数。"""
    value = float(metric_result.value)
    return None if value != value else value


def _is_quota_error(error):
    text = str(error).lower()
    return (
        type(error).__name__ == "RateLimitError"
        or "setlimitexceeded" in text
        or "quota" in text
        or "error code: 429" in text
    )


async def evaluate_with_ragas(
    benchmark,
    rag_record,
    metrics,
    cached_scores=None,
    on_update=None,
):
    """按指标复用并评测；遇到额度错误时立即停止远程调用。"""
    cached_scores = cached_scores or {}
    scores = {
        name: cached_scores.get(name)
        for name in metrics
    }
    errors = []
    quota_exhausted = False

    if (
        not benchmark["answerable"]
        or not rag_record["generation_succeeded"]
        or rag_record["abstained"]
    ):
        return scores, errors, quota_exhausted

    common = {
        "user_input": benchmark["question"],
        "response": rag_record["answer"],
        "reference": benchmark["ground_truth_answer"],
        "retrieved_contexts": [
            context["text"] for context in rag_record["contexts"]
        ],
    }
    metric_arguments = {
        "answer_correctness": {
            key: common[key] for key in ("user_input", "response", "reference")
        },
        "faithfulness": {
            key: common[key]
            for key in ("user_input", "response", "retrieved_contexts")
        },
        "answer_relevance": {
            key: common[key] for key in ("user_input", "response")
        },
    }

    for name, metric in metrics.items():
        if scores[name] is not None:
            print(f"  == {name} 复用：{scores[name]}", flush=True)
            continue

        print(f"  -> {name} 开始", flush=True)
        try:
            result = await asyncio.wait_for(
                metric.ascore(**metric_arguments[name]),
                timeout=RAGAS_METRIC_TIMEOUT,
            )
            scores[name] = _number_or_none(result)
            print(
                f"  <- {name} 完成：{scores[name]}",
                flush=True,
            )
        except TimeoutError:
            message = (
                f"{name}: TimeoutError: 超过 "
                f"{RAGAS_METRIC_TIMEOUT:.0f} 秒"
            )
            errors.append(message)
            print(f"  !! {message}", flush=True)
        except Exception as error:
            message = f"{name}: {type(error).__name__}: {error}"
            errors.append(message)
            print(f"  !! {message}", flush=True)
            if _is_quota_error(error):
                quota_exhausted = True

        if on_update:
            on_update(scores, errors, quota_exhausted)
        if quota_exhausted:
            print("  !! API 额度不足，停止后续 Ragas 请求", flush=True)
            break

    return scores, errors, quota_exhausted
