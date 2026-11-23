"""把已有食品知识 RAG 封装为 FastAPI 服务。"""

import logging
import sys
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import perf_counter

# 兼容 VS Code 的“Run Python File”直接执行方式。
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, status
from starlette.concurrency import run_in_threadpool

from day11_rag.documents import FOOD_DOCUMENTS
from shared import (
    ConfigurationError,
    LLMCallError,
    LLMClient,
    LLMResponseError,
)

from day15.schemas import (
    AskResponse,
    HealthResponse,
    QuestionRequest,
    RetrieveResponse,
    RetrievedDocument,
    Source,
)


logger = logging.getLogger(__name__)

NO_EVIDENCE_MESSAGE = "没有查找到与问题相关的事实证据"
RAG_SYSTEM_PROMPT = (
    "你是一个食品知识问答助手。"
    "只能根据用户消息中的 Retrieved Context 回答，"
    "不得使用 Context 之外的知识，也不得猜测或编造。"
    f"如果 Context 无法支持答案，只回答“{NO_EVIDENCE_MESSAGE}”。"
    "回答应简洁，关键事实必须能在 Context 原文中找到。"
)

app = FastAPI(
    title="Food Knowledge RAG API",
    version="1.0.0",
)

_retriever_resources = None
_retriever_lock = Lock()


def _get_retriever_resources():
    """首次检索时复用已有 build_index，并缓存本地模型和索引。"""
    global _retriever_resources
    if _retriever_resources is None:
        with _retriever_lock:
            if _retriever_resources is None:
                from day11_rag.retriever import build_index

                _retriever_resources = build_index(FOOD_DOCUMENTS)
    return _retriever_resources


@lru_cache(maxsize=1)
def _get_llm_client():
    """复用 shared 公共层，API Key 继续从根目录 .env 读取。"""
    return LLMClient()


def _retrieve_documents(question, top_k):
    from day11_rag.retriever import retrieve

    model, index = _get_retriever_resources()
    return retrieve(
        question,
        model,
        index,
        FOOD_DOCUMENTS,
        top_k=top_k,
    )


def _has_relevant_evidence(results):
    from day11_rag.retriever import has_relevant_evidence

    return has_relevant_evidence(results)


def _build_rag_prompt(question, results):
    contexts = []
    for result in results:
        contexts.append(
            f"[Document {result['document_id']}]\n"
            f"Source: {result['source']}\n"
            f"Content: {result['text']}"
        )
    return (
        "Retrieved Context:\n"
        + "\n\n".join(contexts)
        + f"\n\nQuestion:\n{question}"
    )


def _to_sources(results):
    return [
        Source(
            document_id=result["document_id"],
            source=result["source"],
            score=result["score"],
        )
        for result in results
    ]


def _raise_retriever_error(error):
    """把依赖不可用和普通检索异常映射为明确的 HTTP 错误。"""
    logger.exception("Retriever 执行失败")
    is_windows_policy_block = (
        getattr(error, "winerror", None) == 4551
        or "WinError 4551" in str(error)
    )
    if is_windows_policy_block:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "error",
                "message": (
                    "Retriever 依赖被 Windows 应用控制策略阻止，"
                    "请检查服务端日志中的 torch_python.dll 错误。"
                ),
            },
        ) from error

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={
            "status": "error",
            "message": "Retriever 执行失败。",
        },
    ) from error


@app.get("/health", response_model=HealthResponse)
def health():
    """轻量存活检查，不触发模型加载或远程 API 调用。"""
    return HealthResponse(status="ok")


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(request: QuestionRequest):
    """只运行已有 Retriever，不调用 LLM。"""
    started_at = perf_counter()
    try:
        results = await run_in_threadpool(
            _retrieve_documents,
            request.question,
            request.top_k,
        )
    except Exception as error:
        _raise_retriever_error(error)

    return RetrieveResponse(
        results=[RetrievedDocument(**result) for result in results],
        status="ok",
        latency_ms=round((perf_counter() - started_at) * 1000, 2),
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: QuestionRequest):
    """运行 Retriever，再根据检索证据调用 LLM 生成回答。"""
    started_at = perf_counter()
    try:
        results = await run_in_threadpool(
            _retrieve_documents,
            request.question,
            request.top_k,
        )
    except Exception as error:
        _raise_retriever_error(error)

    if not _has_relevant_evidence(results):
        return AskResponse(
            answer=NO_EVIDENCE_MESSAGE,
            sources=[],
            status="no_evidence",
            latency_ms=round((perf_counter() - started_at) * 1000, 2),
        )

    prompt = _build_rag_prompt(request.question, results)
    try:
        client = _get_llm_client()
        answer = await run_in_threadpool(
            client.generate_text,
            prompt,
            RAG_SYSTEM_PROMPT,
        )
    except ConfigurationError as error:
        logger.exception("LLM 配置无效")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "error",
                "message": str(error),
            },
        ) from error
    except LLMCallError as error:
        logger.exception("LLM 服务调用失败")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "error",
                "message": "LLM 服务暂时不可用。",
            },
        ) from error
    except LLMResponseError as error:
        logger.exception("LLM 返回内容无效")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "status": "error",
                "message": "LLM 返回了无效响应。",
            },
        ) from error
    except Exception as error:
        logger.exception("生成答案时发生未处理异常")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "生成答案时发生内部错误。",
            },
        ) from error

    return AskResponse(
        answer=answer,
        sources=_to_sources(results),
        status="ok",
        latency_ms=round((perf_counter() - started_at) * 1000, 2),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
