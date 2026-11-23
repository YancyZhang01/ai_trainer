"""Day15 HTTP API 的请求与响应模型。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QuestionRequest(BaseModel):
    """`/ask` 和 `/retrieve` 共用的请求参数。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10, strict=True)


class Source(BaseModel):
    """回答引用的文档来源。"""

    document_id: str
    source: str
    score: float


class RetrievedDocument(Source):
    """Retriever 返回的一条完整文档。"""

    rank: int
    text: str


class HealthResponse(BaseModel):
    status: Literal["ok"]


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    status: Literal["ok", "no_evidence"]
    latency_ms: float


class RetrieveResponse(BaseModel):
    results: list[RetrievedDocument]
    status: Literal["ok"]
    latency_ms: float
