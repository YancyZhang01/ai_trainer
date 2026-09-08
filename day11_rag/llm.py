"""RAG Prompt 构造与大语言模型调用。"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "day10"))

from llm_client import ask_llm, get_api_key


NO_EVIDENCE_MESSAGE = "没有查找到与问题相关的事实证据"

RAG_SYSTEM_PROMPT = (
    "你是一个食品知识问答助手。只能根据用户消息中的Retrieved Context回答，"
    "不得使用Context之外的知识，也不得补充或猜测。"
    f"如果Context不能为问题提供事实依据，只回答“{NO_EVIDENCE_MESSAGE}”。"
    "允许回答Context能够支持的部分，并明确说明Context未提供的其他信息。"
    "回答应简洁，关键事实必须能在Context原文中找到。"
)


def build_prompt(query, results):
    """把 Top-K 文档合并为带来源标记的 RAG Prompt。"""
    contexts = []
    for result in results:
        contexts.append(
            f"[Document {result['document_id']}]\n"
            f"Source: {result['source']}\n"
            f"Content: {result['text']}"
        )

    context_text = "\n\n".join(contexts)
    return f"Retrieved Context:\n{context_text}\n\nQuestion:\n{query}"


def generate_answer(query, results, api_key):
    """调用现有 LLM Client，返回 answer 字符串。"""
    prompt = build_prompt(query, results)
    response = ask_llm(
        prompt,
        api_key,
        system_prompt=RAG_SYSTEM_PROMPT,
    )
    if response is None:
        return None
    return response["answer"].strip()


def load_api_key():
    """复用 llm_client 的环境变量读取逻辑。"""
    return get_api_key()
