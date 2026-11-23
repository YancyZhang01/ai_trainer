import json
import os
import time
from io import StringIO
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


TEMPERATURE = 0.2
RETRY_MAX = 3
RETRY_DELAY = 2
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
DEFAULT_DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

load_dotenv(ENV_FILE)

# 优先使用通用变量；DOUBAO_* 和 GEMINI_API_KEY 仅用于兼容已有配置。
MODEL_NAME = os.getenv("LLM_MODEL_NAME") or os.getenv("DOUBAO_MODEL_ID")
BASE_URL = (
    os.getenv("LLM_BASE_URL")
    or os.getenv("DOUBAO_BASE_URL")
    or DEFAULT_DOUBAO_BASE_URL
)
REQUEST_TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
# 业务层已有重试，关闭 SDK 内层重试，避免一次失败等待多轮超时。
CLIENT_MAX_RETRIES = int(os.getenv("LLM_CLIENT_MAX_RETRIES", "0"))

SYSTEM_PROMPT = (
    "你是一个严谨的问答助手。请直接、准确地回答用户的问题。"
    "如果不知道答案或没有足够把握，只回答“不知道”，不要猜测或编造。"
)

JSON_OUTPUT_INSTRUCTION = (
    '只输出 JSON 对象，格式必须为 {"answer": "回答内容"}，不要输出其他文字。'
)


def get_api_key():
    load_dotenv(ENV_FILE)
    return (
        os.getenv("LLM_API_KEY")
        or os.getenv("DOUBAO_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )


def get_config_error(api_key=None):
    """返回缺失配置的明确提示；配置完整时返回 None。"""
    if not (api_key or get_api_key()):
        return "未读取到 API Key，请在 .env 中设置 LLM_API_KEY。"
    if not MODEL_NAME:
        return "未读取到模型名称，请在 .env 中设置 LLM_MODEL_NAME。"
    if not BASE_URL:
        return "未读取到 API 地址，请在 .env 中设置 LLM_BASE_URL。"
    return None


def ask_llm(question, api_key, system_prompt=SYSTEM_PROMPT):
    for attempt in range(1, RETRY_MAX + 1):
        try:
            config_error = get_config_error(api_key)
            if config_error:
                print(config_error)
                return None

            client = OpenAI(
                api_key=api_key,
                base_url=BASE_URL,
                timeout=REQUEST_TIMEOUT,
                max_retries=CLIENT_MAX_RETRIES,
            )
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": f"{system_prompt}\n{JSON_OUTPUT_INSTRUCTION}",
                    },
                    {"role": "user", "content": question},
                ],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
            )

            response_text = response.choices[0].message.content
            if not response_text:
                raise ValueError("模型没有返回内容")

            result = json.load(StringIO(response_text))
            if not isinstance(result.get("answer"), str):
                raise ValueError("模型返回的 answer 不是字符串")
            return result
        except Exception as error:
            if attempt == RETRY_MAX:
                print("调用失败")
                return None

            print(
                f"第 {attempt} 次调用失败：{error}，"
                f"{RETRY_DELAY} 秒后重试。"
            )
            time.sleep(RETRY_DELAY)


def main():
    question = input("请输入你的问题：").strip()

    if not question:
        print("输入无效：问题不能为空。")
        return

    api_key = get_api_key()
    config_error = get_config_error(api_key)
    if config_error:
        print(config_error)
        return

    answer = ask_llm(question, api_key)
    if answer is not None:
        print(json.dumps(answer, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
