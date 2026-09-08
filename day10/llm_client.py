import json
import os
import time
from io import StringIO
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


MODEL_NAME = "gemini-3.6-flash"
TEMPERATURE = 0.2
RETRY_MAX = 3
RETRY_DELAY = 2
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

SYSTEM_PROMPT = (
    "你是一个严谨的问答助手。请直接、准确地回答用户的问题。"
    "如果不知道答案或没有足够把握，只回答“不知道”，不要猜测或编造。"
)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "对用户问题的回答",
        }
    },
    "required": ["answer"],
    "additionalProperties": False,
}


def get_api_key():
    load_dotenv(ENV_FILE)
    return os.getenv("GEMINI_API_KEY")


def ask_llm(question, api_key):
    for attempt in range(1, RETRY_MAX + 1):
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=question,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=TEMPERATURE,
                    response_mime_type="application/json",
                    response_json_schema=ANSWER_SCHEMA,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )

            if not response.text:
                raise ValueError("模型没有返回内容")

            result = json.load(StringIO(response.text))
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
    if not api_key:
        print(
            "未读取到 GEMINI_API_KEY。"
            "请在项目根目录的 .env 文件中设置：GEMINI_API_KEY=你的APIKey"
        )
        return

    answer = ask_llm(question, api_key)
    if answer is not None:
        print(json.dumps(answer, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
