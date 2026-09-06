import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


MODEL_NAME = "gemini-3.6-flash"
TEMPERATURE = 0.1
SYSTEM_PROMPT = (
     "你是一个严谨的问答助手。请直接、准确地回答用户的问题。"
    "如果你不知道答案或没有足够把握，只回答“不知道”，不要猜测或编造。"
)
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def get_api_key():
    load_dotenv(ENV_FILE)
    return os.getenv("GEMINI_API_KEY")


def ask_llm(question, api_key):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=TEMPERATURE,
        ),
    )
    return (response.text or "").strip()


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

    try:
        answer = ask_llm(question, api_key)
        print(answer if answer else "不知道")
    except Exception as error:
        print(f"调用 Gemini API 失败：{error}")


if __name__ == "__main__":
    main()
