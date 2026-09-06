import json
import os
import sys
from pathlib import Path


MODEL_NAME = "gemini-3.6-flash"
TEMPERATURE = 0.1
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

CATEGORIES = [
    "AI产品经理",
    "算法工程师",
    "大模型应用工程师",
    "数据分析师",
    "软件开发工程师",
    "测试工程师",
    "其他",
]

SYSTEM_PROMPT = (
    "你是一个严谨的AI岗位JD分类器。根据岗位职责、技能要求和工作内容，"
    "选择最匹配的岗位类别，并提取明确要求的技能和经验等级。"
    "confidence必须根据强相关关键词的匹配程度和数量评分："
    "多个核心关键词直接匹配时为高分，少量或间接匹配时为中等分，"
    "描述模糊或缺少相关证据时为低分。"
    "reason应简洁说明命中的关键词、匹配强度和评分依据，不要编造JD中没有的信息。"
)

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "job_category": {
            "type": "string",
            "enum": CATEGORIES,
            "description": "与JD最匹配的唯一岗位类别",
        },
        "required_skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "JD中明确要求的主要技能",
        },
        "experience_level": {
            "type": "string",
            "description": "JD要求的经验等级；无法判断时填写未知",
        },
        "ai_related": {
            "type": "boolean",
            "description": "岗位职责或技能是否与AI实质相关",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "分类置信度，范围为0到1",
        },
        "reason": {
            "type": "string",
            "description": "结合相关关键词及其数量给出的评分原因",
        },
    },
    "required": [
        "job_category",
        "required_skills",
        "experience_level",
        "ai_related",
        "confidence",
        "reason",
    ],
    "additionalProperties": False,
}


def error_result(reason):
    return {
        "job_category": "其他",
        "required_skills": [],
        "experience_level": "未知",
        "ai_related": False,
        "confidence": 0.0,
        "reason": reason,
    }


def get_api_key():
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE)
    return os.getenv("GEMINI_API_KEY")


def validate_result(result):
    if result.get("job_category") not in CATEGORIES:
        raise ValueError("模型返回了不允许的岗位类别")
    if not isinstance(result.get("required_skills"), list):
        raise ValueError("required_skills必须是数组")
    if not all(isinstance(skill, str) for skill in result["required_skills"]):
        raise ValueError("required_skills中的技能必须是字符串")
    if not isinstance(result.get("experience_level"), str):
        raise ValueError("experience_level必须是字符串")
    if not isinstance(result.get("ai_related"), bool):
        raise ValueError("ai_related必须是布尔值")

    confidence = result.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError("confidence必须是数字")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence必须在0到1之间")
    if not isinstance(result.get("reason"), str):
        raise ValueError("reason必须是字符串")

    return {
        "job_category": result["job_category"],
        "required_skills": result["required_skills"],
        "experience_level": result["experience_level"],
        "ai_related": result["ai_related"],
        "confidence": float(confidence),
        "reason": result["reason"],
    }


def classify_jd(jd_text, api_key):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=jd_text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=TEMPERATURE,
            response_mime_type="application/json",
            response_json_schema=OUTPUT_SCHEMA,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )

    if not response.text:
        raise ValueError("模型没有返回内容")

    return validate_result(json.loads(response.text))


def read_jd():
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    print(
        "请粘贴岗位JD；支持多行，输入完成后单独输入 END 并按回车：",
        file=sys.stderr,
    )

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break

        if line.strip().upper() == "END":
            break
        lines.append(line)

    return "\n".join(lines).strip()


def main():
    try:
        jd_text = read_jd()
    except KeyboardInterrupt:
        result = error_result("输入已取消")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if not jd_text:
        result = error_result("输入无效：JD内容不能为空")
    else:
        try:
            api_key = get_api_key()
            if not api_key:
                result = error_result(
                    "未读取到GEMINI_API_KEY，请在项目根目录的.env文件中配置"
                )
            else:
                result = classify_jd(jd_text, api_key)
        except Exception as error:
            result = error_result(f"分类失败：{error}")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
