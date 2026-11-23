"""供各 Day 项目复用的 OpenAI-compatible LLM Client。"""

import json
import time

from openai import OpenAI

from .exceptions import LLMCallError, LLMResponseError
from .settings import LLMSettings, get_llm_settings


class LLMClient:
    """统一处理模型配置、请求、重试和响应解析。"""

    def __init__(self, settings=None):
        self.settings = settings or get_llm_settings()
        self._client = OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            timeout=self.settings.request_timeout,
            max_retries=self.settings.client_max_retries,
        )

    @staticmethod
    def _build_messages(user_prompt, system_prompt=None):
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("user_prompt 不能为空。")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt.strip()})
        return messages

    def _request(self, messages, response_format=None, temperature=None):
        last_error = None
        for attempt in range(1, self.settings.retry_max + 1):
            try:
                request = {
                    "model": self.settings.model_name,
                    "messages": messages,
                    "temperature": (
                        self.settings.temperature
                        if temperature is None
                        else temperature
                    ),
                }
                if response_format is not None:
                    request["response_format"] = response_format

                response = self._client.chat.completions.create(**request)
                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise LLMResponseError("模型没有返回有效内容。")
                return content.strip()
            except LLMResponseError:
                raise
            except Exception as error:
                last_error = error
                if attempt < self.settings.retry_max:
                    time.sleep(self.settings.retry_delay)

        raise LLMCallError(
            f"调用 LLM 失败，已重试 {self.settings.retry_max} 次：{last_error}"
        ) from last_error

    def generate_text(self, user_prompt, system_prompt=None,
                      temperature=None):
        """返回模型生成的纯文本。"""
        messages = self._build_messages(user_prompt, system_prompt)
        return self._request(messages, temperature=temperature)

    def generate_json(self, user_prompt, system_prompt=None,
                      temperature=None):
        """要求模型返回 JSON 对象并解析为 dict。"""
        messages = self._build_messages(user_prompt, system_prompt)
        content = self._request(
            messages,
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        try:
            result = json.loads(content)
        except json.JSONDecodeError as error:
            raise LLMResponseError(
                f"模型返回的内容不是有效 JSON：{error}"
            ) from error
        if not isinstance(result, dict):
            raise LLMResponseError("模型返回的 JSON 顶层必须是对象。")
        return result
