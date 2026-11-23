"""从项目根目录 .env 统一读取 LLM 配置。"""

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from .exceptions import ConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def _read_float(name, default):
    value = os.getenv(name, default)
    try:
        return float(value)
    except ValueError as error:
        raise ConfigurationError(
            f"环境变量 {name} 必须是数字，当前值为：{value!r}"
        ) from error


def _read_int(name, default):
    value = os.getenv(name, default)
    try:
        return int(value)
    except ValueError as error:
        raise ConfigurationError(
            f"环境变量 {name} 必须是整数，当前值为：{value!r}"
        ) from error


@dataclass(frozen=True)
class LLMSettings:
    """一组经过校验、可传给 LLM Client 的配置。"""

    api_key: str
    model_name: str
    base_url: str
    temperature: float = 0.2
    request_timeout: float = 60.0
    client_max_retries: int = 0
    retry_max: int = 3
    retry_delay: float = 2.0

    @classmethod
    def from_env(cls, env_file=DEFAULT_ENV_FILE):
        """读取指定 .env；默认读取仓库根目录的 .env。"""
        load_dotenv(Path(env_file), override=False)

        api_key = os.getenv("LLM_API_KEY", "").strip()
        model_name = os.getenv("LLM_MODEL_NAME", "").strip()
        base_url = os.getenv("LLM_BASE_URL", "").strip()

        missing = [
            name
            for name, value in (
                ("LLM_API_KEY", api_key),
                ("LLM_MODEL_NAME", model_name),
                ("LLM_BASE_URL", base_url),
            )
            if not value
        ]
        if missing:
            raise ConfigurationError(
                ".env 缺少必要配置：" + ", ".join(missing)
            )

        parsed_base_url = urlparse(base_url)
        if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.netloc:
            raise ConfigurationError(
                "LLM_BASE_URL 必须是完整的 HTTP(S) 地址，"
                "不能保留 .env.example 中的占位文字。"
            )

        temperature = _read_float("LLM_TEMPERATURE", "0.2")
        request_timeout = _read_float("LLM_TIMEOUT_SECONDS", "60")
        client_max_retries = _read_int("LLM_CLIENT_MAX_RETRIES", "0")
        retry_max = _read_int("LLM_RETRY_MAX", "3")
        retry_delay = _read_float("LLM_RETRY_DELAY_SECONDS", "2")

        if not 0 <= temperature <= 2:
            raise ConfigurationError("LLM_TEMPERATURE 必须在 0 到 2 之间。")
        if request_timeout <= 0:
            raise ConfigurationError("LLM_TIMEOUT_SECONDS 必须大于 0。")
        if client_max_retries < 0:
            raise ConfigurationError("LLM_CLIENT_MAX_RETRIES 不能小于 0。")
        if retry_max < 1:
            raise ConfigurationError("LLM_RETRY_MAX 必须至少为 1。")
        if retry_delay < 0:
            raise ConfigurationError("LLM_RETRY_DELAY_SECONDS 不能小于 0。")

        return cls(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            request_timeout=request_timeout,
            client_max_retries=client_max_retries,
            retry_max=retry_max,
            retry_delay=retry_delay,
        )


def get_llm_settings(env_file=DEFAULT_ENV_FILE):
    """提供简短、统一的配置加载入口。"""
    return LLMSettings.from_env(env_file)
