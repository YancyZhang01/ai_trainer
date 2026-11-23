"""公共基础层使用的异常类型。"""


class SharedInfrastructureError(Exception):
    """公共基础设施异常的基类。"""


class ConfigurationError(SharedInfrastructureError):
    """环境变量缺失或格式不正确。"""


class LLMCallError(SharedInfrastructureError):
    """重试后仍无法完成 LLM 请求。"""


class LLMResponseError(SharedInfrastructureError):
    """LLM 返回空内容或不符合预期格式。"""
