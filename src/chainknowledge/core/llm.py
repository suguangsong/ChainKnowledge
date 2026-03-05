from __future__ import annotations

from typing import Any, Dict, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from chainknowledge.core.config import AppConfig


def _build_openai_constructor_kwargs(
    api_key: str,
    model: str,
    base_url: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> List[Dict[str, Any]]:
    base = {
        "model": model,
    }
    if temperature is not None:
        base["temperature"] = temperature
    if max_tokens is not None:
        base["max_tokens"] = max_tokens

    candidates: List[Dict[str, Any]] = []
    for key_field in ("api_key", "openai_api_key"):
        for url_field in ("base_url", "openai_api_base", None):
            if base_url and url_field:
                kwargs = {key_field: api_key, "model": model, "temperature": temperature} if temperature is not None else {
                    key_field: api_key,
                    "model": model,
                }
                kwargs[url_field] = base_url
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                candidates.append(kwargs)
            elif not base_url:
                kwargs = {key_field: api_key, "model": model}
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                candidates.append(kwargs)
    return candidates


def _safe_init(factory, candidates: List[Dict[str, Any]]):
    last_error = None
    for kwargs in candidates:
        try:
            return factory(**kwargs)
        except TypeError as exc:
            last_error = exc
    if last_error is None:
        raise RuntimeError("无法创建模型客户端，参数配置为空。")
    raise last_error


def create_chat_model(
    config: AppConfig,
    provider: str | None = None,
    api_key_override: str | None = None,
    model_override: str | None = None,
    base_url_override: str | None = None,
    temperature: float = 0.15,
) -> ChatOpenAI:
    provider = (provider or config.llm_provider).lower()

    if provider == "chatglm":
        api_key = (api_key_override or config.chatglm_api_key).strip()
        base_url = (base_url_override if base_url_override is not None else config.chatglm_api_base).strip()
        model = (model_override or config.chatglm_model).strip() or config.chatglm_model
    else:
        api_key = (api_key_override or config.openai_api_key).strip()
        base_url = (base_url_override if base_url_override is not None else config.openai_api_base).strip()
        model = (model_override or config.openai_model).strip() or config.openai_model

    if not api_key:
        raise ValueError("请配置对应供应商的 API Key。")

    candidates = _build_openai_constructor_kwargs(api_key=api_key, model=model, base_url=base_url, temperature=temperature)
    return _safe_init(ChatOpenAI, candidates)


def create_embedding_model(config: AppConfig, api_key_override: str | None = None, base_url_override: str | None = None) -> OpenAIEmbeddings:
    api_key = (api_key_override or config.openai_api_key).strip()
    base_url = (base_url_override if base_url_override is not None else config.openai_api_base).strip()

    if not api_key:
        raise ValueError("请配置 OPENAI_API_KEY 以构建向量库。")

    candidates: List[Dict[str, Any]] = []
    base_kwargs = {
        "model": config.openai_embedding_model,
    }
    for key_field in ("api_key", "openai_api_key"):
        for url_field in ("base_url", "openai_api_base", None):
            if base_url and url_field:
                kwargs = {**base_kwargs, key_field: api_key, url_field: base_url}
                candidates.append(kwargs)
            elif not base_url:
                kwargs = {**base_kwargs, key_field: api_key}
                candidates.append(kwargs)
    return _safe_init(OpenAIEmbeddings, candidates)
