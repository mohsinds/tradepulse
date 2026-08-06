from __future__ import annotations

import os
from typing import Any, Dict, Optional

DEFAULT_PROVIDER = "openai"
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "openrouter": "openai/gpt-4o-mini",
}


def configure_tracing(project: Optional[str] = None) -> bool:
    """Enable LangSmith tracing when an API key is present."""
    if not os.getenv("LANGSMITH_API_KEY"):
        return False
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault(
        "LANGSMITH_ENDPOINT", os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    )
    os.environ["LANGSMITH_PROJECT"] = (
        project or os.getenv("LANGSMITH_PROJECT") or "tradepulse"
    )
    os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]
    return True


def get_chat_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> Any:
    """Single entry point for chat models. Never instantiate provider clients directly."""
    configure_tracing()

    provider = (provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    model = model or DEFAULT_MODELS.get(provider)
    if not model:
        raise ValueError(f"No default model for provider {provider!r}")

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
            **kwargs,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            **kwargs,
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            **kwargs,
        )

    raise ValueError(f"Unsupported LLM provider: {provider!r}")


def get_chat_model_for_config(config: Dict[str, Any], **kwargs: Any) -> Any:
    """Resolve a chat model from an instrument/agent config block."""
    llm_cfg = config.get("llm") or {}
    return get_chat_model(
        provider=llm_cfg.get("provider"),
        model=llm_cfg.get("model"),
        temperature=llm_cfg.get("temperature", 0.0),
        **kwargs,
    )
