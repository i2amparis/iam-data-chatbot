"""LLM factory: route chat models and embeddings to OpenAI or local Ollama.

Selection is configuration-only, so the OpenAI path stays intact:

- Unset everything -> identical behaviour to before (OpenAI only).
- ``LOCAL_LLM_MODEL`` (e.g. ``qwen3:0.6b-q4_K_M``) makes every role default to
  the local Ollama server, unless a role overrides it via ``IAM_ROUTER_MODEL``,
  ``IAM_EXTRACTOR_MODEL`` or ``IAM_QA_MODEL``.
- ``IAM_EMBEDDING_MODEL`` (e.g. ``nomic-embed-text``) switches embeddings to Ollama.
- ``LOCAL_LLM_REASONING_EFFORT=none`` disables reasoning tokens for models such
  as qwen3 (passed to Ollama as ``think: false``).
"""

import os

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434")

_ROLE_MODEL_VARS = ("IAM_ROUTER_MODEL", "IAM_EXTRACTOR_MODEL", "IAM_QA_MODEL")


def _split_names(value: str) -> set:
    return {name.strip().lower() for name in (value or "").split(",") if name.strip()}


def _local_model_names() -> set:
    names = _split_names(os.getenv("IAM_LOCAL_MODELS", ""))
    names |= _split_names(os.getenv("LOCAL_LLM_MODEL", ""))
    return names


def _explicit_role_models() -> set:
    return {
        (os.getenv(var) or "").strip().lower()
        for var in _ROLE_MODEL_VARS
        if (os.getenv(var) or "").strip()
    }


def is_local_model(model_name: str) -> bool:
    """Return True when the model is served by the local Ollama server."""
    name = str(model_name).strip().lower()
    if name in _local_model_names():
        return True
    # An explicitly configured role model wins over the global force-local flag.
    if name in _explicit_role_models():
        return False
    return os.getenv("USE_LOCAL_LLM", "false").strip().lower() == "true"


def _reasoning_setting():
    """Map LOCAL_LLM_REASONING_EFFORT to ChatOllama's ``reasoning`` parameter."""
    value = (os.getenv("LOCAL_LLM_REASONING_EFFORT") or "").strip().lower()
    if value in ("", "default", "auto"):
        return None
    if value in ("none", "false", "off", "0"):
        return False
    if value == "true":
        return True
    return value  # low / medium / high


def _build_local(model_name: str, **kwargs):
    # Translate OpenAI-client kwargs into ChatOllama equivalents.
    timeout = kwargs.pop("timeout", None)
    streaming = kwargs.pop("streaming", None)
    for key in ("max_retries", "openai_api_key", "api_key", "openai_api_base", "base_url"):
        kwargs.pop(key, None)

    if timeout is not None:
        kwargs.setdefault("client_kwargs", {})
        kwargs["client_kwargs"].setdefault("timeout", timeout)
    if streaming is not None:
        kwargs.setdefault("disable_streaming", not streaming)

    return ChatOllama(
        model=model_name,
        base_url=LOCAL_LLM_BASE_URL,
        reasoning=_reasoning_setting(),
        **kwargs,
    )


def get_chat_openai(model_name: str, **kwargs):
    """Return a chat client for OpenAI or for the local Ollama server."""
    if is_local_model(model_name):
        return _build_local(model_name, **kwargs)

    if "openai_api_key" not in kwargs:
        kwargs["openai_api_key"] = os.getenv("OPENAI_API_KEY")
    base_url = kwargs.pop("openai_api_base", None) or os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(model=model_name, **kwargs)


def get_embeddings(**kwargs):
    """Return Ollama embeddings when IAM_EMBEDDING_MODEL is set, else OpenAI."""
    local_model = os.getenv("IAM_EMBEDDING_MODEL", "").strip()
    if local_model:
        # Keep a bounded HTTP timeout so a slow/unreachable Ollama cannot hang
        # the boot-time FAISS build indefinitely.
        client_kwargs = dict(kwargs.get("client_kwargs") or {})
        if kwargs.get("timeout") is not None:
            client_kwargs.setdefault("timeout", kwargs["timeout"])
        return OllamaEmbeddings(
            model=local_model,
            base_url=LOCAL_LLM_BASE_URL,
            client_kwargs=client_kwargs,
        )

    env_model = os.getenv("IAM_OPENAI_EMBEDDING_MODEL", "").strip()
    if env_model:
        kwargs["model"] = env_model
    else:
        kwargs.setdefault("model", "text-embedding-3-small")
    if "api_key" not in kwargs and "openai_api_key" not in kwargs:
        kwargs["openai_api_key"] = os.getenv("OPENAI_API_KEY")
    return OpenAIEmbeddings(**kwargs)
