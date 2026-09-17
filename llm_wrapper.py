"""
Wrapper for LLMs to switch between OpenAI and local models (e.g., Ollama)
without changing the rest of the code.
"""
import os
from langchain_openai import ChatOpenAI

class ConfigurableChatOpenAI(ChatOpenAI):
    def __init__(self, model_name: str, **kwargs):
        use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
        if use_local:
            # For local Ollama: set base URL and dummy API key
            kwargs.setdefault("openai_api_base", "http://localhost:11434/v1")
            kwargs.setdefault("openai_api_key", "ollama")
        # If not local, use OpenAI defaults (or user-provided via kwargs/env)
        super().__init__(model=model_name, **kwargs)