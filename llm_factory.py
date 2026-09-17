import os
from langchain_openai import ChatOpenAI

def get_chat_openai(model_name: str, **kwargs):
    """Factory to create ChatOpenAI instances for either OpenAI or local LLM"""
    use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    
    if use_local:
        # Local Ollama configuration
        kwargs.setdefault("openai_api_base", "http://localhost:11434/v1")
        kwargs.setdefault("openai_api_key", "ollama")  # Required but ignored
    else:
        # OpenAI configuration - ensure API key is present
        if "openai_api_key" not in kwargs:
            kwargs["openai_api_key"] = os.getenv("OPENAI_API_KEY")
    
    return ChatOpenAI(model=model_name, **kwargs)