"""Central LLM model configuration.

Selection is by environment variable only, and the default keeps the previous
OpenAI behaviour unchanged:

- Set ``LOCAL_LLM_MODEL`` (e.g. ``qwen3:0.6b-q4_K_M``) to switch every role to
  the local Ollama server.
- Override a single role with ``IAM_ROUTER_MODEL``, ``IAM_EXTRACTOR_MODEL`` or
  ``IAM_QA_MODEL`` (e.g. keep ``IAM_QA_MODEL=gpt-4o`` while routing locally).
- Leave all of them unset to keep the OpenAI models exactly as before.
"""

import os

# When set, every role below defaults to this local Ollama model.
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "").strip()

# Small/fast model for query routing and entity extraction.
ROUTER_MODEL = os.getenv("IAM_ROUTER_MODEL") or LOCAL_LLM_MODEL or "gpt-4o-mini"
EXTRACTOR_MODEL = os.getenv("IAM_EXTRACTOR_MODEL") or LOCAL_LLM_MODEL or "gpt-4o-mini"

# Larger model for user-facing generated answers (general QA, explanations).
QA_MODEL = os.getenv("IAM_QA_MODEL") or LOCAL_LLM_MODEL or "gpt-4o"
