"""Central LLM model configuration.

gpt-4-turbo is deprecated, slow and expensive. Routing/entity extraction are
simple classification tasks that a small fast model handles well; only the
user-facing Q&A answers use the larger model. Override per deployment with
environment variables.
"""

import os

# Small/fast model for query routing and entity extraction.
ROUTER_MODEL = os.getenv("IAM_ROUTER_MODEL", "gpt-4o-mini")
EXTRACTOR_MODEL = os.getenv("IAM_EXTRACTOR_MODEL", "gpt-4o-mini")

# Larger model for user-facing generated answers (general QA, explanations).
QA_MODEL = os.getenv("IAM_QA_MODEL", "gpt-4o")
