import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_metadata import DataMetadata, build_metadata_with_cache
from link_catalog import DEFAULT_OUTPUT as DEFAULT_LINK_CATALOG


@dataclass
class RuntimeContext:
    """
    Shared resources for the IAM PARIS chatbot runtime.

    The class intentionally exposes dict-like `get` and `__getitem__` methods so
    existing agents can migrate from loose dict resources without a risky rewrite.
    """

    models: list[dict]
    ts: list[dict]
    vector_store: Any
    env: dict[str, Any]
    bot: Any = None
    metadata: DataMetadata | None = None
    link_catalog: list[dict] = field(default_factory=list)
    workspace_lookup: dict[str, list[dict]] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": self.models,
            "ts": self.ts,
            "vector_store": self.vector_store,
            "env": self.env,
            "bot": self.bot,
            "metadata": self.metadata,
            "link_catalog": self.link_catalog,
            "workspace_lookup": self.workspace_lookup,
        }


def build_workspace_lookup(ts: list[dict]) -> dict[str, list[dict]]:
    lookup: dict[str, list[dict]] = {}
    for record in ts:
        if not record:
            continue
        workspace = str(record.get("workspace_code", "unknown") or "unknown")
        lookup.setdefault(workspace, []).append(record)
    return lookup


def load_link_catalog(path: Path = DEFAULT_LINK_CATALOG) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def build_runtime_context(
    *,
    models: list[dict],
    ts: list[dict],
    vector_store: Any,
    env: dict[str, Any],
    bot: Any = None,
    workspace_lookup: dict[str, list[dict]] | None = None,
    link_catalog_path: Path = DEFAULT_LINK_CATALOG,
    metadata_cache_file: str = "cache/data_metadata.pkl",
) -> RuntimeContext:
    metadata = build_metadata_with_cache(ts, models, cache_file=metadata_cache_file)
    link_catalog = load_link_catalog(link_catalog_path)
    return RuntimeContext(
        models=models,
        ts=ts,
        vector_store=vector_store,
        env=env,
        bot=bot,
        metadata=metadata,
        link_catalog=link_catalog,
        workspace_lookup=workspace_lookup or build_workspace_lookup(ts),
    )
