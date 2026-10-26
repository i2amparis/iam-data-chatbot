import os
import re
import yaml
from typing import List, Dict, Any
from pathlib import Path

class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

    def _repr__(self):
        return f'Document(metadata={self.metadata})'

def load_all_yaml_files(directory: str) -> Dict[str, Any]:
    yaml_files = {}
    loaded_count = 0
    error_count = 0

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith((".yaml", ".yml")):
                full_path = os.path.join(root, filename)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = yaml.safe_load(f)
                        if content:
                            yaml_files[full_path] = content
                            loaded_count += 1
                except Exception as e:
                    error_count += 1
                    print(f"[ERROR] Failed to read {full_path}: {e}")

    # Silent loading - no debug output
    return yaml_files



# Bump when the document text produced below changes, so parsed caches keyed
# on the YAML sources are rebuilt.
DOCUMENTS_VERSION = "4"

_MAX_LISTED_COUNTRIES = 40


def _named_entries(items: Any):
    """Yield ``(name, attributes)`` for nomenclature list items.

    Entries are either a plain name (``- World``) or a single-key mapping
    (``- Emissions|CO2: {description: ..., unit: ...}``).
    """
    for item in items if isinstance(items, list) else []:
        if isinstance(item, str):
            yield item.strip(), {}
        elif isinstance(item, dict):
            for name, attributes in item.items():
                yield str(name).strip(), attributes if isinstance(attributes, dict) else {}


def _readable_template(text: str) -> str:
    """``{Electricity Source}`` -> ``<electricity source>``.

    Template placeholders read as literal content to a small answering model
    ("the variable ...|{Electricity Source} reflects ..."); plain wording does not.
    """
    return re.sub(r"\{([^{}]+)\}", lambda match: f"<{match.group(1).strip().lower()}>", str(text))


def _variable_document(name: str, attributes: dict, source: str) -> Document | None:
    # Templates ("Emissions|{Level-1 Species}") are not real variables: their
    # placeholders ended up quoted in answers ("<level-3 species>"). Exact
    # definitions of the expanded variables are answered from
    # variable_definitions instead.
    if "{" in name:
        return None
    name = _readable_template(name)
    attributes = {
        key: _readable_template(value) if isinstance(value, str) else value
        for key, value in attributes.items()
    }
    description = " ".join(str(attributes.get("description") or "").split())
    if not name or not description:
        return None
    text = f"IAM PARIS variable `{name}`: {description}"
    if attributes.get("unit"):
        text += f" Unit: {attributes['unit']}."
    notes = " ".join(str(attributes.get("notes") or "").split())
    if notes:
        text += f" Notes: {notes}"
    return Document(
        page_content=text,
        metadata={"type": "variable", "variable": name, "unit": str(attributes.get("unit") or ""), "source": source},
    )


def _region_document(name: str, group: str, attributes: dict, source: str) -> Document | None:
    if not name:
        return None
    text = f"IAM PARIS region `{name}` (group: {group})."
    description = " ".join(str(attributes.get("description") or "").split())
    if description:
        text += f" {description}"
    countries = [str(country) for country in attributes.get("countries") or []]
    if countries:
        listed = ", ".join(countries[:_MAX_LISTED_COUNTRIES])
        more = len(countries) - _MAX_LISTED_COUNTRIES
        text += f" Countries: {listed}" + (f" and {more} more." if more > 0 else ".")
    return Document(page_content=text, metadata={"type": "region", "region": name, "group": group, "source": source})


def yaml_to_documents(yaml_data: dict) -> List[Document]:
    """Convert parsed nomenclature YAML into searchable documents.

    Variable files hold ``- <variable>: {description, unit, notes, ...}``;
    region files hold ``- <group>: [<region> | {<region>: {...}}, ...]``.
    Each variable or region becomes one document naming it, so retrieval and
    the answering model both see what the text is about. Variable templates,
    ``tag_*`` files (lists used to expand those templates) and model-native
    region files (one model's own region/country lists) are not indexed.
    """
    documents = []
    for filename, items in yaml_data.items():
        path = str(filename).replace("\\", "/")
        base = os.path.basename(path)
        if base.startswith("tag_") or "/native_regions/" in path:
            continue
        if "/region/" in path:
            for group, members in _named_entries(items):
                for item in items if isinstance(items, list) else []:
                    if isinstance(item, dict) and isinstance(item.get(group), list):
                        members = item[group]
                        break
                else:
                    members = []
                for region, attributes in _named_entries(members):
                    document = _region_document(region, group, attributes, base)
                    if document:
                        documents.append(document)
            continue
        for name, attributes in _named_entries(items):
            document = _variable_document(name, attributes, base)
            if document:
                documents.append(document)
    return documents
