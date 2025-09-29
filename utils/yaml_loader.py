import os
import yaml
from typing import List, Dict
from pathlib import Path

class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

    def _repr__(self):
        return f'Document(metadata={self.metadata})'

def load_all_yaml_files(directory: str) -> Dict[str, dict]:
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



def yaml_to_documents(yaml_dict: Dict[str, dict]) -> List[Document]:
    docs = []
    for file_path, content in yaml_dict.items():
        if not content:
            continue
        doc = Document(
            page_content=str(content),  # You can format this better if needed
            metadata={"source": file_path}
        )
        docs.append(doc)
    return docs
