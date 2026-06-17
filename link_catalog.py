import argparse
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


SOURCE_EXCEL = Path("iamparis_chatbot_links.xlsx")
DEFAULT_OUTPUT = Path("docs/iamparis_link_catalog.json")

_MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


@dataclass(frozen=True)
class LinkCatalogEntry:
    id: str
    title: str
    url: str
    category: str
    project: str = ""
    workspace: str = ""
    item_type: str = ""
    keywords: list[str] | None = None
    verified_direct_url: bool = False
    search_hint: str = ""
    fallback_instruction: str = ""
    source_sheet: str = ""
    status: str = ""


def _slugify(value: str) -> str:
    value = str(value or "").strip().lower()
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "entry"


def _column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in str(cell_ref or "") if ch.isalpha())
    index = 0
    for ch in letters:
        index = index * 26 + ord(ch.upper()) - 64
    return max(index - 1, 0)


def _resolve_sheet_path(base_path: str, target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    return str(PurePosixPath(base_path).parent.joinpath(target))


def _read_shared_strings(zip_file: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zip_file.namelist():
        return []
    root = ET.fromstring(zip_file.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall(f"{_MAIN_NS}si"):
        strings.append("".join(t.text or "" for t in item.iter(f"{_MAIN_NS}t")))
    return strings


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    if cell.attrib.get("t") == "inlineStr":
        return "".join(t.text or "" for t in cell.iter(f"{_MAIN_NS}t")).strip()

    value = cell.find(f"{_MAIN_NS}v")
    if value is None:
        return ""

    raw = value.text or ""
    if cell.attrib.get("t") == "s" and raw.isdigit():
        index = int(raw)
        if index < len(shared_strings):
            return shared_strings[index].strip()
    return raw.strip()


def read_xlsx_tables(path: Path = SOURCE_EXCEL) -> dict[str, list[dict[str, str]]]:
    """
    Read simple tabular sheets from an .xlsx file without openpyxl.

    The IAM PARIS link workbook is deliberately simple: first row headers, then
    text/numeric values. This parser keeps runtime catalog generation independent
    of optional Excel dependencies.
    """
    tables: dict[str, list[dict[str, str]]] = {}
    with ZipFile(path) as zip_file:
        shared_strings = _read_shared_strings(zip_file)

        workbook_path = "xl/workbook.xml"
        workbook = ET.fromstring(zip_file.read(workbook_path))
        rels = ET.fromstring(zip_file.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}

        sheets = workbook.find(f"{_MAIN_NS}sheets")
        if sheets is None:
            return tables

        for sheet in sheets:
            sheet_name = sheet.attrib["name"]
            rel_id = sheet.attrib[f"{_REL_NS}id"]
            sheet_path = _resolve_sheet_path(workbook_path, rel_targets[rel_id])
            sheet_root = ET.fromstring(zip_file.read(sheet_path))

            rows: list[list[str]] = []
            for row in sheet_root.findall(f".//{_MAIN_NS}row"):
                cells: dict[int, str] = {}
                max_index = -1
                for cell in row.findall(f"{_MAIN_NS}c"):
                    index = _column_index(cell.attrib.get("r", "A1"))
                    max_index = max(max_index, index)
                    cells[index] = _cell_value(cell, shared_strings)
                rows.append([cells.get(i, "") for i in range(max_index + 1)])

            if not rows:
                tables[sheet_name] = []
                continue

            headers = [header.strip() for header in rows[0]]
            records: list[dict[str, str]] = []
            for row in rows[1:]:
                if not any(str(value).strip() for value in row):
                    continue
                record = {
                    headers[i]: str(row[i]).strip() if i < len(row) else ""
                    for i in range(len(headers))
                    if headers[i]
                }
                records.append(record)
            tables[sheet_name] = records

    return tables


def _split_keywords(*values: str) -> list[str]:
    keywords: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        pieces = re.split(r";|\n", text)
        for piece in pieces:
            cleaned = re.sub(r"\s+", " ", piece).strip(" ,")
            if cleaned and cleaned.lower() not in {k.lower() for k in keywords}:
                keywords.append(cleaned)
    return keywords


def _is_verified_direct_url(url: str, status: str) -> bool:
    url = str(url or "")
    status_lower = str(status or "").lower()
    if not url.startswith("https://iamparis.eu"):
        return False
    if "detail url not exposed" in status_lower:
        return False
    if "general library link" in status_lower:
        return False
    if "general model directory" in status_lower:
        return False
    return "verified" in status_lower or "observed link" in status_lower or "/application_library/" in url


def _infer_main_category(row: dict[str, str]) -> str:
    category = (row.get("Category") or "").lower()
    title = (row.get("Subcategory / Page") or "").lower()
    if "model" in category or "model" in title:
        return "models"
    if "result" in category or "result" in title:
        return "results"
    if "application" in category or "library" in title:
        return "application_library"
    if "analysis" in title:
        return "analysis"
    if "contact" in title:
        return "contact"
    if "data stor" in title:
        return "data_stories"
    return "main"


def _infer_result_project(title: str, url: str) -> str:
    text = f"{title} {url}".lower()
    projects = {
        "ndc-aspects": "NDC ASPECTS",
        "iam-compact": "IAM COMPACT",
        "paris-reinforce": "PARIS REINFORCE",
        "diamond": "DIAMOND",
        "enclude": "ENCLUDE",
        "decipher": "DECIPHER",
        "transience": "TRANSIENCE",
        "eu-china": "EU-CHINA BRIDGE",
    }
    for token, project in projects.items():
        if token in text:
            return project
    return ""


def _infer_workspace(url: str) -> str:
    match = re.search(r"/results/[^/]+/([^/]+)", str(url or ""))
    return match.group(1) if match else ""


def _entry(
    *,
    prefix: str,
    title: str,
    url: str,
    category: str,
    item_type: str,
    source_sheet: str,
    status: str = "",
    project: str = "",
    workspace: str = "",
    keywords: list[str] | None = None,
    search_hint: str = "",
) -> LinkCatalogEntry | None:
    title = str(title or "").strip()
    url = str(url or "").strip()
    if not title or not url:
        return None
    return LinkCatalogEntry(
        id=f"{prefix}-{_slugify(title)}-{_slugify(url)}",
        title=title,
        url=url,
        category=category,
        project=project,
        workspace=workspace,
        item_type=item_type,
        keywords=keywords or [],
        verified_direct_url=_is_verified_direct_url(url, status),
        search_hint=search_hint,
        fallback_instruction=_fallback_instruction(url, search_hint, status),
        source_sheet=source_sheet,
        status=status,
    )


def _fallback_instruction(url: str, search_hint: str, status: str) -> str:
    if _is_verified_direct_url(url, status):
        return ""
    if str(url or "").rstrip("/") == "https://iamparis.eu/application_library" and search_hint:
        return f"Open the Application Library and search for: {search_hint}"
    if "detail url not exposed" in str(status or "").lower() and search_hint:
        return f"Direct detail URL is not exposed by the rendered card; search for: {search_hint}"
    return ""


def build_link_catalog(path: Path = SOURCE_EXCEL) -> list[dict]:
    tables = read_xlsx_tables(path)
    entries: list[LinkCatalogEntry] = []

    for row in tables.get("01_Main_Routes", []):
        entry = _entry(
            prefix="main",
            title=row.get("Subcategory / Page", ""),
            url=row.get("URL", ""),
            category=_infer_main_category(row),
            item_type="route",
            source_sheet="01_Main_Routes",
            status=row.get("Status", ""),
            keywords=_split_keywords(row.get("Category", ""), row.get("Description", ""), row.get("Chatbot routing hint", "")),
        )
        if entry:
            entries.append(entry)

    for row in tables.get("02_Data_Stories", []):
        entry = _entry(
            prefix="story",
            title=row.get("Title", ""),
            url=row.get("URL", ""),
            category="data_stories",
            item_type="data_story",
            source_sheet="02_Data_Stories",
            status=row.get("Status", ""),
            keywords=_split_keywords(row.get("Title", ""), row.get("Description", ""), row.get("Keywords / intents", "")),
        )
        if entry:
            entries.append(entry)

    for row in tables.get("03_Results", []):
        title = row.get("Title", "")
        url = row.get("URL", "")
        entry = _entry(
            prefix="result",
            title=title,
            url=url,
            category="results",
            item_type="workspace" if _infer_workspace(url) else "project",
            source_sheet="03_Results",
            status=row.get("Status", ""),
            project=_infer_result_project(title, url),
            workspace=_infer_workspace(url),
            keywords=_split_keywords(row.get("Title", ""), row.get("Type", ""), row.get("Keywords / intents", "")),
        )
        if entry:
            entries.append(entry)

    for row in tables.get("04_Models", []):
        model_name = row.get("Model name", "")
        entry = _entry(
            prefix="model",
            title=model_name,
            url=row.get("URL to use", ""),
            category="models",
            item_type="model",
            source_sheet="04_Models",
            status=row.get("URL status", ""),
            keywords=_split_keywords(model_name, row.get("Full name / Description", ""), row.get("Organisation", "")),
            search_hint=model_name,
        )
        if entry:
            entries.append(entry)

    for row in tables.get("05_App_Library", []):
        title = row.get("Title", "")
        status = row.get("URL status", "")
        url = row.get("URL to use", "")
        verified = _is_verified_direct_url(url, status)
        entry = _entry(
            prefix="app",
            title=title,
            url=url,
            category="application_library",
            item_type="application",
            source_sheet="05_App_Library",
            status=status,
            keywords=_split_keywords(title, row.get("Type / Subcategory", ""), row.get("Source / cataloguer", ""), row.get("Keywords / intents", "")),
            search_hint="" if verified else title,
        )
        if entry:
            entries.append(entry)

    unique: dict[str, LinkCatalogEntry] = {}
    for entry in entries:
        unique.setdefault(entry.id, entry)

    return [asdict(entry) for entry in sorted(unique.values(), key=lambda item: item.id)]


def write_link_catalog(
    source: Path = SOURCE_EXCEL,
    output: Path = DEFAULT_OUTPUT,
) -> list[dict]:
    catalog = build_link_catalog(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IAM PARIS runtime link catalog JSON.")
    parser.add_argument("--source", type=Path, default=SOURCE_EXCEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    catalog = write_link_catalog(args.source, args.output)
    print(f"Wrote {len(catalog)} link catalog entries to {args.output}")


if __name__ == "__main__":
    main()
