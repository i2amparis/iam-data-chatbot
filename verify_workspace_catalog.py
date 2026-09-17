"""Verify local workspace mappings against the public IAM PARIS results page.

Usage: python verify_workspace_catalog.py /path/to/downloaded-results.html
The page's server-rendered data is parsed as text, never executed.
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def public_workspaces(html):
    pattern = re.compile(
        r'\{id:(\d+),title:("(?:[^"\\]|\\.)*"),description:(?:"(?:[^"\\]|\\.)*"|null)'
        r',slug:("(?:[^"\\]|\\.)*"),internal_name:("(?:[^"\\]|\\.)*"),'
        r'project_id:\[\{projects_id:\{slug:("(?:[^"\\]|\\.)*")',
        re.DOTALL,
    )
    entries = {}
    for identity, title, slug, code, project in pattern.findall(html):
        title, slug, code, project = map(json.loads, (title, slug, code, project))
        base = f"https://iamparis.eu/results/{project}/{slug}"
        entries[code] = {
            "code": code, "title": title,
            "explorer_url": base + "/graphs",
            "explainer_url": base + "/policy_questions",
        }
    if not entries:
        raise ValueError("No public workspace metadata found; page format may have changed")
    return entries


if __name__ == "__main__":
    public = public_workspaces(Path(sys.argv[1]).read_text())
    local = json.loads(Path("config/workspace_catalog.json").read_text())["workspaces"]
    differences = []
    for entry in local:
        expected = public.get(entry["code"])
        if expected is None:
            differences.append({"code": entry["code"], "error": "not present in public catalogue"})
        elif any(entry.get(key) != value for key, value in expected.items()):
            differences.append(expected)
    print(json.dumps({"public_count": len(public), "local_count": len(local), "corrections": differences}, indent=2))
    if "--check-links" in sys.argv:
        def check(url):
            try:
                response = requests.get(url, timeout=30)
                if response.status_code != 200 or response.url.rstrip("/") != url.rstrip("/"):
                    return {"url": url, "status": response.status_code, "destination": response.url}
            except requests.RequestException as error:
                return {"url": url, "error": str(error)}
            return None

        urls = [entry[key] for entry in local for key in ("explorer_url", "explainer_url")]
        with ThreadPoolExecutor(max_workers=4) as executor:
            errors = [result for result in executor.map(check, urls) if result]
        print(json.dumps({"checked_links": len(urls), "link_errors": errors}, indent=2))
        differences.extend(errors)
    sys.exit(bool(differences))
