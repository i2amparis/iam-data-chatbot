# utils_query.py
import re
from typing import List, Dict, Any, Optional, Tuple
from yaml_support import pick_region_and_variable

YEAR_RE = re.compile(r"\b(19|20|21)\d{2}\b")

def extract_years(text: str) -> List[int]:
    seen, out = set(), []
    for m in YEAR_RE.finditer(text or ""):
        y = int(m.group(0))
        if y not in seen:
            seen.add(y); out.append(y)
    return out

def extract_range(text: str) -> Optional[Tuple[int, int]]:
    t = (text or "").lower()
    m = re.search(r"\b(19|20|21)\d{2}\s*[-–]\s*(19|20|21)\d{2}\b", t)
    if m:
        a, b = int(m.group(0)[:4]), int(m.group(0)[-4:])
        return (min(a,b), max(a,b))
    m2 = re.search(r"from\s+(19|20|21)\d{2}\s+to\s+(19|20|21)\d{2}", t)
    if m2:
        years = [int(s) for s in YEAR_RE.findall(m2.group(0))]
        if len(years) >= 2:
            return (min(years[0], years[-1]), max(years[0], years[-1]))
    return None

def resolve_query(text: str, aliases: Dict[str, Dict[str, str]], country_to_region: Dict[str, str]) -> Dict[str, Any]:
    picks = pick_region_and_variable(text, aliases, country_to_region)
    ql = (text or "").lower()

    # capture explicit country token if present
    country_token = None
    for country in sorted(country_to_region.keys(), key=len, reverse=True):
        if country in ql:
            country_token = country
            break

    yr_range = extract_range(text)
    years = extract_years(text)

    info: Dict[str, Any] = {
        "variable": picks.get("variable"),
        "region": picks.get("region"),
        "country_token": country_token
    }
    if yr_range:
        info["year_range"] = {"from": yr_range[0], "to": yr_range[1]}
    elif years:
        info["years"] = years
    return info

def compose_explanation(info: Dict[str, Any]) -> str:
    var = info.get("variable")
    reg = info.get("region")
    country_token = info.get("country_token")
    years = info.get("years")
    yr_range = info.get("year_range")

    if not var and not reg:
        return ("I couldn’t recognize a variable or region. "
                "Try: “show electricity generation in Greece for 2030”.")

    var_nice = var.replace("|", " ") if var else "the requested variable"
    when = ""
    if years:
        when = f"in **{years[0]}**" if len(years) == 1 else f"in **{', '.join(map(str, years))}**"
    elif yr_range:
        when = f"from **{yr_range['from']}** to **{yr_range['to']}**"

    if country_token:
        cdisp = country_token.capitalize()
        msg = (f"I don’t have country-level series for **{cdisp}** in this dataset. "
               f"{cdisp} belongs to **{reg}**, the smallest reporting region available here, "
               f"so I’ll show **{var_nice}** for **{reg}**")
        return msg + (f" {when}." if when else ".")
    else:
        msg = f"Showing **{var_nice}** for **{reg}**"
        return msg + (f" {when}." if when else ".")
