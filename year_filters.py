import re
from typing import Optional


LATEST_YEAR_SENTINEL = -1


def is_latest_year_filter(start_year: Optional[int], end_year: Optional[int]) -> bool:
    return start_year == LATEST_YEAR_SENTINEL and end_year == LATEST_YEAR_SENTINEL


def extract_year_range(text: str) -> tuple[Optional[int], Optional[int]]:
    value = (text or "").lower()

    if re.search(r"\b(latest|most recent|newest)\b", value):
        return LATEST_YEAR_SENTINEL, LATEST_YEAR_SENTINEL

    match = re.search(r"\b(19\d{2}|20\d{2})\s*(?:-|to|–|—)\s*(19\d{2}|20\d{2})\b", value)
    if match:
        first = int(match.group(1))
        second = int(match.group(2))
        return min(first, second), max(first, second)

    match = re.search(r"\b(?:by|until|up to|through|before)\s+(19\d{2}|20\d{2})\b", value)
    if match:
        return None, int(match.group(1))

    match = re.search(r"\bafter\s+(19\d{2}|20\d{2})\b", value)
    if match:
        return int(match.group(1)) + 1, None

    match = re.search(r"\b(?:from|since)\s+(19\d{2}|20\d{2})\b", value)
    if match:
        return int(match.group(1)), None

    match = re.search(r"\b(?:in|for)\s+(19\d{2}|20\d{2})\b", value)
    if match:
        year = int(match.group(1))
        return year, year

    match = re.search(r"\b(19\d{2}|20\d{2})\b", value)
    if match:
        year = int(match.group(1))
        return year, year

    return None, None


def select_years(
    years: list[str],
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> list[str]:
    numeric_years = []
    for year in years:
        try:
            numeric_years.append((int(year), str(year)))
        except Exception:
            continue

    numeric_years = sorted(set(numeric_years), key=lambda item: item[0])
    if not numeric_years:
        return []

    if is_latest_year_filter(start_year, end_year):
        return [numeric_years[-1][1]]

    selected = []
    for year_int, year_text in numeric_years:
        if start_year is not None and year_int < start_year:
            continue
        if end_year is not None and year_int > end_year:
            continue
        selected.append(year_text)
    return selected
