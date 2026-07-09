import re
from typing import Optional


LATEST_YEAR_SENTINEL = -1


def is_latest_year_filter(start_year: Optional[int], end_year: Optional[int]) -> bool:
    return start_year == LATEST_YEAR_SENTINEL and end_year == LATEST_YEAR_SENTINEL


# IAM PARIS timeseries run to 2100, so a valid year is 1900-2199. The previous
# ``19\d{2}|20\d{2}`` pattern silently excluded 2100 (it starts with "21").
_YEAR = r"(?:19|20|21)\d{2}"


def extract_year_range(text: str) -> tuple[Optional[int], Optional[int]]:
    value = (text or "").lower()

    if re.search(r"\b(latest|most recent|newest)\b", value):
        return LATEST_YEAR_SENTINEL, LATEST_YEAR_SENTINEL

    # Two years joined by a range/list connective ("2030 to 2060", "between 2030
    # and 2060", "in 2030 and 2050", "2030-2050") -> min..max span.
    match = re.search(rf"\b({_YEAR})\s*(?:-|–|—|to|and|&|,)\s*({_YEAR})\b", value)
    if match:
        first = int(match.group(1))
        second = int(match.group(2))
        return min(first, second), max(first, second)

    match = re.search(rf"\b(?:by|until|up to|through|before)\s+({_YEAR})\b", value)
    if match:
        return None, int(match.group(1))

    match = re.search(rf"\bafter\s+({_YEAR})\b", value)
    if match:
        return int(match.group(1)) + 1, None

    match = re.search(rf"\b(?:from|since)\s+({_YEAR})\b", value)
    if match:
        return int(match.group(1)), None

    match = re.search(rf"\b(?:in|for|at|around)\s+({_YEAR})\b", value)
    if match:
        year = int(match.group(1))
        return year, year

    match = re.search(rf"\b({_YEAR})\b", value)
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
