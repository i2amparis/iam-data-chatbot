from dataclasses import dataclass
import math
import re
from typing import Optional


LATEST_YEAR_SENTINEL = -1


def is_finite_numeric_value(value: object) -> bool:
    """True for usable numeric observations, excluding null, bool and NaN/inf."""
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def is_latest_year_filter(start_year: Optional[int], end_year: Optional[int]) -> bool:
    return start_year == LATEST_YEAR_SENTINEL and end_year == LATEST_YEAR_SENTINEL


# Accept requested bounds through the 2200s even when the loaded catalogue ends
# earlier.  The selector will naturally return only reported years; rejecting a
# valid four-digit upper bound instead leaves words such as ``through`` in the
# variable matcher and can derail an otherwise grounded plot request.
_YEAR = r"(?:19|20|21|22)\d{2}"

_SPOKEN_DECADES = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SPOKEN_ONES = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}


def _normalize_spoken_years(text: str) -> str:
    """Normalize common spoken climate-horizon years to four digits.

    This intentionally accepts only the unambiguous 19xx/20xx forms used for
    projection years; it is not a general natural-language number parser.
    """
    value = str(text or "").lower()
    value = re.sub(r"\btwenty[\s-]+one[\s-]+hundred\b", "2100", value)
    decades = "|".join(_SPOKEN_DECADES)
    ones = "|".join(_SPOKEN_ONES)

    def replace(match: re.Match[str]) -> str:
        century = 1900 if match.group(1) == "nineteen" else 2000
        return str(
            century
            + _SPOKEN_DECADES[match.group(2)]
            + _SPOKEN_ONES.get(match.group(3) or "", 0)
        )

    return re.sub(
        rf"\b(nineteen|twenty)[\s-]+({decades})(?:[\s-]+({ones}))?\b",
        replace,
        value,
    )


@dataclass(frozen=True)
class YearFilter:
    """A parsed year constraint with explicit open-ended semantics.

    ``(None, None)`` used to mean both "no year was mentioned" and an
    open-ended filter whose absent side must clear a carried bound.  The
    ``explicit`` flag keeps those cases distinct during follow-up scope
    mutation.
    """

    start_year: Optional[int] = None
    end_year: Optional[int] = None
    explicit: bool = False
    operator: str = "none"

    @property
    def is_latest(self) -> bool:
        return is_latest_year_filter(self.start_year, self.end_year)

    def apply(self, scope: dict) -> dict:
        """Return ``scope`` with this filter applied as a replacement."""
        updated = dict(scope or {})
        if not self.explicit:
            return updated
        # Clearing both keys first is important for mutations such as
        # ``2030`` -> ``until 2050`` and ``2030-2050`` -> ``after 2030``.
        updated.pop("start_year", None)
        updated.pop("end_year", None)
        if self.start_year is not None:
            updated["start_year"] = self.start_year
        if self.end_year is not None:
            updated["end_year"] = self.end_year
        return updated

    def render(self) -> str:
        """Render a normalized phrase without changing its bounds."""
        if not self.explicit:
            return ""
        if self.is_latest:
            return "at the latest available year"
        if self.operator == "after" and self.start_year is not None:
            return f"after {self.start_year - 1}"
        if self.start_year is None and self.end_year is not None:
            return f"until {self.end_year}"
        if self.start_year is not None and self.end_year is None:
            return f"from {self.start_year}"
        if self.start_year == self.end_year:
            return f"in {self.start_year}"
        if self.start_year is not None and self.end_year is not None:
            return f"from {self.start_year} to {self.end_year}"
        return ""


def extract_year_filter(text: str) -> YearFilter:
    """Parse a year expression and retain whether it was explicitly present."""
    value = _normalize_spoken_years(text)

    if re.search(r"\b(latest|most recent|newest)\b", value):
        return YearFilter(
            LATEST_YEAR_SENTINEL,
            LATEST_YEAR_SENTINEL,
            explicit=True,
            operator="latest",
        )

    match = re.search(
        rf"\b(?:from\s+|between\s+)?({_YEAR})\s*"
        rf"(?:-|–|—|to|through|until|and|&|,)\s*({_YEAR})\b",
        value,
    )
    if match:
        first = int(match.group(1))
        second = int(match.group(2))
        return YearFilter(min(first, second), max(first, second), True, "range")

    match = re.search(rf"\b(?:by|to|until|up to|through|before)\s+({_YEAR})\b", value)
    if match:
        return YearFilter(None, int(match.group(1)), True, "until")

    match = re.search(rf"\bafter\s+({_YEAR})\b", value)
    if match:
        return YearFilter(int(match.group(1)) + 1, None, True, "after")

    match = re.search(rf"\b(?:from|since)\s+({_YEAR})\b", value)
    if match:
        return YearFilter(int(match.group(1)), None, True, "from")

    match = re.search(rf"\b(?:in|for|at|around|only|just)\s+({_YEAR})\b", value)
    if match:
        year = int(match.group(1))
        return YearFilter(year, year, True, "exact")

    match = re.search(rf"\b({_YEAR})\b", value)
    if match:
        year = int(match.group(1))
        return YearFilter(year, year, True, "exact")

    return YearFilter()


def extract_year_range(text: str) -> tuple[Optional[int], Optional[int]]:
    parsed = extract_year_filter(text)
    return parsed.start_year, parsed.end_year


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
