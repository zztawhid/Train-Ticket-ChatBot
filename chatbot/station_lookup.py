from __future__ import annotations

"""
station_lookup.py
-----------------
Loads UK station names and CRS codes from a CSV file and provides
fuzzy-matching so that user input like "norwich", "kings cross", or
even slight typos resolve to the correct official CRS code.

Uses rapidfuzz for high-performance fuzzy string matching.
"""

import os
import csv
from rapidfuzz import process, fuzz

# ---------------------------------------------------------------------------
# Path to the CSV file shipped with the project
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_CSV_PATH = os.path.join(_DATA_DIR, "StationNameAndCode.csv")

# ---------------------------------------------------------------------------
# Internal storage: populated once at import time
# ---------------------------------------------------------------------------
_station_name_to_crs: dict[str, str] = {}   # normalised name  -> CRS
_crs_to_station_name: dict[str, str] = {}   # CRS              -> display name
_lookup_names: list[str] = []                # list for rapidfuzz search


def _normalise(name: str) -> str:
    """Lower-case and strip common suffixes that are not useful for matching."""
    name = name.strip().lower()
    # Remove punctuation that hurts fuzzy matching (e.g. apostrophes)
    name = name.replace("'", "").replace("'", "")
    # Remove suffixes like "(bus)", "platform 2", "metrolink", "lt" etc.
    for suffix in ["(bus)", "metrolink", " lt", " el", " br"]:
        name = name.replace(suffix, "")
    return name.strip()


def _load_stations() -> None:
    """Read the CSV once and populate the module-level dictionaries."""
    global _station_name_to_crs, _crs_to_station_name, _lookup_names

    if _station_name_to_crs:
        return  # already loaded

    with open(_CSV_PATH, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_name = row["NAME"].strip()
            crs = row["CRS"].strip().upper()

            # Skip entries with non-standard CRS codes (internal / engineering)
            if crs.startswith("X") or crs.startswith("Z"):
                continue

            norm = _normalise(raw_name)

            # Keep the first (usually most common) entry per normalised name
            if norm not in _station_name_to_crs:
                _station_name_to_crs[norm] = crs

            # Keep a display-friendly version keyed by CRS
            if crs not in _crs_to_station_name:
                _crs_to_station_name[crs] = raw_name.title()

    _lookup_names = list(_station_name_to_crs.keys())


# Load on first import
_load_stations()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup_station(query: str, threshold: int = 70, top_n: int = 3):
    """
    Attempt to resolve a user-provided station name to a CRS code.

    Returns
    -------
    tuple (crs_code: str | None, candidates: list[tuple[str, str, int]])
        - crs_code  : the best-match CRS code if confidence >= threshold,
                       else None.
        - candidates : up to *top_n* (display_name, crs, score) tuples so
                       the chatbot can ask the user to clarify if needed.
    """
    norm_query = _normalise(query)

    # 1. Try exact match first (fastest path)
    if norm_query in _station_name_to_crs:
        crs = _station_name_to_crs[norm_query]
        display = _crs_to_station_name.get(crs, norm_query.title())
        return crs, [(display, crs, 100)]

    # 2. Fuzzy match using rapidfuzz (WRatio handles partial and token matching)
    results = process.extract(
        norm_query, _lookup_names, scorer=fuzz.WRatio, limit=top_n
    )

    if not results:
        return None, []

    best_name, best_score, _ = results[0]

    # Build candidates with a tiebreaker: prefer names that contain
    # more of the user's query words (e.g. "london liverpool street"
    # should prefer "liverpool street london" over "liverpool street el").
    query_words = set(norm_query.split())
    scored = []
    for name, score, _ in results:
        crs = _station_name_to_crs[name]
        display = _crs_to_station_name.get(crs, name.title())
        # Count how many query words appear in the candidate name
        name_words = set(name.split())
        overlap = len(query_words & name_words)
        scored.append((display, crs, int(score), overlap))

    # Sort by fuzzy score DESC, then by word overlap DESC
    scored.sort(key=lambda x: (x[2], x[3]), reverse=True)
    candidates = [(d, c, s) for d, c, s, _ in scored]

    if scored[0][2] >= threshold:
        return scored[0][1], candidates
    else:
        return None, candidates


def crs_to_name(crs: str) -> str:
    """Return the display name for a CRS code, or the code itself."""
    return _crs_to_station_name.get(crs.upper(), crs.upper())


def get_all_station_names() -> list[str]:
    """Return all normalised station names (useful for NLP phrase matching)."""
    return list(_lookup_names)
