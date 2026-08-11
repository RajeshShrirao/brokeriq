"""NAICS classification from free-text signals.

Uses a curated subset of the 2022 NAICS (public domain, U.S. Census Bureau)
focused on the industries commercial insurance brokers actually see.
"""

import csv
import logging
from functools import lru_cache
from importlib import resources

logger = logging.getLogger(__name__)

_DATASET_NAME = "naics.csv"


@lru_cache(maxsize=1)
def _load_codes() -> list[dict]:
    rows: list[dict] = []
    with resources.files("brokeriq.data").joinpath(_DATASET_NAME).open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({"code": row["code"], "label": row["label"], "keywords": row["keywords"]})
    logger.info("loaded %d naics codes from %s", len(rows), _DATASET_NAME)
    return rows


def lookup_naics(company_name: str, industry_hint: str | None = None) -> dict | None:
    """Best-effort NAICS guess from the company name + optional industry hint.

    Returns {"code", "label", "matches"} or None when nothing is plausible.
    """
    query = f"{company_name} {industry_hint or ''}".lower()
    tokens = {t for t in query.replace(",", " ").split() if len(t) > 2}

    best: dict | None = None
    best_hits = 0
    for row in _load_codes():
        keyword_set = {k.strip().lower() for k in row["keywords"].split("|") if k.strip()}
        hits = len(tokens & keyword_set)
        # whole-word bonus for distinctive tokens
        for tok in tokens:
            if tok in keyword_set:
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best = {"code": row["code"], "label": row["label"], "matches": keyword_set & tokens}

    if best is None or best_hits == 0:
        return None
    return best
