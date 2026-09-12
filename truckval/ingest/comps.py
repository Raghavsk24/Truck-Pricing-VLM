"""Turn whatever you can source into the one table pricing.py expects.

READ THIS BEFORE WIRING A SCRAPER
---------------------------------
Ritchie Bros and Copart both restrict automated access in their terms. The
third-party actors that scrape them operate in a grey zone — at least one
openly states it is not affiliated with or endorsed by the site. That exposure
is survivable for a class project and is not survivable for a funded company
with a diligence process.

The order to do this in:
  1. Prototype against the free UI. Ritchie Bros gives free access to 1M+ past
     auction results from the last 24 months, searchable by type, model year,
     location and hours, returning low/median/high per category. Hand-collect
     a few hundred rows. It is enough to fit a first model and it costs you
     nothing but an afternoon.
  2. In parallel, request a MarketCheck quote. Their Heavy Equipments API is
     the licensed path; they flag it themselves as having narrower coverage
     than their Cars API, so find out what you actually get before you depend
     on it.
  3. Only build automated collection against sources whose terms permit it, or
     under a written agreement.

WHY AUCTION PRICES AND NOT LISTING PRICES
-----------------------------------------
Ritchie Bros auctions run unreserved — no minimum bid, no reserve — so the
hammer price is a real clearing price. Classified ask prices are aspirational
and will bias a quantile model high at every level. If you mix both sources,
carry a `price_kind` column and never train on them interchangeably.

TARGET SCHEMA
-------------
One row per SOLD vehicle. `sale_price` is the realized price, not an ask.
Columns must match truckval.features.FEATURES plus the target.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from truckval.features import FEATURES

REQUIRED = ["sale_price", *FEATURES]

MAKE_ALIASES = {
    "freightliner trucks": "freightliner",
    "intl": "international",
    "international harvester": "international",
    "mack trucks": "mack",
    "gmc truck": "gmc",
    "hino motors": "hino",
    "isuzu commercial": "isuzu",
}

BODY_KEYWORDS = [
    (r"\breefer|refrigerat", "reefer"),
    (r"\bbox\b|van body|dry van body|cube", "box_truck"),
    (r"\bflat ?bed\b|\bstake\b.*flat", "flatbed"),
    (r"\bdump\b", "dump"),
    (r"\bstake\b|\bplatform\b", "stake"),
    (r"\btractor\b|\bsleeper\b|day ?cab.*tractor", "tractor"),
    (r"\bservice\b|\bmechanic\b|\butility body\b", "service"),
    (r"\btank(er)?\b", "tanker"),
    (r"chassis|cab ?& ?chassis", "chassis_cab"),
]

MILES_TO_KM = 1.609344


def normalize_make(s: object) -> str:
    v = str(s or "").strip().lower()
    return MAKE_ALIASES.get(v, v)


def infer_body_type(text: object) -> str:
    t = str(text or "").lower()
    for pattern, label in BODY_KEYWORDS:
        if re.search(pattern, t):
            return label
    return "other"


def parse_box_length(text: object) -> float:
    """Listing titles carry this constantly: '24ft box', '26' van body'."""
    t = str(text or "").lower()
    m = re.search(r"(\d{2})\s*(?:'|ft|foot|feet)\b", t)
    if m:
        val = float(m.group(1))
        return val if 10 <= val <= 60 else np.nan
    return np.nan


def parse_distance(value: object, unit_hint: str = "") -> float:
    """Return km. Auction sheets mix miles and km without warning."""
    t = str(value or "").replace(",", "").lower()
    m = re.search(r"([\d.]+)", t)
    if not m:
        return np.nan
    n = float(m.group(1))
    if "mi" in t or "mi" in unit_hint.lower():
        n *= MILES_TO_KM
    return n


def normalize(raw: pd.DataFrame, current_year: int = 2026) -> pd.DataFrame:
    """Map a scraped/exported frame onto the training schema.

    Expects at minimum: sale_price, year, make, model, and some free text
    (title/description) to mine for body type and box length.
    """
    df = pd.DataFrame(index=raw.index)
    text = raw.get("title", pd.Series("", index=raw.index)).astype(str) + " " + \
        raw.get("description", pd.Series("", index=raw.index)).astype(str)

    df["sale_price"] = pd.to_numeric(raw["sale_price"], errors="coerce")
    year = pd.to_numeric(raw.get("year"), errors="coerce")
    df["age_years"] = (current_year - year).clip(lower=0.5)

    odo_km = raw.get("odometer_km")
    if odo_km is None:
        odo_km = raw.get("odometer", pd.Series(np.nan, index=raw.index)).map(
            lambda v: parse_distance(v, str(raw.get("odometer_unit", "")))
        )
    df["odometer_km"] = pd.to_numeric(odo_km, errors="coerce")
    df["log_odometer_km"] = np.log1p(df["odometer_km"])

    df["make"] = raw.get("make", "").map(normalize_make)
    df["model"] = raw.get("model", "").astype(str).str.strip().str.lower()
    df["body_type"] = raw.get("body_type", text).map(infer_body_type)
    df["cab_type"] = raw.get("cab_type", "unknown").astype(str).str.lower()
    df["region"] = raw.get("region", raw.get("state", "unknown")).astype(str).str.lower()

    df["axle_count"] = pd.to_numeric(raw.get("axles"), errors="coerce")
    df["box_length_ft"] = pd.to_numeric(raw.get("box_length_ft"), errors="coerce")
    df["box_length_ft"] = df["box_length_ft"].fillna(text.map(parse_box_length))
    df["gvwr_lb"] = pd.to_numeric(raw.get("gvwr_lb"), errors="coerce")

    lower = text.str.lower()
    df["liftgate"] = lower.str.contains("lift ?gate|tuck.?under").astype(float)
    df["reefer_unit"] = lower.str.contains("reefer|thermo ?king|carrier tran").astype(float)
    df["crane"] = lower.str.contains("crane|boom|knuckle").astype(float)
    df["fifth_wheel"] = lower.str.contains("fifth wheel|5th wheel").astype(float)
    df["pto"] = lower.str.contains(r"\bpto\b|power take.?off").astype(float)

    df["price_kind"] = raw.get("price_kind", "hammer")
    return df


def validate(df: pd.DataFrame, min_rows: int = 500) -> list[str]:
    """Fail loudly before you train on garbage."""
    problems = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    if len(df) < min_rows:
        problems.append(f"only {len(df)} rows; want at least {min_rows}")
    if "sale_price" in df:
        bad = df["sale_price"].isna().sum()
        if bad:
            problems.append(f"{bad} rows have no sale price")
        if (df["sale_price"] <= 0).any():
            problems.append("non-positive sale prices present")
    if "price_kind" in df and df["price_kind"].nunique() > 1:
        problems.append(
            "mixed price kinds (ask vs hammer) — split these, do not train "
            "on them interchangeably"
        )
    for c in ("make", "model"):
        if c in df and df[c].isna().mean() > 0.05:
            problems.append(f"{c} missing on >5% of rows")
    return problems


def save(df: pd.DataFrame, path: str | Path = "data/comps.parquet") -> None:
    problems = validate(df)
    if problems:
        raise ValueError("comps table failed validation:\n  " + "\n  ".join(problems))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
