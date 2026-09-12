"""Turn evidence (plus one draw of the unknowns) into a feature row.

Kept deliberately boring. The interesting behaviour lives in uncertainty.py;
this module just has to be deterministic and identical between training and
inference, which is the usual place these systems quietly break.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from .schema import Evidence

NUMERIC = [
    "age_years",
    "log_odometer_km",
    "axle_count",
    "box_length_ft",
    "gvwr_lb",
]
CATEGORICAL = ["make", "model", "body_type", "cab_type", "region"]
BOOLEAN = ["liftgate", "reefer_unit", "crane", "fifth_wheel", "pto"]

FEATURES = NUMERIC + CATEGORICAL + BOOLEAN
CURRENT_YEAR = 2026


def _pick(ev: Evidence, path: str, draw: dict[str, Any]) -> Any:
    """Observed value wins; otherwise take this draw's sampled value."""
    if path in draw:
        return draw[path]
    return ev.get(path)


def build_row(
    ev: Evidence,
    draw: Optional[dict[str, Any]] = None,
    current_year: int = CURRENT_YEAR,
) -> dict[str, Any]:
    draw = draw or {}

    year = _pick(ev, "identity.year", draw)
    odo = _pick(ev, "condition.odometer_km", draw)
    box = _pick(ev, "configuration.box_length_ft", draw)

    row: dict[str, Any] = {
        "age_years": float(current_year - int(year)) if year else np.nan,
        "log_odometer_km": float(np.log1p(float(odo))) if odo else np.nan,
        "axle_count": _pick(ev, "configuration.axle_count", draw),
        "box_length_ft": float(box) if box else np.nan,
        "gvwr_lb": _pick(ev, "configuration.gvwr_lb", draw),
        "make": _norm(_pick(ev, "identity.make", draw)),
        "model": _norm(_pick(ev, "identity.model", draw)),
        "body_type": _norm(_pick(ev, "configuration.body_type", draw)),
        "cab_type": _norm(_pick(ev, "configuration.cab_type", draw)),
        "region": _norm(ev.region),
    }
    for f in BOOLEAN:
        val = _pick(ev, f"equipment.{f}", draw)
        row[f] = float(val) if val is not None else 0.0
    return row


def _norm(v: Any) -> str:
    return str(v).strip().lower() if v not in (None, "") else "unknown"


def rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=FEATURES)
    for c in NUMERIC + BOOLEAN:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


UNKNOWN = "unknown"


def align_categories(df: pd.DataFrame, reference: dict[str, list]) -> pd.DataFrame:
    """LightGBM needs the same category universe at train and predict time.

    A make or model never seen in training is mapped to "unknown" rather than
    left to become NaN. Semantically that is what it is, and the previous
    silent-NaN behaviour meant an unrecognised truck got a missing-value split
    instead of the unknown-category split the model was actually trained on.
    """
    df = df.copy()
    for c, cats in reference.items():
        allowed = list(cats)
        if UNKNOWN not in allowed:
            allowed.append(UNKNOWN)
        vals = df[c].astype(str).where(df[c].astype(str).isin(allowed), UNKNOWN)
        df[c] = pd.Categorical(vals, categories=allowed)
    return df
