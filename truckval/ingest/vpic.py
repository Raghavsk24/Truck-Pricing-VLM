"""Identity spine. Free, no key, and the only layer here with no legal grey area.

vPIC is populated from manufacturer 565 submittals, covers trucks, buses,
trailers and incomplete vehicles sold in the US since 1981, and returns make,
model, model year and GVWR among 100+ fields.

Two ways in:

  API      https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/{VIN}?format=json
           Free, no registration, decodes up to 50 VINs per call. NHTSA applies
           automated rate control, so this is fine for development and wrong
           for a hot path.

  Standalone DB
           NHTSA publishes the whole catalog for download as a SQL Server
           backup. Restore it, export the pattern tables, and your VIN decode
           becomes a local table lookup with no network dependency inside the
           perception loop. Do this before you have users.
           https://vpic.nhtsa.dot.gov/api/  ->  "Standalone vPIC Databases"

This module handles the API path and the local-cache path. The SQL Server
restore is a one-off you do by hand; `load_local_table` reads whatever you
export from it.
"""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd

BATCH_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvaluesbatch/"
KEEP = [
    "VIN", "Make", "Model", "ModelYear", "Series", "Trim", "BodyClass",
    "VehicleType", "GVWR", "GrossVehicleWeightRatingFrom", "Axles",
    "EngineModel", "DisplacementL", "FuelTypePrimary", "BrakeSystemType",
    "PlantCity", "PlantCountry", "ErrorCode", "ErrorText",
]


def decode_batch(vins: Iterable[str], pause: float = 1.0) -> pd.DataFrame:
    """Up to 50 VINs per request. Be polite: NHTSA rate-limits."""
    vins = list(vins)
    frames = []
    for i in range(0, len(vins), 50):
        chunk = vins[i:i + 50]
        data = urllib.parse.urlencode({
            "DATA": ";".join(chunk), "format": "json"
        }).encode()
        req = urllib.request.Request(BATCH_URL, data=data)
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = json.loads(r.read().decode())
        frames.append(pd.DataFrame(payload.get("Results", [])))
        time.sleep(pause)
    if not frames:
        return pd.DataFrame(columns=KEEP)
    df = pd.concat(frames, ignore_index=True)
    return df[[c for c in KEEP if c in df.columns]]


def build_cache(vins: Iterable[str], db_path: str | Path = "data/vpic.sqlite") -> int:
    """Decode once, store forever. VIN attributes never change."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    existing: set[str] = set()
    try:
        existing = set(pd.read_sql("SELECT VIN FROM vpic", con)["VIN"])
    except Exception:
        pass
    todo = [v for v in vins if v not in existing]
    if todo:
        decode_batch(todo).to_sql("vpic", con, if_exists="append", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_vin ON vpic(VIN)")
        con.commit()
    con.close()
    return len(todo)


def lookup(vin: str, db_path: str | Path = "data/vpic.sqlite") -> Optional[dict]:
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql("SELECT * FROM vpic WHERE VIN = ?", con, params=(vin.upper(),))
    finally:
        con.close()
    return df.iloc[0].to_dict() if len(df) else None


def load_local_table(path: str | Path) -> pd.DataFrame:
    """Read an export from the restored standalone vPIC database."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
