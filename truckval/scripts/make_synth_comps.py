"""Generate a synthetic comps table so the pipeline runs before you scrape.

This exists so you can validate the architecture — especially that the range
widens correctly with missing information — on day one. Replace it with real
Ritchie Bros hammer prices as soon as you have them; nothing downstream changes
except the numbers getting real.

    python scripts/make_synth_comps.py --n 12000 --out data/comps.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MAKES = {
    "freightliner": ["m2 106", "cascadia", "business class"],
    "international": ["durastar", "mv607", "lt625"],
    "hino": ["268", "338", "l6"],
    "isuzu": ["nqr", "npr hd", "ftr"],
    "kenworth": ["t270", "t680", "t880"],
    "peterbilt": ["337", "579", "348"],
    "ford": ["f-650", "f-750"],
    "volvo": ["vnl 760", "vnr 300"],
}
BODY_BY_MAKE = {
    "cascadia": "tractor", "lt625": "tractor", "t680": "tractor",
    "579": "tractor", "vnl 760": "tractor", "vnr 300": "tractor",
}
REGIONS = ["salt_belt", "coastal", "arid", "midwest", "southeast"]
CAB_TYPES = ["day_cab", "sleeper", "crew", "standard"]

# Rough new-price anchors by make, in USD. Depreciation is applied on top.
NEW_PRICE = {
    "freightliner": 148_000, "international": 142_000, "hino": 118_000,
    "isuzu": 96_000, "kenworth": 172_000, "peterbilt": 176_000,
    "ford": 105_000, "volvo": 168_000,
}


def build(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    makes = rng.choice(list(MAKES), size=n, p=_make_weights(rng))
    models = np.array([rng.choice(MAKES[m]) for m in makes])
    body = np.array([
        BODY_BY_MAKE.get(mo, rng.choice(
            ["box_truck", "flatbed", "dump", "stake", "service", "reefer"],
            p=[0.42, 0.16, 0.13, 0.11, 0.11, 0.07]))
        for mo in models
    ])

    age = np.clip(rng.gamma(2.6, 2.6, n), 0.5, 26.0)
    annual = np.where(body == "tractor",
                      rng.normal(96_000, 30_000, n),
                      rng.normal(34_000, 16_000, n))
    odo = np.clip(annual * age * rng.lognormal(0, 0.22, n), 3_000, 2_200_000)

    axles = np.where(body == "tractor",
                     rng.choice([2, 3], n, p=[0.35, 0.65]),
                     rng.choice([2, 3], n, p=[0.86, 0.14]))
    box_len = np.where(
        np.isin(body, ["box_truck", "reefer"]),
        rng.choice([14, 16, 18, 20, 22, 24, 26], n,
                   p=[0.07, 0.19, 0.10, 0.14, 0.08, 0.31, 0.11]),
        np.nan,
    )
    gvwr = np.where(body == "tractor", 52_000,
                    np.where(box_len > 22, 33_000, 26_000)) \
        + rng.normal(0, 1_800, n)

    liftgate = ((np.isin(body, ["box_truck", "reefer"]))
                & (rng.random(n) < 0.46)).astype(float)
    reefer = (body == "reefer").astype(float)
    crane = ((body == "service") & (rng.random(n) < 0.35)).astype(float)
    fifth = (body == "tractor").astype(float)
    pto = ((np.isin(body, ["dump", "service"])) & (rng.random(n) < 0.7)).astype(float)

    region = rng.choice(REGIONS, n, p=[0.24, 0.18, 0.16, 0.24, 0.18])
    cab = np.where(body == "tractor",
                   rng.choice(["day_cab", "sleeper"], n, p=[0.42, 0.58]),
                   rng.choice(["standard", "crew"], n, p=[0.84, 0.16]))

    base = np.array([NEW_PRICE[m] for m in makes], dtype=float)
    # Declining-balance depreciation plus a distance penalty.
    price = base * (0.86 ** age) * np.exp(-odo / 1_150_000)
    price *= 1 + 0.030 * liftgate + 0.16 * reefer + 0.11 * crane + 0.02 * pto
    price *= np.where(region == "arid", 1.04,
                      np.where(region == "salt_belt", 0.93, 1.0))
    price *= np.where(np.isnan(box_len), 1.0, 1 + (box_len - 20) * 0.006)
    price *= np.where(cab == "sleeper", 1.07, 1.0)
    price *= rng.lognormal(0, 0.19, n)  # irreducible market spread
    price = np.clip(price, 2_400, None)

    return pd.DataFrame({
        "sale_price": price.round(0),
        "age_years": age,
        "log_odometer_km": np.log1p(odo),
        "odometer_km": odo,
        "axle_count": axles.astype(float),
        "box_length_ft": box_len,
        "gvwr_lb": gvwr,
        "make": makes,
        "model": models,
        "body_type": body,
        "cab_type": cab,
        "region": region,
        "liftgate": liftgate,
        "reefer_unit": reefer,
        "crane": crane,
        "fifth_wheel": fifth,
        "pto": pto,
    })


def _make_weights(rng: np.random.Generator) -> np.ndarray:
    w = np.array([0.21, 0.17, 0.13, 0.14, 0.11, 0.10, 0.08, 0.06])
    return w / w.sum()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12_000)
    ap.add_argument("--out", default="data/comps.parquet")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    df = build(a.n, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.out.endswith(".parquet"):
        df.to_parquet(a.out, index=False)
    else:
        df.to_csv(a.out, index=False)
    print(f"wrote {len(df):,} rows to {a.out}")
    print(df[["sale_price", "age_years", "odometer_km"]].describe().round(0))
