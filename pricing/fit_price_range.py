"""Fit a type + brand + era + condition price RANGE from scraped listings.

Point estimate
--------------
Hierarchical shrinkage down four levels, each pulled toward its parent:

    global  ->  type  ->  (type, brand)  ->  (type, brand, era)

    mu_type  = shrink(type rows,            global, k_type)
    mu_brand = shrink((type,brand) rows,    mu_type,  k_brand)
    mu_era   = shrink((type,brand,era) rows, mu_brand, k_era)

    center = exp( mu_era + condition_scale * beta * (score - score_bar) )

shrink(rows, parent, k) = w*mean(rows) + (1-w)*parent, with w = n/(n+k).
A brand with one listing barely moves off its type baseline; a brand with
sixty listings is trusted almost fully. That is what lets the full 12-brand
list be used instead of a five-brands-plus-OTHER bucket.

Two-stage tuning
----------------
Stage A tunes k_type/k_brand/k_era on ALL priced listings (~358), since type,
brand and era all come from scrape metadata and need no VLM labels.
Stage B fixes the cells and tunes only the condition term on the VLM-labeled
rows, which are far fewer. Tuning everything on the labeled rows would just
overfit them.

Range
-----
Adaptive: error_bound is a percentile of population leave-one-out APE within
the truck's own (type, era) group, falling back to (type) then global. Tight
groups get tight ranges; volatile groups get wide ones.

Usage:
    python pricing/fit_price_range.py
    python pricing/fit_price_range.py --no-tune
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATASET = REPO / "image_scraper" / "truckpaper_scraped_images_dataset"


def _resolve_listings() -> Path:
    """The listings file has lived both inside and beside the dataset folder."""
    candidates = (
        DATASET / "truckpaper_scraped_listings.json",
        REPO / "image_scraper" / "truckpaper_scraped_listings.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


LISTINGS_JSON = _resolve_listings()
PROGRESS_JSON = DATASET / ".state" / "progress.json"
SEARCH_CACHE_JSON = DATASET / ".state" / "search_cache.json"
LABELS_PATH = ROOT / "labels.jsonl"
MODEL_OUT = ROOT / "price_range_model.json"
PLOT_OUT = ROOT / "price_range_plot.png"

TRUCK_TYPES = ("day_cab", "sleeper", "dump")
OTHER_TYPE = "other"
UNKNOWN_BRAND = "UNKNOWN"
ERAS = ("pre_2010", "2010_2015", "2016_2020", "2021_plus")
UNKNOWN_ERA = "unknown"

# How much probability mass to leave on the era bucket the VLM named, by the
# confidence it stated. The remainder is split over the ADJACENT buckets.
#
# Measured on the labeled sample: a "high" era call was right 5/5 times,
# "medium" 7/11, and "low" 0/4 -- but low calls still landed within one bucket,
# so a low call is treated as "roughly this old" rather than thrown away.
# Training rows carry the true year and are not smoothed.
ERA_TRUST = {"high": 0.8, "medium": 0.55, "low": 0.34}

CATEGORY_TO_TYPE = {
    "day-cab-trucks": "day_cab",
    "sleeper-trucks": "sleeper",
    "dump-trucks": "dump",
}
CUE_TO_TYPE = {
    "day_cab_tractor": "day_cab",
    "sleeper_tractor": "sleeper",
    "heavy_dump_truck": "dump",
    "day_cab": "day_cab",
    "sleeper": "sleeper",
    "dump": "dump",
    "dumper": "dump",
    "day-cab-truck": "day_cab",
    "sleeper-truck": "sleeper",
    "dump-truck": "dump",
    "day_cab_truck": "day_cab",
    "sleeper_truck": "sleeper",
    "dump_truck": "dump",
}

# Brand spellings collapse to one canonical label so a brand's listings land in
# the same cell instead of splitting across variants.
BRAND_ALIASES = {
    "FREIGHTLINER": "FREIGHTLINER",
    "INTERNATIONAL": "INTERNATIONAL",
    "NAVISTAR": "INTERNATIONAL",
    "INTERNATIONAL/NAVISTAR": "INTERNATIONAL",
    "KENWORTH": "KENWORTH",
    "PETERBILT": "PETERBILT",
    "MACK": "MACK",
    "VOLVO": "VOLVO",
    "WESTERN STAR": "WESTERN STAR",
    "WESTERNSTAR": "WESTERN STAR",
    "STERLING": "STERLING",
    "FORD": "FORD",
    "CHEVROLET": "CHEVROLET",
    "CHEVY": "CHEVROLET",
    "GMC": "GMC",
    "HINO": "HINO",
    "ISUZU": "ISUZU",
    "UD": "UD",
    "NISSAN DIESEL": "UD",
    "AUTOCAR": "AUTOCAR",
    "OSHKOSH": "OSHKOSH",
}


def canonical_brand(brand: str | None) -> str:
    if not brand:
        return UNKNOWN_BRAND
    b = " ".join(brand.strip().upper().split())
    return BRAND_ALIASES.get(b, b)


# Kept for import compatibility with predict.py / label_sample.py.
def brand_cell(brand: str | None) -> str:
    return canonical_brand(brand)


def normalize_truck_type(value: str | None) -> str:
    if not value:
        return OTHER_TYPE
    v = value.strip().lower().replace(" ", "_")
    v_hyphen = v.replace("_", "-")
    if v in TRUCK_TYPES:
        return v
    dashed = value.strip().lower()
    if dashed in CATEGORY_TO_TYPE:
        return CATEGORY_TO_TYPE[dashed]
    underscored = dashed.replace("_", "-")
    if underscored in CATEGORY_TO_TYPE:
        return CATEGORY_TO_TYPE[underscored]
    if v in CUE_TO_TYPE:
        return CUE_TO_TYPE[v]
    if v_hyphen in CUE_TO_TYPE:
        return CUE_TO_TYPE[v_hyphen]
    if v == "none" or v_hyphen == "none":
        return OTHER_TYPE
    if "dump" in v:
        return "dump"
    if "sleeper" in v:
        return "sleeper"
    if "day" in v and "cab" in v:
        return "day_cab"
    return OTHER_TYPE


def normalize_era(value: str | None) -> str:
    """Map assorted spellings of the era buckets onto the canonical labels."""
    if not value:
        return UNKNOWN_ERA
    v = value.strip().lower().replace("-", "_").replace(" ", "_")
    v = v.replace("+", "_plus")
    aliases = {
        "pre_2010": "pre_2010",
        "pre2010": "pre_2010",
        "before_2010": "pre_2010",
        "2010_2015": "2010_2015",
        "2016_2020": "2016_2020",
        "2021_plus": "2021_plus",
        "2021plus": "2021_plus",
        "2021_": "2021_plus",
        "unknown": UNKNOWN_ERA,
    }
    return aliases.get(v, UNKNOWN_ERA)


def era_from_year(year: int | None) -> str:
    if not year:
        return UNKNOWN_ERA
    y = int(year)
    if y <= 2009:
        return "pre_2010"
    if y <= 2015:
        return "2010_2015"
    if y <= 2020:
        return "2016_2020"
    return "2021_plus"


def load_year_map(progress_path: Path, cache_path: Path) -> dict[str, int]:
    """Map friendly listing code -> model year, using scraper state files.

    Years are metadata, not image-derived, so they are used only to TRAIN the
    era effect. At prediction time the era comes from the VLM's photo estimate.
    """
    if not progress_path.exists() or not cache_path.exists():
        return {}
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    cache = json.loads(cache_path.read_text(encoding="utf-8"))

    stubs: dict[str, dict] = {}
    for entry in cache.values():
        for tp_id, stub in (entry.get("stubs") or {}).items():
            stubs[tp_id] = stub

    years: dict[str, int] = {}
    for tp_id, rec in progress.items():
        code = rec.get("code")
        stub = stubs.get(tp_id)
        if not code or not stub:
            continue
        year = stub.get("year")
        if year:
            try:
                years[code] = int(year)
            except (TypeError, ValueError):
                continue
    return years


def load_all_priced(
    path: Path,
    year_map: dict[str, int] | None = None,
) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    year_map = year_map or {}
    rows = []
    for folder, listings in data.items():
        if not folder.startswith("positive"):
            continue
        category = folder.split("/", 1)[1] if "/" in folder else folder
        truck_type = normalize_truck_type(category)
        for listing in listings:
            if listing.get("price") is None or not listing.get("brand"):
                continue
            if listing.get("currency") not in (None, "USD"):
                continue
            year = year_map.get(listing["id"])
            rows.append(
                {
                    "listing_id": listing["id"],
                    "brand": canonical_brand(listing["brand"]),
                    "brand_cell": canonical_brand(listing["brand"]),
                    "truck_type": truck_type,
                    "category": category,
                    "year": year,
                    "era": era_from_year(year),
                    "price": float(listing["price"]),
                }
            )
    return rows


def load_ok_labels(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Labels not found: {path}. Run merge_batch_labels.py first.")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("status") != "ok":
            continue
        truck_type = normalize_truck_type(
            obj.get("truck_type") or obj.get("vehicle_cue") or obj.get("category")
        )
        rows.append(
            {
                "listing_id": obj["listing_id"],
                # VLM-read brand where available: that is what inference will see.
                "brand": canonical_brand(obj.get("vlm_brand") or obj.get("listing_brand")),
                "brand_cell": canonical_brand(obj.get("vlm_brand") or obj.get("listing_brand")),
                "truck_type": truck_type,
                "era": normalize_era(obj.get("era")),
                "era_confidence": obj.get("era_confidence"),
                "model_series": obj.get("model_series"),
                "true_year": obj.get("true_year"),
                "price": float(obj["price"]),
                "overall_score": obj.get("overall_score"),
                "total_penalty_percent": obj.get("total_penalty_percent"),
                "vlm_brand": obj.get("vlm_brand"),
                "listing_brand": obj.get("listing_brand"),
                "primary_subject": obj.get("primary_subject"),
            }
        )
    if not rows:
        raise SystemExit(f"No usable ok labels in {path}")
    return rows


# --------------------------------------------------------------------------
# hierarchical cell means
# --------------------------------------------------------------------------


class CellIndex:
    """Group sums over log-price, supporting cheap leave-one-out."""

    def __init__(self, rows: list[dict]) -> None:
        self.total_n = 0
        self.total_s = 0.0
        self.type_n: dict[str, int] = defaultdict(int)
        self.type_s: dict[str, float] = defaultdict(float)
        self.brand_n: dict[tuple, int] = defaultdict(int)
        self.brand_s: dict[tuple, float] = defaultdict(float)
        self.era_n: dict[tuple, int] = defaultdict(int)
        self.era_s: dict[tuple, float] = defaultdict(float)
        for r in rows:
            self.add(r, +1)

    def add(self, row: dict, sign: int) -> None:
        lp = math.log(row["price"])
        t, b, e = row["truck_type"], row["brand"], row["era"]
        self.total_n += sign
        self.total_s += sign * lp
        self.type_n[t] += sign
        self.type_s[t] += sign * lp
        self.brand_n[(t, b)] += sign
        self.brand_s[(t, b)] += sign * lp
        if e != UNKNOWN_ERA:
            self.era_n[(t, b, e)] += sign
            self.era_s[(t, b, e)] += sign * lp

    @staticmethod
    def _shrink(n: int, s: float, parent: float, k: float) -> float:
        if n <= 0:
            return parent
        w = n / (n + k)
        return w * (s / n) + (1.0 - w) * parent

    def mu(self, truck_type: str, brand: str, era: str, k: dict[str, float]) -> float:
        glob = self.total_s / self.total_n if self.total_n else 0.0
        mu_t = self._shrink(
            self.type_n.get(truck_type, 0), self.type_s.get(truck_type, 0.0), glob, k["k_type"]
        )
        key_b = (truck_type, brand)
        mu_b = self._shrink(
            self.brand_n.get(key_b, 0), self.brand_s.get(key_b, 0.0), mu_t, k["k_brand"]
        )
        if era == UNKNOWN_ERA:
            return mu_b
        key_e = (truck_type, brand, era)
        return self._shrink(
            self.era_n.get(key_e, 0), self.era_s.get(key_e, 0.0), mu_b, k["k_era"]
        )

    def mu_soft(
        self,
        truck_type: str,
        brand: str,
        era: str,
        era_confidence: str | None,
        k: dict[str, float],
    ) -> float:
        """mu with the era treated as a distribution when it came from a photo.

        A VLM era call is a guess, and a wrong bucket is worse than no bucket at
        all. So the stated confidence is turned into probability mass on the named
        bucket, with the rest spread over its neighbors, and mu is averaged over
        that distribution. Rows with no stated confidence (training rows, whose
        era comes from the real model year) use the era cell directly.
        """
        if era_confidence is None:
            return self.mu(truck_type, brand, era, k)
        post = era_posterior(era, era_confidence)
        if post is None:
            return self.mu(truck_type, brand, UNKNOWN_ERA, k)
        return sum(w * self.mu(truck_type, brand, e, k) for e, w in post.items())


def era_posterior(era: str, confidence: str | None) -> dict[str, float] | None:
    """Probability over era buckets for a photo-derived era call.

    Returns None when the era is unusable, meaning the caller should fall back
    to the (type, brand) level.
    """
    if era == UNKNOWN_ERA or era not in ERAS:
        return None
    p = ERA_TRUST.get((confidence or "low").strip().lower(), 0.0)
    if p <= 0.0:
        return None
    i = ERAS.index(era)
    neighbors = [j for j in (i - 1, i + 1) if 0 <= j < len(ERAS)]
    if not neighbors:
        return {era: 1.0}
    out = {era: p}
    share = (1.0 - p) / len(neighbors)
    for j in neighbors:
        out[ERAS[j]] = share
    return out


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    w = idx - lo
    return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w


def population_loo(rows: list[dict], k: dict[str, float]) -> list[dict]:
    """Leave-one-out prediction for every priced listing (cells only)."""
    index = CellIndex(rows)
    out = []
    for r in rows:
        index.add(r, -1)
        mu = index.mu(r["truck_type"], r["brand"], r["era"], k)
        index.add(r, +1)
        pred = math.exp(mu)
        out.append(
            {
                "listing_id": r["listing_id"],
                "truck_type": r["truck_type"],
                "brand": r["brand"],
                "era": r["era"],
                "actual": r["price"],
                "pred": pred,
                "ape": abs(pred - r["price"]) / r["price"],
            }
        )
    return out


def tune_cells(rows: list[dict]) -> tuple[dict[str, float], dict]:
    """Stage A: tune shrinkage on the full priced population."""
    grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    best_k = None
    best = None
    tried = 0
    for k_type, k_brand, k_era in itertools.product(grid, repeat=3):
        k = {"k_type": k_type, "k_brand": k_brand, "k_era": k_era}
        apes = sorted(p["ape"] for p in population_loo(rows, k))
        med = percentile(apes, 50)
        mean_ape = statistics.mean(apes)
        tried += 1
        if best is None or med < best["median_ape"] - 1e-12:
            best = {"median_ape": med, "mean_ape": mean_ape}
            best_k = k
        elif best is not None and abs(med - best["median_ape"]) <= 1e-12 and mean_ape < best["mean_ape"]:
            best = {"median_ape": med, "mean_ape": mean_ape}
            best_k = k
    assert best_k is not None and best is not None
    best["grid_tried"] = tried
    return best_k, best


# --------------------------------------------------------------------------
# condition term
# --------------------------------------------------------------------------


def ols_slope(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    if n < 3:
        return 0.0, (statistics.mean(ys) if ys else 0.0), float("inf")
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-12:
        return 0.0, my, float("inf")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sigma2 = sum(r**2 for r in resid) / max(n - 2, 1)
    return slope, intercept, math.sqrt(sigma2 / sxx)


def condition_x(row: dict) -> float | None:
    """Condition regressor: the 1-5 overall score.

    The rubric's total_penalty_percent is 0 for BOTH score 4 and score 5, so it
    cannot separate Good from Excellent. The raw score can.
    """
    score = row.get("overall_score")
    if score is None:
        return None
    return float(score)


def fit_condition(
    labeled: list[dict],
    index: CellIndex,
    k: dict[str, float],
    condition_scale: float,
) -> dict:
    xs, ys = [], []
    for r in labeled:
        x = condition_x(r)
        if x is None:
            continue
        mu = index.mu_soft(
            r["truck_type"], r["brand"], r["era"], r.get("era_confidence"), k
        )
        xs.append(x)
        ys.append(math.log(r["price"]) - mu)
    if not xs:
        return {
            "beta": 0.0,
            "beta_se": float("inf"),
            "score_bar": 0.0,
            "condition_scale": condition_scale,
            "n": 0,
        }
    score_bar = statistics.mean(xs)
    centered = [x - score_bar for x in xs]
    beta, _intercept, se = ols_slope(centered, ys)
    return {
        "beta": beta,
        "beta_se": se,
        "score_bar": score_bar,
        "condition_scale": condition_scale,
        "n": len(xs),
    }


def predict_center(
    row: dict,
    index: CellIndex,
    k: dict[str, float],
    cond: dict,
) -> float:
    mu = index.mu_soft(
        row["truck_type"], row["brand"], row["era"], row.get("era_confidence"), k
    )
    x = condition_x(row)
    effect = 0.0
    if x is not None and cond.get("n", 0) >= 3:
        effect = (
            float(cond["condition_scale"])
            * float(cond["beta"])
            * (x - float(cond["score_bar"]))
        )
    return math.exp(mu + effect)


def evaluate_labeled_loo(
    labeled: list[dict],
    population: list[dict],
    k: dict[str, float],
    condition_scale: float,
) -> dict:
    """LOO over labeled rows: hold out of BOTH the cells and the condition fit."""
    pop_by_id = {r["listing_id"] for r in population}
    apes = []
    preds = []
    for i, held in enumerate(labeled):
        train_labels = labeled[:i] + labeled[i + 1 :]
        pop = [r for r in population if r["listing_id"] != held["listing_id"]]
        index = CellIndex(pop)
        cond = fit_condition(train_labels, index, k, condition_scale)
        center = predict_center(held, index, k, cond)
        ape = abs(center - held["price"]) / held["price"]
        apes.append(ape)
        preds.append(
            {
                "listing_id": held["listing_id"],
                "actual": held["price"],
                "center": center,
                "ape": ape,
                "truck_type": held["truck_type"],
                "brand": held["brand"],
                "era": held["era"],
                "era_confidence": held.get("era_confidence"),
                "in_population": held["listing_id"] in pop_by_id,
            }
        )
    apes_sorted = sorted(apes)
    return {
        "median_ape": percentile(apes_sorted, 50),
        "mean_ape": statistics.mean(apes) if apes else 0.0,
        "p75_ape": percentile(apes_sorted, 75),
        "n": len(labeled),
        "preds": preds,
    }


def tune_condition(
    labeled: list[dict],
    population: list[dict],
    k: dict[str, float],
) -> tuple[float, dict]:
    """Stage B: tune only the condition scale, on the labeled rows."""
    best_scale = 0.0
    best = None
    for scale in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        metrics = evaluate_labeled_loo(labeled, population, k, scale)
        if best is None or metrics["median_ape"] < best["median_ape"] - 1e-12:
            best = metrics
            best_scale = scale
    assert best is not None
    return best_scale, best


# --------------------------------------------------------------------------
# adaptive range
# --------------------------------------------------------------------------


def build_error_table(
    population: list[dict],
    k: dict[str, float],
    error_percentile: float,
) -> dict:
    """Per-(type, era) APE percentile from population LOO, with fallbacks."""
    loo = population_loo(population, k)
    by_type_era: dict[tuple, list[float]] = defaultdict(list)
    by_type: dict[str, list[float]] = defaultdict(list)
    allv: list[float] = []
    for p in loo:
        by_type_era[(p["truck_type"], p["era"])].append(p["ape"])
        by_type[p["truck_type"]].append(p["ape"])
        allv.append(p["ape"])

    MIN_N = 8  # below this a percentile is too noisy to trust; fall back
    type_era = {
        f"{t}|{e}": {"n": len(v), "bound": percentile(sorted(v), error_percentile)}
        for (t, e), v in by_type_era.items()
        if len(v) >= MIN_N
    }
    types = {
        t: {"n": len(v), "bound": percentile(sorted(v), error_percentile)}
        for t, v in by_type.items()
    }
    return {
        "min_n": MIN_N,
        "percentile": error_percentile,
        "type_era": type_era,
        "types": types,
        "global": {"n": len(allv), "bound": percentile(sorted(allv), error_percentile)},
    }


def lookup_error_bound(
    table: dict,
    truck_type: str,
    era: str,
    era_confidence: str | None = None,
) -> float:
    """Error bound for one truck, averaged over the era posterior.

    A shaky era call therefore widens the range on its own, because mass lands
    on neighboring buckets that may carry larger errors.
    """

    def flat(e: str) -> float | None:
        hit = table.get("type_era", {}).get(f"{truck_type}|{e}")
        return float(hit["bound"]) if hit else None

    if era_confidence is not None:
        post = era_posterior(era, era_confidence)
        if post:
            num = 0.0
            den = 0.0
            for e, w in post.items():
                b = flat(e)
                if b is not None:
                    num += w * b
                    den += w
            if den > 0:
                return num / den
    else:
        direct = flat(era)
        if direct is not None:
            return direct

    hit = table.get("types", {}).get(truck_type)
    if hit:
        return float(hit["bound"])
    return float(table["global"]["bound"])


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def brand_weight_table(population: list[dict], k: dict[str, float]) -> dict:
    """Learned brand multipliers relative to each type's baseline.

    multiplier = exp(mu_(type,brand) - mu_type): >1 means the brand prices
    above its body-style baseline. This is the "brand weight" list, learned
    from listings rather than hand-assigned.
    """
    index = CellIndex(population)
    glob = index.total_s / index.total_n if index.total_n else 0.0
    out: dict[str, dict] = {}
    counts = Counter((r["truck_type"], r["brand"]) for r in population)
    for (t, b), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        mu_t = CellIndex._shrink(
            index.type_n.get(t, 0), index.type_s.get(t, 0.0), glob, k["k_type"]
        )
        mu_b = index.mu(t, b, UNKNOWN_ERA, k)
        out[f"{t}|{b}"] = {
            "n": n,
            "multiplier": round(math.exp(mu_b - mu_t), 4),
            "baseline_usd": round(math.exp(mu_b), 2),
        }

    overall: dict[str, dict] = {}
    by_brand = Counter(r["brand"] for r in population)
    for b, n in by_brand.most_common():
        mults = [
            math.log(out[f"{t}|{b}"]["multiplier"])
            for t in TRUCK_TYPES
            if f"{t}|{b}" in out
        ]
        if mults:
            overall[b] = {"n": n, "multiplier": round(math.exp(statistics.mean(mults)), 4)}
    return {"by_type_brand": out, "by_brand": overall}


def era_weight_table(population: list[dict], k: dict[str, float]) -> dict:
    """Learned era multipliers relative to each (type, brand) baseline."""
    index = CellIndex(population)
    counts = Counter((r["truck_type"], r["era"]) for r in population)
    out: dict[str, dict] = {}
    for (t, e), n in sorted(counts.items()):
        if e == UNKNOWN_ERA:
            continue
        logs = [
            math.log(r["price"])
            for r in population
            if r["truck_type"] == t and r["era"] == e
        ]
        base = [math.log(r["price"]) for r in population if r["truck_type"] == t]
        if not logs or not base:
            continue
        out[f"{t}|{e}"] = {
            "n": n,
            "multiplier": round(math.exp(statistics.mean(logs) - statistics.mean(base)), 4),
            "median_usd": round(math.exp(statistics.median(logs)), 2),
        }
    return out


def _display_truck_type(truck_type: str) -> str:
    return {
        "day_cab": "Day Cab",
        "sleeper": "Sleeper",
        "dump": "Dump",
    }.get(truck_type, truck_type.replace("_", " ").title())


def render_bar_chart(
    preds: list[dict],
    error_table: dict,
    out_path: Path,
    title_suffix: str,
) -> None:
    del title_suffix  # kept for call-site compatibility; title is fixed below
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
    except ImportError:
        print("matplotlib not installed; skipping plot. pip install matplotlib")
        return

    rows = sorted(preds, key=lambda r: r["actual"])
    type_labels: list[str] = []
    actuals: list[float] = []
    centers: list[float] = []
    lo_err: list[float] = []
    hi_err: list[float] = []
    for r in rows:
        bound = lookup_error_bound(
            error_table, r["truck_type"], r["era"], r.get("era_confidence")
        )
        center = r["center"]
        low = max(0.0, center * (1.0 - bound))
        high = center * (1.0 + bound)
        type_labels.append(_display_truck_type(r["truck_type"]))
        actuals.append(r["actual"])
        centers.append(center)
        lo_err.append(center - low)
        hi_err.append(high - center)

    n = len(rows)
    x = np.arange(n)
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(12, n * 0.55), 6.8))

    ax.bar(
        x - width / 2,
        actuals,
        width,
        label="Actual price",
        color="#0B3D91",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.bar(
        x + width / 2,
        centers,
        width,
        label="Predicted Price Center",
        color="#FF7A00",
        edgecolor="white",
        linewidth=0.4,
        yerr=np.vstack([lo_err, hi_err]),
        error_kw={"ecolor": "#3d3d3d", "capsize": 3, "elinewidth": 1.1},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(type_labels, fontsize=8, fontweight="bold", rotation=45, ha="right")
    ax.tick_params(axis="x", pad=2)

    ax.set_ylabel("USD price")
    ax.set_title(
        "Actual vs Predicted Price Ranges",
        fontweight="bold",
        fontsize=14,
        loc="center",
        pad=12,
    )
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20_000))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"${v / 1000:.0f}k"))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listings", type=Path, default=LISTINGS_JSON)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--out", type=Path, default=MODEL_OUT)
    parser.add_argument("--plot", type=Path, default=PLOT_OUT)
    # p75 keeps the range near ±50% while holding the true price ~70% of the
    # time; p50 halves the width but only covers ~45%.
    parser.add_argument("--error-percentile", type=float, default=75.0)
    parser.add_argument("--no-tune", action="store_true", help="use default shrinkage")
    args = parser.parse_args()

    year_map = load_year_map(PROGRESS_JSON, SEARCH_CACHE_JSON)
    population = load_all_priced(args.listings, year_map)
    labeled = load_ok_labels(args.labels)

    with_era = sum(1 for r in population if r["era"] != UNKNOWN_ERA)
    print(f"Population priced listings: {len(population)}  (with model year: {with_era})")
    print(f"Labeled rows:               {len(labeled)}")
    print("Population eras:", dict(Counter(r["era"] for r in population)))
    print("Population brands:", dict(Counter(r["brand"] for r in population).most_common()))

    # ---- Stage A: cells, tuned on all 358 listings
    if args.no_tune:
        k = {"k_type": 1.0, "k_brand": 1.0, "k_era": 1.0}
        apes = sorted(p["ape"] for p in population_loo(population, k))
        cell_metrics = {
            "median_ape": percentile(apes, 50),
            "mean_ape": statistics.mean(apes),
            "grid_tried": 0,
        }
    else:
        print("\nStage A: tuning hierarchical shrinkage on the full population...")
        k, cell_metrics = tune_cells(population)
    print(
        f"  k_type={k['k_type']}  k_brand={k['k_brand']}  k_era={k['k_era']}"
        f"   ({cell_metrics['grid_tried']} combos)"
    )
    print(
        f"  population LOO medAPE={100 * cell_metrics['median_ape']:.1f}%  "
        f"meanAPE={100 * cell_metrics['mean_ape']:.1f}%"
    )

    # Ablations so the contribution of each feature is visible.
    print("\n  ablations (population LOO medAPE):")
    for name, mutate in (
        ("type only", lambda r: {**r, "brand": UNKNOWN_BRAND, "era": UNKNOWN_ERA}),
        ("type+brand", lambda r: {**r, "era": UNKNOWN_ERA}),
        ("type+era", lambda r: {**r, "brand": UNKNOWN_BRAND}),
        ("type+brand+era", lambda r: r),
    ):
        rows = [mutate(r) for r in population]
        apes = sorted(p["ape"] for p in population_loo(rows, k))
        print(f"    {name:<16} {100 * percentile(apes, 50):.1f}%")

    # ---- Stage B: condition, tuned on the labeled rows only
    print("\nStage B: tuning condition scale on labeled rows...")
    condition_scale, label_metrics = tune_condition(labeled, population, k)
    index = CellIndex(population)
    cond = fit_condition(labeled, index, k, condition_scale)
    print(
        f"  condition_scale={condition_scale}  beta={cond['beta']:+.4f} "
        f"(se={cond['beta_se']:.4f}, n={cond['n']})"
    )
    sig = cond["beta_se"] not in (float("inf"),) and abs(cond["beta"]) > 2 * cond["beta_se"]
    print(f"  condition is {'SIGNIFICANT' if sig else 'NOT significant'} at this sample size")
    print(
        f"  labeled LOO medAPE={100 * label_metrics['median_ape']:.1f}%  "
        f"meanAPE={100 * label_metrics['mean_ape']:.1f}%  n={label_metrics['n']}"
    )

    # ---- VLM feature accuracy vs scrape metadata (how much the photo misses)
    type_truth = {r["listing_id"]: r["truck_type"] for r in population}
    brand_truth = {r["listing_id"]: r["brand"] for r in population}
    era_truth = {r["listing_id"]: r["era"] for r in population}
    t_ok = sum(1 for r in labeled if type_truth.get(r["listing_id"]) == r["truck_type"])
    b_ok = sum(1 for r in labeled if brand_truth.get(r["listing_id"]) == r["brand"])
    era_cmp = [
        (era_truth[r["listing_id"]], r["era"])
        for r in labeled
        if r["listing_id"] in era_truth and era_truth[r["listing_id"]] != UNKNOWN_ERA
    ]
    e_ok = sum(1 for a, b in era_cmp if a == b)
    print("\nVLM feature accuracy vs listing metadata:")
    print(f"  truck_type {t_ok}/{len(labeled)}   brand {b_ok}/{len(labeled)}", end="")
    if era_cmp:
        print(f"   era {e_ok}/{len(era_cmp)}")
    else:
        print("   era n/a (labels carry no era yet)")

    # ---- adaptive range
    error_table = build_error_table(population, k, args.error_percentile)
    print(
        f"\nAdaptive range (p{args.error_percentile:.0f} of population LOO APE): "
        f"global ±{100 * error_table['global']['bound']:.0f}%"
    )
    for key, val in sorted(error_table["type_era"].items()):
        print(f"    {key:<26} n={val['n']:<4} ±{100 * val['bound']:.0f}%")

    hits = 0
    for p in label_metrics["preds"]:
        bound = lookup_error_bound(
            error_table, p["truck_type"], p["era"], p.get("era_confidence")
        )
        low, high = p["center"] * (1 - bound), p["center"] * (1 + bound)
        if low <= p["actual"] <= high:
            hits += 1
    coverage = hits / len(label_metrics["preds"]) if label_metrics["preds"] else 0.0
    print(f"  range contains the actual price for {100 * coverage:.0f}% of labeled trucks")

    model = {
        "formula": (
            "mu = shrink((type,brand,era) -> (type,brand) -> type -> global); "
            "center = exp(mu + condition_scale*beta*(overall_score - score_bar)); "
            "low/high = center*(1 -/+ error_bound(type, era))"
        ),
        "model_family": "type+brand+era+condition",
        "shrinkage": k,
        "era_trust": ERA_TRUST,
        "condition": cond,
        "error_table": error_table,
        "error_percentile": args.error_percentile,
        "metrics": {
            "population_loo_median_ape": cell_metrics["median_ape"],
            "population_loo_mean_ape": cell_metrics["mean_ape"],
            "labeled_loo_median_ape": label_metrics["median_ape"],
            "labeled_loo_mean_ape": label_metrics["mean_ape"],
            "labeled_range_coverage": coverage,
            "labeled_n": label_metrics["n"],
            "population_n": len(population),
        },
        "brand_weights": brand_weight_table(population, k),
        "era_weights": era_weight_table(population, k),
        "cells": {
            "global_mu": index.total_s / index.total_n if index.total_n else 0.0,
            "types": {
                t: {"n": index.type_n[t], "mu": index.mu(t, UNKNOWN_BRAND, UNKNOWN_ERA, k)}
                for t in index.type_n
            },
            "type_brand": {
                f"{t}|{b}": {"n": n, "mu": index.mu(t, b, UNKNOWN_ERA, k)}
                for (t, b), n in index.brand_n.items()
                if n > 0
            },
            "type_brand_era": {
                f"{t}|{b}|{e}": {"n": n, "mu": index.mu(t, b, e, k)}
                for (t, b, e), n in index.era_n.items()
                if n > 0
            },
        },
        "truck_types": list(TRUCK_TYPES),
        "eras": list(ERAS),
        "unknown_brand": UNKNOWN_BRAND,
        "unknown_era": UNKNOWN_ERA,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote model -> {args.out}")

    render_bar_chart(
        label_metrics["preds"],
        error_table,
        args.plot,
        f"type+brand+era+condition, LOO medAPE {100 * label_metrics['median_ape']:.0f}%",
    )


if __name__ == "__main__":
    main()
