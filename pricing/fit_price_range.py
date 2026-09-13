"""Fit a price RANGE model from brand + truck type + condition.

Point estimate
--------------
center = exp( mu_(type, brand) + beta * (penalty - penalty_bar) )

mu_(type, brand) is the shrunk log-price mean for that (truck_type, brand_cell)
cell, learned from all priced Class 7/8 listings (folder category = ground-truth
type). beta is learned from the labeled sample.

Range = prediction error bound (not brand-spread width)
-------------------------------------------------------
On leave-one-out, measure absolute percentage error of `center`.
error_bound = median APE (default). Then:

    low  = center * (1 - error_bound)
    high = center * (1 + error_bound)

So the band is "how wrong the point estimate usually is", not how wide raw
prices are inside a brand.

Variants
--------
V1  truck type only
V2  truck type + brand
V3  truck type + brand + condition

Selection: among variants, pick the smallest error_bound (tightest range).

Usage:
    python pricing/fit_price_range.py
    python pricing/fit_price_range.py --labels pricing/labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATASET = REPO / "image_scraper" / "truckpaper_scraped_images_dataset"
LISTINGS_JSON = DATASET / "truckpaper_scraped_listings.json"
LABELS_PATH = ROOT / "labels.jsonl"
MODEL_OUT = ROOT / "price_range_model.json"
PLOT_OUT = ROOT / "price_range_plot.png"

MAJOR_BRANDS = ("FREIGHTLINER", "INTERNATIONAL", "KENWORTH", "PETERBILT", "MACK")
OTHER_BRAND = "OTHER"
TRUCK_TYPES = ("day_cab", "sleeper", "dump")
OTHER_TYPE = "other"
SHRINK_K = 1.0
ERROR_PERCENTILE = 50  # median APE -> half-width; raise to 75 for wider bands

# Folder category / VLM vehicle_cue -> truck_type
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


def brand_cell(brand: str | None) -> str:
    if not brand:
        return OTHER_BRAND
    b = brand.strip().upper()
    return b if b in MAJOR_BRANDS else OTHER_BRAND


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
    if v in ("none",) or v_hyphen == "none":
        return OTHER_TYPE
    if "dump" in v:
        return "dump"
    if "sleeper" in v:
        return "sleeper"
    if "day" in v and "cab" in v:
        return "day_cab"
    return OTHER_TYPE


def load_all_priced(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
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
            rows.append(
                {
                    "listing_id": listing["id"],
                    "brand": listing["brand"].strip().upper(),
                    "brand_cell": brand_cell(listing["brand"]),
                    "truck_type": truck_type,
                    "category": category,
                    "price": float(listing["price"]),
                }
            )
    return rows


def load_ok_labels(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Labels not found: {path}. Run label_sample.py first.")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("status") != "ok":
            continue
        penalty = obj.get("total_penalty_percent")
        if penalty is None:
            continue
        # Prefer VLM truck type; fall back to listing folder category
        truck_type = normalize_truck_type(
            obj.get("truck_type") or obj.get("vehicle_cue") or obj.get("category")
        )
        rows.append(
            {
                "listing_id": obj["listing_id"],
                "brand": (obj.get("listing_brand") or obj.get("vlm_brand") or "").strip().upper(),
                "brand_cell": obj.get("brand_cell")
                or brand_cell(obj.get("listing_brand") or obj.get("vlm_brand")),
                "truck_type": truck_type,
                "price": float(obj["price"]),
                "total_penalty_percent": float(penalty),
                "overall_score": obj.get("overall_score"),
                "vlm_brand": obj.get("vlm_brand"),
                "vehicle_cue": obj.get("vehicle_cue"),
                "primary_subject": obj.get("primary_subject"),
            }
        )
    if not rows:
        raise SystemExit(f"No usable ok labels with total_penalty_percent in {path}")
    return rows


def fit_cell_stats(
    log_prices: list[float], parent_mu: float, parent_sd: float, k: float = SHRINK_K
) -> dict:
    n = len(log_prices)
    if n == 0:
        return {"n": 0, "mu": parent_mu, "sd": parent_sd}
    raw_mu = statistics.mean(log_prices)
    raw_sd = statistics.pstdev(log_prices) if n >= 2 else parent_sd
    w = n / (n + k)
    mu = w * raw_mu + (1 - w) * parent_mu
    sd = math.sqrt(w * (raw_sd**2) + (1 - w) * (parent_sd**2))
    return {"n": n, "mu": mu, "sd": max(sd, 1e-6)}


def build_tables(population: list[dict]) -> dict:
    """Build global / type / (type, brand) log-price tables."""
    logs = [math.log(r["price"]) for r in population]
    global_mu = statistics.mean(logs)
    global_sd = statistics.pstdev(logs) if len(logs) >= 2 else 1.0

    by_type: dict[str, list[float]] = defaultdict(list)
    by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in population:
        lp = math.log(r["price"])
        by_type[r["truck_type"]].append(lp)
        by_cell[(r["truck_type"], r["brand_cell"])].append(lp)

    type_stats = {
        t: fit_cell_stats(vals, global_mu, global_sd) for t, vals in by_type.items()
    }
    cell_stats = {}
    for key, vals in by_cell.items():
        t, _b = key
        parent = type_stats.get(t, {"mu": global_mu, "sd": global_sd})
        cell_stats[f"{key[0]}|{key[1]}"] = fit_cell_stats(vals, parent["mu"], parent["sd"])

    return {
        "global_mu": global_mu,
        "global_sd": global_sd,
        "types": type_stats,
        "cells": cell_stats,  # key "type|brand"
        "population_n": len(population),
        "type_counts": dict(Counter(r["truck_type"] for r in population)),
        "brand_counts": dict(Counter(r["brand_cell"] for r in population)),
    }


def lookup_mu(tables: dict, truck_type: str, brand_cell_name: str, use_brand: bool) -> float:
    if use_brand:
        key = f"{truck_type}|{brand_cell_name}"
        if key in tables["cells"]:
            return tables["cells"][key]["mu"]
    if truck_type in tables["types"]:
        return tables["types"][truck_type]["mu"]
    return tables["global_mu"]


def ols_slope(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    if n < 3:
        return 0.0, statistics.mean(ys) if ys else 0.0, float("inf")
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-12:
        return 0.0, my, float("inf")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sse = sum(r**2 for r in resid)
    sigma2 = sse / max(n - 2, 1)
    slope_se = math.sqrt(sigma2 / sxx)
    return slope, intercept, slope_se


def fit_beta(labeled: list[dict], tables: dict, use_brand: bool) -> dict:
    penalties = [r["total_penalty_percent"] for r in labeled]
    penalty_bar = statistics.mean(penalties)
    xs, ys = [], []
    for r in labeled:
        mu = lookup_mu(tables, r["truck_type"], r["brand_cell"], use_brand=use_brand)
        xs.append(r["total_penalty_percent"] - penalty_bar)
        ys.append(math.log(r["price"]) - mu)
    beta, intercept, se = ols_slope(xs, ys)
    return {
        "beta": beta,
        "intercept": intercept,
        "beta_se": se,
        "penalty_bar": penalty_bar,
        "n": len(labeled),
    }


def point_estimate(
    truck_type: str,
    brand_cell_name: str,
    penalty: float | None,
    tables: dict,
    beta_info: dict,
    *,
    use_brand: bool,
    use_condition: bool,
) -> float:
    mu = lookup_mu(tables, truck_type, brand_cell_name, use_brand=use_brand)
    if use_condition and penalty is not None:
        return math.exp(mu + beta_info["beta"] * (penalty - beta_info["penalty_bar"]))
    return math.exp(mu)


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


def evaluate_loo(
    labeled: list[dict],
    population: list[dict],
    *,
    use_brand: bool,
    use_condition: bool,
    error_percentile: float = ERROR_PERCENTILE,
) -> dict:
    apes: list[float] = []
    betas: list[float] = []

    for i, held in enumerate(labeled):
        train_labels = labeled[:i] + labeled[i + 1 :]
        pop = [r for r in population if r["listing_id"] != held["listing_id"]]
        tables = build_tables(pop)
        if use_condition and len(train_labels) >= 3:
            beta_info = fit_beta(train_labels, tables, use_brand=use_brand)
        else:
            beta_info = {
                "beta": 0.0,
                "beta_se": float("inf"),
                "penalty_bar": 0.0,
                "n": 0,
                "intercept": 0.0,
            }
        betas.append(beta_info["beta"])
        yhat = point_estimate(
            held["truck_type"],
            held["brand_cell"],
            held["total_penalty_percent"],
            tables,
            beta_info,
            use_brand=use_brand,
            use_condition=use_condition,
        )
        apes.append(abs(yhat - held["price"]) / held["price"])

    apes_sorted = sorted(apes)
    error_bound = percentile(apes_sorted, error_percentile)
    # coverage of the +/- error_bound band on the same LOO residuals
    coverage = sum(1 for a in apes if a <= error_bound) / len(apes) if apes else 0.0
    return {
        "error_bound": error_bound,
        "coverage": coverage,
        "median_ape": percentile(apes_sorted, 50),
        "p75_ape": percentile(apes_sorted, 75),
        "mean_ape": statistics.mean(apes) if apes else 0.0,
        "mean_beta": statistics.mean(betas) if betas else 0.0,
        "n": len(labeled),
        "error_percentile": error_percentile,
    }


def predict_range(
    truck_type: str,
    brand_cell_name: str,
    penalty: float | None,
    tables: dict,
    beta_info: dict,
    error_bound: float,
    *,
    use_brand: bool,
    use_condition: bool,
) -> dict:
    center = point_estimate(
        truck_type,
        brand_cell_name,
        penalty,
        tables,
        beta_info,
        use_brand=use_brand,
        use_condition=use_condition,
    )
    low = max(0.0, center * (1.0 - error_bound))
    high = center * (1.0 + error_bound)
    return {
        "center": center,
        "low": low,
        "high": high,
        "error_bound": error_bound,
        "truck_type": truck_type,
        "brand_cell": brand_cell_name,
    }


def pick_variant(results: list[dict]) -> dict:
    # Tightest error bound wins (that's the range half-width)
    return min(results, key=lambda r: r["metrics"]["error_bound"])


def render_plot(
    labeled: list[dict],
    tables: dict,
    beta_info: dict,
    error_bound: float,
    *,
    use_brand: bool,
    use_condition: bool,
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot. pip install matplotlib")
        return

    # One panel per truck type
    types_present = [t for t in TRUCK_TYPES if t in tables["types"]]
    if not types_present:
        types_present = list(tables["types"].keys())[:3]
    fig, axes = plt.subplots(
        1, len(types_present), figsize=(4.5 * max(len(types_present), 1), 4), sharey=True
    )
    if len(types_present) == 1:
        axes = [axes]

    xs = list(range(0, 13))
    # Use most common brand overall for the illustrative curve
    top_brand = max(tables["brand_counts"], key=tables["brand_counts"].get)

    for ax, t in zip(axes, types_present):
        lows, centers, highs = [], [], []
        for p in xs:
            pred = predict_range(
                t,
                top_brand,
                float(p),
                tables,
                beta_info,
                error_bound,
                use_brand=use_brand,
                use_condition=use_condition,
            )
            lows.append(pred["low"])
            centers.append(pred["center"])
            highs.append(pred["high"])
        ax.fill_between(xs, lows, highs, alpha=0.25, color="steelblue", label="error band")
        ax.plot(xs, centers, color="steelblue", linewidth=2, label="center")
        pts = [r for r in labeled if r["truck_type"] == t]
        if pts:
            ax.scatter(
                [r["total_penalty_percent"] for r in pts],
                [r["price"] for r in pts],
                color="black",
                s=18,
                zorder=3,
                label="actual",
            )
        n = tables["types"].get(t, {}).get("n", 0)
        ax.set_title(f"{t} (n={n})")
        ax.set_xlabel("total_penalty_percent")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("USD price")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"Price ± error_bound ({100 * error_bound:.0f}%)  "
        f"beta={beta_info['beta']:.4f}  brand={top_brand}"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote plot -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listings", type=Path, default=LISTINGS_JSON)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--out", type=Path, default=MODEL_OUT)
    parser.add_argument("--plot", type=Path, default=PLOT_OUT)
    parser.add_argument(
        "--error-percentile",
        type=float,
        default=ERROR_PERCENTILE,
        help="APE percentile used as +/- error_bound (50=median, 75=wider)",
    )
    args = parser.parse_args()

    population = load_all_priced(args.listings)
    labeled = load_ok_labels(args.labels)
    print(f"Population priced listings: {len(population)}")
    print(f"Labeled ok rows:            {len(labeled)}")
    print("Population types:", dict(Counter(r["truck_type"] for r in population)))
    print("Labeled types:   ", dict(Counter(r["truck_type"] for r in labeled)))

    tables = build_tables(population)

    variants = [
        {"id": "V1", "name": "type", "use_brand": False, "use_condition": False},
        {"id": "V2", "name": "type+brand", "use_brand": True, "use_condition": False},
        {"id": "V3", "name": "type+brand+condition", "use_brand": True, "use_condition": True},
    ]

    results = []
    print("\n=== Leave-one-out harness (range = center ± error_bound) ===")
    print(
        f"{'var':<6} {'name':<22} {'err%':>7} {'cov%':>7} "
        f"{'medAPE%':>9} {'p75APE%':>9} {'beta':>9}"
    )
    for v in variants:
        metrics = evaluate_loo(
            labeled,
            population,
            use_brand=v["use_brand"],
            use_condition=v["use_condition"],
            error_percentile=args.error_percentile,
        )
        # Fit beta on full labeled set for display when condition is on
        if v["use_condition"]:
            beta_full = fit_beta(labeled, tables, use_brand=v["use_brand"])
            metrics["mean_beta"] = beta_full["beta"]
            metrics["beta_se"] = beta_full["beta_se"]
        results.append({**v, "metrics": metrics})
        print(
            f"{v['id']:<6} {v['name']:<22} "
            f"{100 * metrics['error_bound']:7.1f} "
            f"{100 * metrics['coverage']:7.1f} "
            f"{100 * metrics['median_ape']:9.1f} "
            f"{100 * metrics['p75_ape']:9.1f} "
            f"{metrics['mean_beta']:9.4f}"
        )

    winner = pick_variant(results)
    print(
        f"\nWinner: {winner['id']} ({winner['name']})  "
        f"error_bound=±{100 * winner['metrics']['error_bound']:.1f}%  "
        f"(covers {100 * winner['metrics']['coverage']:.0f}% of LOO residuals)"
    )

    beta_info = fit_beta(labeled, tables, use_brand=winner["use_brand"])
    if winner["id"] != "V3":
        # Still store beta for diagnostics even if unused
        beta_info = fit_beta(labeled, tables, use_brand=True)

    if abs(beta_info["beta"]) < 1.96 * beta_info["beta_se"]:
        print(
            "Note: beta not distinguishable from 0 at ~95% confidence "
            "(condition may not move the point estimate)."
        )

    model = {
        "formula": (
            "center = exp(mu_(truck_type, brand) + beta*(penalty - penalty_bar)); "
            "low/high = center * (1 +/- error_bound)  "
            "where error_bound is LOO APE at the chosen percentile"
        ),
        "winner": {
            "id": winner["id"],
            "name": winner["name"],
            "use_brand": winner["use_brand"],
            "use_condition": winner["use_condition"],
            "error_bound": winner["metrics"]["error_bound"],
            "metrics": winner["metrics"],
        },
        "error_percentile": args.error_percentile,
        "beta": beta_info,
        "tables": {
            "global_mu": tables["global_mu"],
            "global_sd": tables["global_sd"],
            "population_n": tables["population_n"],
            "type_counts": tables["type_counts"],
            "brand_counts": tables["brand_counts"],
            "types": {
                t: {"n": s["n"], "mu": s["mu"], "median_price": math.exp(s["mu"])}
                for t, s in tables["types"].items()
            },
            "cells": {
                k: {"n": s["n"], "mu": s["mu"], "median_price": math.exp(s["mu"])}
                for k, s in tables["cells"].items()
            },
        },
        "major_brands": list(MAJOR_BRANDS),
        "other_brand": OTHER_BRAND,
        "truck_types": list(TRUCK_TYPES),
        "variants": [
            {"id": r["id"], "name": r["name"], "metrics": r["metrics"]} for r in results
        ],
        "labeled_n": len(labeled),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote model -> {args.out}")

    render_plot(
        labeled,
        tables,
        beta_info,
        winner["metrics"]["error_bound"],
        use_brand=winner["use_brand"],
        use_condition=winner["use_condition"],
        out_path=args.plot,
    )


if __name__ == "__main__":
    main()
