"""Fit a price RANGE model from listing brands + VLM condition labels.

Uses all priced Class 7/8 listings for brand log-price bands, and the labeled
sample for the condition slope beta and calibration constant z.

Variants (leave-one-out on labeled rows):
  V1  global band
  V2  brand only
  V3  brand + condition slope

Selection: smallest median band width with coverage >= 70% (raise z if needed).

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
OTHER = "OTHER"
COVERAGE_TARGET = 0.70
SHRINK_K = 1.0


def brand_cell(brand: str | None) -> str:
    if not brand:
        return OTHER
    b = brand.strip().upper()
    return b if b in MAJOR_BRANDS else OTHER


def load_all_priced(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for folder, listings in data.items():
        if not folder.startswith("positive"):
            continue
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
        rows.append(
            {
                "listing_id": obj["listing_id"],
                "brand": obj.get("listing_brand") or obj.get("vlm_brand") or "",
                "brand_cell": obj.get("brand_cell") or brand_cell(obj.get("listing_brand")),
                "price": float(obj["price"]),
                "total_penalty_percent": float(penalty),
                "overall_score": obj.get("overall_score"),
                "vlm_brand": obj.get("vlm_brand"),
                "primary_subject": obj.get("primary_subject"),
            }
        )
    if not rows:
        raise SystemExit(f"No usable ok labels with total_penalty_percent in {path}")
    return rows


def fit_cell_stats(log_prices: list[float], global_mu: float, global_sd: float, k: float = SHRINK_K):
    n = len(log_prices)
    if n == 0:
        return {"n": 0, "mu": global_mu, "sd": global_sd, "raw_mu": None, "raw_sd": None}
    raw_mu = statistics.mean(log_prices)
    raw_sd = statistics.pstdev(log_prices) if n >= 2 else global_sd
    w = n / (n + k)
    mu = w * raw_mu + (1 - w) * global_mu
    # shrink variance toward global too
    sd = math.sqrt(w * (raw_sd ** 2) + (1 - w) * (global_sd ** 2))
    sd = max(sd, 1e-6)
    return {"n": n, "mu": mu, "sd": sd, "raw_mu": raw_mu, "raw_sd": raw_sd}


def build_brand_bands(population: list[dict]) -> dict:
    logs = [math.log(r["price"]) for r in population]
    global_mu = statistics.mean(logs)
    global_sd = statistics.pstdev(logs) if len(logs) >= 2 else 1.0
    by_cell: dict[str, list[float]] = defaultdict(list)
    for r in population:
        by_cell[r["brand_cell"]].append(math.log(r["price"]))
    cells = {
        cell: fit_cell_stats(vals, global_mu, global_sd)
        for cell, vals in by_cell.items()
    }
    return {
        "global_mu": global_mu,
        "global_sd": global_sd,
        "cells": cells,
        "population_n": len(population),
        "brand_counts": dict(Counter(r["brand_cell"] for r in population)),
    }


def ols_slope(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Return (slope, intercept, slope_se). Intercept is mean(y) - slope*mean(x)."""
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
    sse = sum(r ** 2 for r in resid)
    sigma2 = sse / max(n - 2, 1)
    slope_se = math.sqrt(sigma2 / sxx)
    return slope, intercept, slope_se


def fit_beta(labeled: list[dict], bands: dict) -> dict:
    """Regress log(price) - mu_brand on (penalty - penalty_bar)."""
    penalties = [r["total_penalty_percent"] for r in labeled]
    penalty_bar = statistics.mean(penalties)
    xs = []
    ys = []
    for r in labeled:
        cell = bands["cells"].get(r["brand_cell"])
        mu = cell["mu"] if cell else bands["global_mu"]
        xs.append(r["total_penalty_percent"] - penalty_bar)
        ys.append(math.log(r["price"]) - mu)
    beta, intercept, se = ols_slope(xs, ys)
    return {
        "beta": beta,
        "intercept": intercept,  # should be ~0; kept for diagnostics
        "beta_se": se,
        "penalty_bar": penalty_bar,
        "n": len(labeled),
    }


def predict_range(
    brand_cell_name: str,
    penalty: float | None,
    bands: dict,
    beta_info: dict,
    z: float,
    use_condition: bool,
) -> dict:
    cell = bands["cells"].get(brand_cell_name)
    if cell is None:
        mu, sd = bands["global_mu"], bands["global_sd"]
        cell_n = 0
    else:
        mu, sd, cell_n = cell["mu"], cell["sd"], cell["n"]

    if use_condition and penalty is not None:
        center_log = mu + beta_info["beta"] * (penalty - beta_info["penalty_bar"])
    else:
        center_log = mu

    center = math.exp(center_log)
    low = center * math.exp(-z * sd)
    high = center * math.exp(z * sd)
    return {
        "center": center,
        "low": low,
        "high": high,
        "brand_cell": brand_cell_name,
        "sd": sd,
        "cell_n": cell_n,
        "z": z,
    }


def evaluate_loo(
    labeled: list[dict],
    population: list[dict],
    *,
    use_brand: bool,
    use_condition: bool,
    z: float,
) -> dict:
    """Leave-one-out on labeled rows; brand bands rebuilt from population each time
    excluding the held-out listing id when present."""
    coverages = []
    widths = []
    ape = []
    betas = []

    for i, held in enumerate(labeled):
        train_labels = labeled[:i] + labeled[i + 1 :]
        pop = [r for r in population if r["listing_id"] != held["listing_id"]]
        bands = build_brand_bands(pop)
        if use_condition and len(train_labels) >= 3:
            beta_info = fit_beta(train_labels, bands)
        else:
            beta_info = {"beta": 0.0, "beta_se": float("inf"), "penalty_bar": 0.0, "n": 0, "intercept": 0.0}
        betas.append(beta_info["beta"])

        cell_name = held["brand_cell"] if use_brand else OTHER
        # For global (V1), force global cell by using a fake empty brand lookup:
        if not use_brand:
            # predict using global mu/sd only
            mu, sd = bands["global_mu"], bands["global_sd"]
            center_log = mu
            if use_condition:
                center_log = mu + beta_info["beta"] * (
                    held["total_penalty_percent"] - beta_info["penalty_bar"]
                )
            center = math.exp(center_log)
            low = center * math.exp(-z * sd)
            high = center * math.exp(z * sd)
            pred = {"center": center, "low": low, "high": high}
        else:
            pred = predict_range(
                cell_name,
                held["total_penalty_percent"],
                bands,
                beta_info,
                z,
                use_condition=use_condition,
            )

        price = held["price"]
        coverages.append(1.0 if pred["low"] <= price <= pred["high"] else 0.0)
        widths.append((pred["high"] - pred["low"]) / pred["center"])
        ape.append(abs(pred["center"] - price) / price)

    widths_sorted = sorted(widths)
    ape_sorted = sorted(ape)
    return {
        "coverage": statistics.mean(coverages) if coverages else 0.0,
        "median_width": widths_sorted[len(widths_sorted) // 2] if widths_sorted else 0.0,
        "mean_width": statistics.mean(widths) if widths else 0.0,
        "median_ape": ape_sorted[len(ape_sorted) // 2] if ape_sorted else 0.0,
        "mean_ape": statistics.mean(ape) if ape else 0.0,
        "mean_beta": statistics.mean(betas) if betas else 0.0,
        "n": len(labeled),
        "z": z,
    }


def calibrate_z(
    labeled: list[dict],
    population: list[dict],
    *,
    use_brand: bool,
    use_condition: bool,
    target: float = COVERAGE_TARGET,
) -> tuple[float, dict]:
    """Find smallest z in a grid that hits coverage target; refine around it."""
    best_z = None
    best_metrics = None
    for z in [x / 10 for x in range(4, 25)]:  # 0.4 .. 2.4
        m = evaluate_loo(
            labeled, population, use_brand=use_brand, use_condition=use_condition, z=z
        )
        if m["coverage"] >= target:
            best_z = z
            best_metrics = m
            break
        best_z = z
        best_metrics = m
    assert best_z is not None and best_metrics is not None
    return best_z, best_metrics


def pick_variant(results: list[dict]) -> dict:
    eligible = [r for r in results if r["metrics"]["coverage"] >= COVERAGE_TARGET]
    pool = eligible if eligible else results
    return min(pool, key=lambda r: r["metrics"]["median_width"])


def render_plot(
    labeled: list[dict],
    bands: dict,
    beta_info: dict,
    z: float,
    use_condition: bool,
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot. pip install matplotlib")
        return

    # Show top 3 brand cells by count
    top_cells = sorted(
        bands["cells"].items(), key=lambda kv: -kv[1]["n"]
    )[:3]
    fig, axes = plt.subplots(1, len(top_cells), figsize=(4.5 * len(top_cells), 4), sharey=True)
    if len(top_cells) == 1:
        axes = [axes]

    penalties = sorted({r["total_penalty_percent"] for r in labeled})
    if not penalties:
        penalties = [0, 3, 6, 9, 12]
    xs = list(range(0, int(max(penalties + [12])) + 1))

    for ax, (cell_name, cell) in zip(axes, top_cells):
        lows, centers, highs = [], [], []
        for p in xs:
            pred = predict_range(cell_name, float(p), bands, beta_info, z, use_condition)
            lows.append(pred["low"])
            centers.append(pred["center"])
            highs.append(pred["high"])
        ax.fill_between(xs, lows, highs, alpha=0.25, color="steelblue", label="range")
        ax.plot(xs, centers, color="steelblue", linewidth=2, label="center")
        pts = [r for r in labeled if r["brand_cell"] == cell_name]
        if pts:
            ax.scatter(
                [r["total_penalty_percent"] for r in pts],
                [r["price"] for r in pts],
                color="black",
                s=18,
                zorder=3,
                label="actual",
            )
        ax.set_title(f"{cell_name} (n={cell['n']})")
        ax.set_xlabel("total_penalty_percent")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("USD price")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Price range vs condition (z={z:.2f}, beta={beta_info['beta']:.4f})")
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
    parser.add_argument("--coverage-target", type=float, default=COVERAGE_TARGET)
    args = parser.parse_args()

    population = load_all_priced(args.listings)
    labeled = load_ok_labels(args.labels)
    print(f"Population priced listings: {len(population)}")
    print(f"Labeled ok rows:            {len(labeled)}")

    brand_agree = [
        r for r in labeled
        if r.get("vlm_brand")
        and r["vlm_brand"].strip().upper() == r["brand"].strip().upper()
    ]
    with_vlm = [r for r in labeled if r.get("vlm_brand")]
    if with_vlm:
        print(
            f"VLM brand agreement: {len(brand_agree)}/{len(with_vlm)} "
            f"({100 * len(brand_agree) / len(with_vlm):.0f}%)"
        )

    # Full-data bands + beta for export
    bands = build_brand_bands(population)
    beta_info = fit_beta(labeled, bands)
    print(
        f"beta={beta_info['beta']:.5f}  se={beta_info['beta_se']:.5f}  "
        f"penalty_bar={beta_info['penalty_bar']:.2f}"
    )

    variants = [
        {"id": "V1", "name": "global", "use_brand": False, "use_condition": False},
        {"id": "V2", "name": "brand", "use_brand": True, "use_condition": False},
        {"id": "V3", "name": "brand+condition", "use_brand": True, "use_condition": True},
    ]

    results = []
    print("\n=== Leave-one-out harness ===")
    print(f"{'var':<6} {'name':<18} {'z':>5} {'cov%':>7} {'medW%':>8} {'medAPE%':>9} {'beta':>9}")
    for v in variants:
        z, metrics = calibrate_z(
            labeled,
            population,
            use_brand=v["use_brand"],
            use_condition=v["use_condition"],
            target=args.coverage_target,
        )
        row = {**v, "z": z, "metrics": metrics}
        results.append(row)
        print(
            f"{v['id']:<6} {v['name']:<18} {z:5.2f} "
            f"{100 * metrics['coverage']:7.1f} "
            f"{100 * metrics['median_width']:8.1f} "
            f"{100 * metrics['median_ape']:9.1f} "
            f"{metrics['mean_beta']:9.4f}"
        )

    winner = pick_variant(results)
    print(
        f"\nWinner: {winner['id']} ({winner['name']})  "
        f"z={winner['z']:.2f}  coverage={100 * winner['metrics']['coverage']:.1f}%  "
        f"median_width={100 * winner['metrics']['median_width']:.1f}%"
    )

    if winner["id"] == "V2" and abs(beta_info["beta"]) < 1.96 * beta_info["beta_se"]:
        print(
            "Note: beta is not distinguishable from 0 at ~95% confidence; "
            "brand-only (V2) is expected."
        )

    model = {
        "formula": (
            "center = exp(mu_cell + beta * (penalty - penalty_bar)); "
            "low/high = center * exp(+/- z * sd_cell)"
        ),
        "winner": {
            "id": winner["id"],
            "name": winner["name"],
            "use_brand": winner["use_brand"],
            "use_condition": winner["use_condition"],
            "z": winner["z"],
            "metrics": winner["metrics"],
        },
        "coverage_target": args.coverage_target,
        "beta": beta_info,
        "bands": {
            "global_mu": bands["global_mu"],
            "global_sd": bands["global_sd"],
            "population_n": bands["population_n"],
            "brand_counts": bands["brand_counts"],
            "cells": {
                cell: {
                    "n": stats["n"],
                    "mu": stats["mu"],
                    "sd": stats["sd"],
                    "median_price": math.exp(stats["mu"]),
                }
                for cell, stats in bands["cells"].items()
            },
        },
        "major_brands": list(MAJOR_BRANDS),
        "other_cell": OTHER,
        "variants": [
            {
                "id": r["id"],
                "name": r["name"],
                "z": r["z"],
                "metrics": r["metrics"],
            }
            for r in results
        ],
        "labeled_n": len(labeled),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote model -> {args.out}")

    render_plot(
        labeled,
        bands,
        beta_info,
        winner["z"],
        use_condition=winner["use_condition"],
        out_path=args.plot,
    )


if __name__ == "__main__":
    main()
