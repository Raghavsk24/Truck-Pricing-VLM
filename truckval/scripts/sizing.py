"""How much data does each stage need? Measured on your own pipeline.

Three experiments, three different answers:

  --curve     total training rows vs. accuracy and interval width
  --cells     rows per make|model cell (this is the real driver)
  --conformal calibration slice size vs. stability of realized coverage

    python scripts/make_synth_comps.py --n 60000 --out data/comps_big.parquet
    python scripts/sizing.py --all
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from truckval.calibration import coverage, pinball_loss  # noqa: E402
from truckval.features import FEATURES  # noqa: E402
from truckval.pricing import QuantilePricer  # noqa: E402


def split(path: str):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    u = np.random.default_rng(0).random(len(df))
    return df[u < 0.72], df[(u >= 0.72) & (u < 0.86)], df[u >= 0.86]


def learning_curve(pool, calib, test, sizes) -> None:
    """Accuracy saturates early. Interval WIDTH is what keeps improving."""
    rows = test[FEATURES].to_dict("records")
    truth = test["sale_price"].to_numpy(float)
    print("=== total training rows ===")
    print(f"{'n_train':>8}{'pinball50':>11}{'medAE':>9}{'medAPE':>9}{'cover80':>9}{'relwidth':>10}")
    for n in sizes:
        if n > len(pool):
            break
        p = QuantilePricer.train(pool.sample(n=n, random_state=1), num_boost_round=700)
        p.calibrate(calib, alpha=0.20)
        pr = p.predict(rows)
        cov = coverage(truth, pr[:, 0], pr[:, 1], pr[:, 2], 0.80)
        ape = float(np.median(np.abs(truth - pr[:, 1]) / truth))
        print(f"{n:>8,}{pinball_loss(truth, pr[:, 1], .5):>11,.0f}"
              f"{cov.median_absolute_error:>9,.0f}{ape:>8.1%}"
              f"{cov.empirical:>9.1%}{cov.median_relative_width:>10.1%}")
    print("\nCoverage holds at every size because conformal calibration is\n"
          "unbiased regardless of n. Thin data does not make you WRONG, it\n"
          "makes you VAGUE — which is a launchable failure mode.")


def cell_depth(pool, calib, test) -> None:
    """The number that actually matters: realized sales per make|model."""
    pool = pool.assign(cell=pool["make"].astype(str) + "|" + pool["model"].astype(str))
    test = test.assign(cell=test["make"].astype(str) + "|" + test["model"].astype(str))
    cells = sorted(pool["cell"].unique())
    ladder = [10, 20, 40, 75, 150, 300, 600]
    cap_of = {c: ladder[i % len(ladder)] for i, c in enumerate(cells)}
    train = pd.concat([g.head(cap_of[c]) for c, g in pool.groupby("cell")])

    p = QuantilePricer.train(train, num_boost_round=500)
    p.calibrate(calib, alpha=0.20)
    rows = test[FEATURES].to_dict("records")
    truth = test["sale_price"].to_numpy(float)
    pr = p.predict(rows)
    ape = np.abs(truth - pr[:, 1]) / truth
    inside = (truth >= pr[:, 0]) & (truth <= pr[:, 2])
    relw = (pr[:, 2] - pr[:, 0]) / pr[:, 1]
    depth = test["cell"].map(cap_of).to_numpy()

    print(f"\n=== sales per make|model cell (total train {len(train):,}) ===")
    print(f"{'in cell':>9}{'n_test':>8}{'medAPE':>9}{'cover80':>9}{'relwidth':>10}")
    for d in sorted(set(cap_of.values())):
        m = depth == d
        if m.sum() < 40:
            continue
        print(f"{d:>9}{m.sum():>8}{np.median(ape[m]):>8.1%}"
              f"{inside[m].mean():>9.1%}{np.median(relw[m]):>10.1%}")
    print("\nBelow ~20 sales in a cell the model extrapolates and coverage\n"
          "breaks. Above it, adding rows to that cell buys almost nothing.\n"
          "Budget by cells covered, not by total rows.")


def conformal_size(pool, calib, test, reps: int = 60) -> None:
    """Calibration slice size is pure binomial noise on the coverage."""
    test = test.sample(n=min(3000, len(test)), random_state=2)
    rows = test[FEATURES].to_dict("records")
    truth = test["sale_price"].to_numpy(float)
    p = QuantilePricer.train(pool.sample(n=min(6000, len(pool)), random_state=1),
                             num_boost_round=400)
    raw = p._raw(rows)

    def cov_at(off: float) -> float:
        lo, hi = np.exp(raw[:, 0] - off), np.exp(raw[:, -1] + off)
        return float(((truth >= lo) & (truth <= hi)).mean())

    print("\n=== conformal calibration slice ===")
    print(f"{'n_calib':>9}{'mean':>8}{'sd':>7}{'p5':>8}{'p95':>8}{'sqrt(a(1-a)/n)':>16}")
    for nc in [30, 50, 100, 200, 500, 1000, 3000]:
        if nc > len(calib):
            break
        covs = []
        for t in range(reps):
            p.calibrate(calib.sample(n=nc, random_state=1000 + t), alpha=0.20)
            covs.append(cov_at(p.conformal_offset))
        c = np.array(covs)
        print(f"{nc:>9,}{c.mean():>8.1%}{c.std():>7.1%}{np.percentile(c, 5):>8.1%}"
              f"{np.percentile(c, 95):>8.1%}{np.sqrt(.2 * .8 / nc):>16.1%}")
    print("\nEmpirical sd tracks sqrt(a(1-a)/n) exactly. Same formula sizes\n"
          "your EVAL set: to measure 80% coverage to +/-2% you need ~1,600 rows.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--comps", default="data/comps_big.parquet")
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--cells", action="store_true")
    ap.add_argument("--conformal", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    pool, calib, test = split(a.comps)
    print(f"pool {len(pool):,}  calib {len(calib):,}  test {len(test):,}\n")
    if a.curve or a.all:
        learning_curve(pool, calib, test,
                       [250, 500, 1000, 2000, 4000, 8000, 16000, 32000])
    if a.cells or a.all:
        cell_depth(pool, calib, test)
    if a.conformal or a.all:
        conformal_size(pool, calib, test)
