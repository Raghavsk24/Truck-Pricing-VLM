"""Train the quantile pricing model, conformalize it, report held-out coverage.

Three-way split, and the middle slice matters: conformal calibration must be
fitted on data used for neither training nor evaluation, or the guarantee is
worthless and you have simply overfitted your error bars.

    python scripts/train_pricing.py --comps data/comps.parquet --out data/pricer.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from truckval.calibration import coverage, pinball_loss  # noqa: E402
from truckval.features import FEATURES  # noqa: E402
from truckval.pricing import QuantilePricer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comps", default="data/comps.parquet")
    ap.add_argument("--out", default="data/pricer.pkl")
    ap.add_argument("--rounds", type=int, default=700)
    ap.add_argument("--alpha", type=float, default=0.20)
    a = ap.parse_args()

    df = (pd.read_parquet(a.comps) if a.comps.endswith(".parquet")
          else pd.read_csv(a.comps))

    rng = np.random.default_rng(0)
    u = rng.random(len(df))
    train, calib, test = df[u < 0.65], df[(u >= 0.65) & (u < 0.82)], df[u >= 0.82]
    print(f"train {len(train):,}   calib {len(calib):,}   test {len(test):,}\n")

    pricer = QuantilePricer.train(train, num_boost_round=a.rounds)

    rows = test[FEATURES].to_dict("records")
    truth = test["sale_price"].to_numpy(dtype=float)

    before = pricer.predict(rows, conformal=False)
    print("before conformal calibration")
    for i, q in enumerate(pricer.quantile_levels):
        print(f"  pinball@{q:.2f}  {pinball_loss(truth, before[:, i], q):>9,.0f}")
    print(coverage(truth, before[:, 0], before[:, 1], before[:, 2],
                   nominal=1 - a.alpha))

    offset = pricer.calibrate(calib, alpha=a.alpha)
    after = pricer.predict(rows, conformal=True)
    print(f"\nafter conformal calibration (offset {offset:.4f} in log space, "
          f"x{np.exp(offset):.3f} on price)")
    print(coverage(truth, after[:, 0], after[:, 1], after[:, 2],
                   nominal=1 - a.alpha))

    pricer.save(a.out)
    print(f"\nsaved -> {a.out}")
    print(
        "\nThis is MODEL coverage on fully-observed rows. The number that\n"
        "decides whether anyone trusts you twice comes from\n"
        "scripts/eval_calibration.py, which runs the same check through the\n"
        "uncertainty pass with fields deliberately hidden."
    )


if __name__ == "__main__":
    main()
