"""Stage 3. Baseline market price. Never touches photos.

Quantile regression, not point prediction. Three LightGBM models at alpha
0.1 / 0.5 / 0.9 give a range as native output instead of a point estimate with
a made-up plus-or-minus bolted on afterwards.

Train on realized prices. Ritchie Bros unreserved auction results are the good
source here: no reserve floor means the hammer price is a real clearing price.
Ask prices from classified listings are aspirational and will bias you high.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .features import CATEGORICAL, FEATURES, NUMERIC, align_categories, rows_to_frame

QUANTILES = (0.10, 0.50, 0.90)

LGB_PARAMS = dict(
    objective="quantile",
    metric="quantile",
    learning_rate=0.05,
    num_leaves=48,
    min_data_in_leaf=25,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    verbose=-1,
)


@dataclass
class QuantilePricer:
    models: dict[float, lgb.Booster] = field(default_factory=dict)
    categories: dict[str, list] = field(default_factory=dict)
    log_target: bool = True
    conformal_offset: float = 0.0  # additive in model space; see calibrate()

    # ---------------------------------------------------------------- train
    @classmethod
    def train(
        cls,
        comps: pd.DataFrame,
        target: str = "sale_price",
        quantiles: Sequence[float] = QUANTILES,
        num_boost_round: int = 700,
        log_target: bool = True,
    ) -> "QuantilePricer":
        missing = [c for c in FEATURES if c not in comps.columns]
        if missing:
            raise ValueError(f"comps table is missing feature columns: {missing}")

        X = comps[FEATURES].copy()
        for c in CATEGORICAL:
            X[c] = X[c].astype(str)
        categories = {c: sorted(set(X[c].unique().tolist()) | {"unknown"})
                      for c in CATEGORICAL}
        X = align_categories(X, categories)
        for c in NUMERIC:
            X[c] = pd.to_numeric(X[c], errors="coerce")

        y = comps[target].to_numpy(dtype=float)
        if log_target:
            y = np.log(np.clip(y, 1.0, None))

        models: dict[float, lgb.Booster] = {}
        for q in quantiles:
            ds = lgb.Dataset(X, label=y, categorical_feature=CATEGORICAL,
                             free_raw_data=False)
            models[q] = lgb.train({**LGB_PARAMS, "alpha": q}, ds,
                                  num_boost_round=num_boost_round)
        return cls(models=models, categories=categories, log_target=log_target)

    # -------------------------------------------------------------- predict
    def _raw(self, rows: list[dict]) -> np.ndarray:
        X = rows_to_frame(rows)
        X = align_categories(X, self.categories)
        for c in NUMERIC:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        qs = sorted(self.models)
        return np.column_stack([self.models[q].predict(X) for q in qs])

    def predict(self, rows: list[dict], conformal: bool = True) -> np.ndarray:
        """Returns an (n_rows, n_quantiles) array of prices, sorted ascending."""
        preds = self._raw(rows)
        if conformal and self.conformal_offset:
            preds[:, 0] -= self.conformal_offset
            preds[:, -1] += self.conformal_offset
        if self.log_target:
            preds = np.exp(preds)
        # Quantile crossing is common with independently fitted models.
        return np.sort(preds, axis=1)

    # ------------------------------------------------------------ calibrate
    def calibrate(
        self,
        holdout: pd.DataFrame,
        target: str = "sale_price",
        alpha: float = 0.20,
    ) -> float:
        """Conformalized quantile regression (Romano et al., 2019).

        Gradient-boosted quantile models come out systematically under-dispersed
        — the outer quantiles get shrunk toward the median, so a nominal 80%
        interval covers something more like 65% on held-out data. Tuning does
        not reliably fix this; a conformal correction does, with a finite-sample
        marginal coverage guarantee and no distributional assumptions.

        Fit on data used for NEITHER training nor final evaluation. In log
        space the offset is additive, which makes the widening multiplicative
        on price — the right shape when a $200k tractor and a $9k box truck
        share one model.
        """
        rows = holdout[FEATURES].to_dict("records")
        raw = self._raw(rows)
        y = holdout[target].to_numpy(dtype=float)
        if self.log_target:
            y = np.log(np.clip(y, 1.0, None))

        lo, hi = raw[:, 0], raw[:, -1]
        scores = np.maximum(lo - y, y - hi)
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        self.conformal_offset = float(max(0.0, np.quantile(scores, level)))
        return self.conformal_offset

    @property
    def quantile_levels(self) -> list[float]:
        return sorted(self.models)

    # ------------------------------------------------------------- persist
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "models": {q: m.model_to_string() for q, m in self.models.items()},
            "categories": self.categories,
            "log_target": self.log_target,
            "conformal_offset": self.conformal_offset,
        }
        path.write_bytes(pickle.dumps(blob))

    @classmethod
    def load(cls, path: str | Path) -> "QuantilePricer":
        blob = pickle.loads(Path(path).read_bytes())
        return cls(
            models={float(q): lgb.Booster(model_str=s)
                    for q, s in blob["models"].items()},
            categories=blob["categories"],
            log_target=blob["log_target"],
            conformal_offset=blob.get("conformal_offset", 0.0),
        )


# --------------------------------------------------------------------------
# Comps retrieval — the explainable fallback, and a sanity check on the model
# --------------------------------------------------------------------------


@dataclass
class CompsIndex:
    """Nearest realized sales within the same make/model bucket.

    Worth keeping even after the GBM works: when a seller asks 'why that
    number', showing five actual trucks that sold is more persuasive than any
    feature importance plot.
    """

    frame: pd.DataFrame
    _index: dict[str, tuple[NearestNeighbors, np.ndarray, pd.Index]] = field(
        default_factory=dict
    )
    numeric_cols: tuple[str, ...] = ("age_years", "log_odometer_km", "box_length_ft")

    def build(self) -> "CompsIndex":
        for key, grp in self.frame.groupby(
            self.frame["make"].astype(str) + "|" + self.frame["model"].astype(str)
        ):
            mat = grp[list(self.numeric_cols)].to_numpy(dtype=float)
            mat = np.nan_to_num(mat, nan=np.nanmean(mat) if mat.size else 0.0)
            if len(mat) < 3:
                continue
            scale = mat.std(axis=0)
            scale[scale == 0] = 1.0
            nn = NearestNeighbors(n_neighbors=min(8, len(mat))).fit(mat / scale)
            self._index[key] = (nn, scale, grp.index)
        return self

    def query(self, row: dict, k: int = 5) -> pd.DataFrame:
        key = f"{row.get('make')}|{row.get('model')}"
        entry = self._index.get(key)
        if entry is None:
            return self.frame.iloc[0:0]
        nn, scale, idx = entry
        vec = np.array(
            [[float(row.get(c) or 0.0) for c in self.numeric_cols]], dtype=float
        )
        _, ind = nn.kneighbors(vec / scale, n_neighbors=min(k, len(idx)))
        return self.frame.loc[idx[ind[0]]]

    def price_range(self, row: dict, k: int = 8) -> Optional[tuple[float, float, float]]:
        hits = self.query(row, k=k)
        if len(hits) < 3:
            return None
        p = hits["sale_price"].to_numpy(dtype=float)
        return tuple(np.percentile(p, [10, 50, 90]))  # type: ignore[return-value]
