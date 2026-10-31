"""How sure are the test results? Confidence ranges for the gain over the baselines, and whether
the prediction intervals hold.

Usage:
    python -m src.evaluate        # after python -m src.model for each target

Reads output/results/<target>_predictions.csv and <target>_model.json, and writes:
    <target>_gain_ci.csv     regional error of each model vs LightGBM, with a 95% range for the gap
    <target>_coverage.csv    share of actual values inside the INTERVAL prediction range, overall,
                             by practice size, by calendar month, and for the regional total

The gap is resampled by month: the test has only 30 months, and errors in neighbouring months
are related (a late or early season moves several months at once), so months are drawn in
blocks of BLOCK consecutive months rather than one by one. Both models are scored on the same
resampled months, so the range is for the difference itself.
"""

import json
import logging

import numpy as np
import pandas as pd

from . import feature as F
from . import model as M

log = logging.getLogger(__name__)

BLOCK = 3
RESAMPLES = 10_000
COMPARE = ["last_year_lift", "last_year", "pytorch"]


def monthly_ape(rows, target, models):
    """Absolute % error of the regional total, one value per month, for each model."""
    region = rows[[target, *models]].mul(rows["PATIENTS"] / 1000, axis=0).groupby(rows["YEAR_MONTH"]).sum()
    return region[models].sub(region[target], axis=0).abs().div(region[target], axis=0) * 100


def block_resample(n, block=BLOCK, resamples=RESAMPLES, seed=M.SEED):
    """Indices for a moving-block bootstrap of a series of length n: each row is one resample."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(resamples, int(np.ceil(n / block))))
    return (starts[:, :, None] + np.arange(block)).reshape(resamples, -1)[:, :n]


def gain_ci(group):
    """Regional error of LightGBM vs each comparison model, with 95% ranges for the gap."""
    target = f"per1000_{group}"
    rows = pd.read_csv(M.RESULTS_DIR / f"{group}_predictions.csv").dropna(subset=M.BASELINES)
    others = [m for m in COMPARE if m in rows]
    ape = monthly_ape(rows, target, ["lightgbm", *others])
    idx = block_resample(len(ape))
    out = []
    for m in others:
        lgb, other = ape["lightgbm"].to_numpy()[idx].mean(axis=1), ape[m].to_numpy()[idx].mean(axis=1)
        gap, rel = other - lgb, (lgb / other - 1) * 100
        out.append({
            "model": m,
            "lightgbm_pct": ape["lightgbm"].mean(), "other_pct": ape[m].mean(),
            "gap_points": ape[m].mean() - ape["lightgbm"].mean(),
            "gap_lo": np.percentile(gap, 2.5), "gap_hi": np.percentile(gap, 97.5),
            "change_pct": (ape["lightgbm"].mean() / ape[m].mean() - 1) * 100,
            "change_lo": np.percentile(rel, 2.5), "change_hi": np.percentile(rel, 97.5),
            "months_lightgbm_better": int((ape["lightgbm"] < ape[m]).sum()), "months": len(ape),
        })
    return pd.DataFrame(out).round(2)


def coverage(group):
    """Share of actual values inside the prediction range, by slice, plus the range's mean width."""
    target = f"per1000_{group}"
    rows = pd.read_csv(M.RESULTS_DIR / f"{group}_predictions.csv").dropna(subset=["lightgbm_lo", target])
    q = json.loads((M.RESULTS_DIR / f"{group}_model.json").read_text())["quantiles"]
    inside = rows[target].between(rows["lightgbm_lo"], rows["lightgbm_hi"])
    width = rows["lightgbm_hi"] - rows["lightgbm_lo"]
    size = pd.qcut(rows["PATIENTS"], 4, labels=["smallest 25%", "2nd", "3rd", "largest 25%"])
    out = [{"slice": "all practice-months", "by": "all", "coverage": inside.mean(), "width": width.mean(),
            "n": len(rows)}]
    for by, key in [("list size", size), ("calendar month", rows["YEAR_MONTH"] % 100)]:
        for k, part in inside.groupby(key, observed=True):
            out.append({"slice": str(k), "by": by, "coverage": part.mean(), "width": width[part.index].mean(),
                        "n": len(part)})
    weight = rows["PATIENTS"] / 1000
    region = rows[[target, "lightgbm"]].mul(weight, axis=0).groupby(rows["YEAR_MONTH"]).sum()
    lo, hi = region["lightgbm"] * (1 + q["region"][0]), region["lightgbm"] * (1 + q["region"][1])
    out.append({"slice": "regional total", "by": "region", "coverage": region[target].between(lo, hi).mean(),
                "width": ((hi - lo) / region["lightgbm"] * 100).mean(), "n": len(region)})
    return pd.DataFrame(out).round(3)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for group in F.GROUPS:
        if not (M.RESULTS_DIR / f"{group}_model.json").exists():
            log.warning("skipped %s: run python -m src.model --target %s first", group, group)
            continue
        gains, cover = gain_ci(group), coverage(group)
        gains.to_csv(M.RESULTS_DIR / f"{group}_gain_ci.csv", index=False)
        cover.to_csv(M.RESULTS_DIR / f"{group}_coverage.csv", index=False)
        print(f"\n{group}\n", gains.to_string(index=False), "\n",
              cover[cover["by"] != "calendar month"].to_string(index=False))


if __name__ == "__main__":
    main()
