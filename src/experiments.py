"""Experiment log for the tuning loop.

Each row records one change and its validation scores (CHOOSE_BY in src.model, one per
VALID_YEARS fold, lower is better), so runs can be compared and dead ends are not retried.
Test-period metrics are deliberately not stored here; they are reported once at the end
of a loop.

Usage:
    python -m src.model --target antihist --valid-only --change "add town-level lag"
                                                   # runs validation and logs it here
    python -m src.experiments --summary            # best kept run per target
    python -m src.experiments --summary --all      # every logged run

    from src.experiments import log_run, best_run
    log_run("antihist", "lgb_poisson", "add town-level lag", {2021: 4.10, 2022: 3.71, 2023: 3.55})

Keep rule (``is_improvement``): a change is kept only if it beats the best kept run for that
target in every validation year AND improves the mean by more than MIN_GAIN (relative).
The first run logged for a target is kept as the reference.

Why so strict (measured on antihistamines, 2021-2023): one year's score is the mean of only
12 monthly regional errors, with a standard error of 0.7-0.9 points on a score of about 3.9.
Comparing two models month by month is steadier, but a one-year difference still has a
standard error of 0.1-0.5 points, and the 3-year mean about 0.15 (~4%). Variants that look
different in one year swap order in the next (l2_excess vs poisson: +0.17, -0.53, +0.01).
MIN_GAIN is set to about one standard error of that 3-year difference, and needing every
year to agree cuts chance "wins" further. Even so, a long loop will keep the odd change by
luck, so keep runs few and confirm anything kept on the test once, at the end.
"""

import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import io

log = logging.getLogger(__name__)

LOG_PATH = Path(__file__).resolve().parents[1] / "output" / "experiments.csv"
MIN_GAIN = 0.04  # relative improvement of the mean needed to keep a change (4%, ~1 standard error)


def _commit():
    """Short hash of HEAD, with "-dirty" if tracked files have uncommitted changes."""
    root = LOG_PATH.parents[1]
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=root).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _year_cols(df):
    return sorted(c for c in df.columns if c.startswith("val_") and c[4:].isdigit())


def load_log():
    return pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame()


def best_run(target):
    """Best kept run for a target (lowest val_mean), or None if nothing is kept yet."""
    df = load_log()
    if df.empty:
        return None
    df = df[(df["target"] == target) & df["kept"].astype(bool)]
    return None if df.empty else df.loc[df["val_mean"].idxmin()]


def is_improvement(target, year_scores):
    """True if ``year_scores`` ({year: score}) beats the best kept run in every year and
    improves its mean by more than MIN_GAIN."""
    best = best_run(target)
    if best is None:
        return True
    mean = sum(year_scores.values()) / len(year_scores)
    every_year = all(score < best[f"val_{year}"] for year, score in year_scores.items())
    return every_year and mean < best["val_mean"] * (1 - MIN_GAIN)


def log_run(target, variant, change, year_scores, kept=None, note=""):
    """Append one run to the log. If ``kept`` is None it is decided by ``is_improvement``.
    Returns whether the run was kept."""
    if kept is None:
        kept = is_improvement(target, year_scores)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "commit": _commit(),
        "target": target,
        "variant": variant,
        "change": change,
        **{f"val_{y}": round(s, 3) for y, s in year_scores.items()},
        "val_mean": round(sum(year_scores.values()) / len(year_scores), 3),
        "kept": kept,
        "note": note,
    }
    io.save_dataset(pd.concat([load_log(), pd.DataFrame([row])], ignore_index=True), LOG_PATH)
    log.info("%s | %s | mean %.3f | kept=%s", target, change, row["val_mean"], kept)
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", action="store_true", help="best kept run per target (default)")
    parser.add_argument("--all", action="store_true", help="show every run")
    args = parser.parse_args()

    df = load_log()
    if df.empty:
        print("No runs logged yet.")
        return
    if args.all:
        print(df.to_string(index=False))
        return
    kept = df[df["kept"].astype(bool)]
    best = kept.loc[kept.groupby("target")["val_mean"].idxmin()]
    print(best[["target", "variant", "change", *_year_cols(df), "val_mean", "commit"]].to_string(index=False))
    print(f"\n{len(df)} runs logged, {len(kept)} kept.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    main()
