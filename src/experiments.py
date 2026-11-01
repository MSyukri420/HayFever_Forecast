"""Log of validation runs, the rule for keeping a change, and Optuna tuning on the 2021-22 folds.

    python -m src.experiments --summary [--all]
    python -m src.experiments --tune antihist
"""

import argparse
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import feature as F
from . import io

log = logging.getLogger(__name__)

LOG_PATH = Path(__file__).resolve().parents[1] / "output" / "experiments.csv"
MIN_GAIN = 0.04  # about one standard error of the 3-year mean
TUNE_YEARS = (2021, 2022)
TRIALS = 25


def _commit():
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
    df = load_log()
    if df.empty:
        return None
    df = df[(df["target"] == target) & df["kept"].astype(bool)]
    return None if df.empty else df.loc[df["val_mean"].idxmin()]


def is_improvement(target, year_scores):
    # keep only if it wins every year and the mean by more than MIN_GAIN
    best = best_run(target)
    if best is None:
        return True
    mean = sum(year_scores.values()) / len(year_scores)
    every_year = all(score < best[f"val_{year}"] for year, score in year_scores.items())
    return every_year and mean < best["val_mean"] * (1 - MIN_GAIN)


def log_run(target, variant, change, year_scores, kept=None, note=""):
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


def _space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def tune(group, trials=TRIALS):
    import optuna

    from . import model as M  # model imports this module

    target = f"per1000_{group}"
    df = pd.read_csv(io.DATA_DIR / "processed" / "features.csv")
    df[M.BASELINES] = M.baselines(df, group)
    variant = best_run(group)
    variant = variant["variant"] if variant is not None and variant["variant"] in M.VARIANTS else None
    if variant is None:
        raise SystemExit(f"log a reference run first: python -m src.model --target {group} --valid-only "
                         f"--change reference")
    objective, form = M.VARIANTS[variant]
    log.info("tuning %s for %s on %s", variant, group, TUNE_YEARS)

    def score(trial):
        params = {**M.PARAMS, **_space(trial)}
        df["tuned"], _ = M.validate(df, target, group, objective, form, params, years=TUNE_YEARS)
        rows = df[(df["YEAR_MONTH"] // 100).isin(TUNE_YEARS)].dropna(subset=M.BASELINES)
        return sum(M._scores(rows[rows["YEAR_MONTH"] // 100 == y], target, ["tuned"])[M.CHOOSE_BY].iloc[0]
                   for y in TUNE_YEARS) / len(TUNE_YEARS)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=M.SEED))
    study.enqueue_trial({k: M.PARAMS[k] for k in _space(optuna.trial.FixedTrial(M.PARAMS))})
    study.optimize(score, n_trials=trials)

    M.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (M.RESULTS_DIR / f"{group}_tuned_params.json").write_text(json.dumps(study.best_params, indent=2))
    study.trials_dataframe().to_csv(M.RESULTS_DIR / f"{group}_tuning.csv", index=False)
    log.info("defaults %.3f -> best %.3f on %s: %s", study.trials[0].value, study.best_value,
             TUNE_YEARS, study.best_params)
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", action="store_true", help="best kept run per target (default)")
    parser.add_argument("--all", action="store_true", help="show every run")
    parser.add_argument("--tune", choices=F.GROUPS, help="run the Optuna search for this target")
    parser.add_argument("--trials", type=int, default=TRIALS)
    args = parser.parse_args()

    if args.tune:
        tune(args.tune, args.trials)
        return
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
