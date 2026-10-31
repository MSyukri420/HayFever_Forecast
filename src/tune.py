"""Tune LightGBM settings with Optuna for one target's chosen variant.

Usage:
    python -m src.tune --target antihist              # 25 trials, ~10 min
    python -m src.tune --target eye --trials 40
    python -m src.model --target antihist --valid-only --params tuned --change "optuna settings"

The search only sees the TUNE_YEARS folds (2021, 2022). 2023 is left out, so when the tuned
settings are judged by src.model on all three validation years (and must win in each, see
src.experiments), one of those years played no part in the search. Every trial is another
try against the same months, so trials are kept few; the best trial's score is optimistic.

The variant tuned is the one src.model chooses with default settings. The tree count is not
tuned: early stopping picks it in every fold, as in src.model.
Saves output/results/<target>_tuned_params.json (read by src.model --params tuned) and
<target>_tuning.csv (every trial).
"""

import argparse
import json
import logging

import optuna
import pandas as pd

from . import feature as F
from . import io
from . import model as M

log = logging.getLogger(__name__)

TUNE_YEARS = (2021, 2022)
TRIALS = 25


def space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def tune(group, trials=TRIALS):
    target = f"per1000_{group}"
    df = pd.read_csv(io.DATA_DIR / "processed" / "features.csv")
    df[M.BASELINES] = M.baselines(df, group)
    variant = M.experiments.best_run(group)
    variant = variant["variant"] if variant is not None and variant["variant"] in M.VARIANTS else None
    if variant is None:
        raise SystemExit(f"log a reference run first: python -m src.model --target {group} --valid-only "
                         f"--change reference")
    objective, form = M.VARIANTS[variant]
    log.info("tuning %s for %s on %s", variant, group, TUNE_YEARS)

    def score(trial):
        params = {**M.PARAMS, **space(trial)}
        df["tuned"], _ = M.validate(df, target, group, objective, form, params, years=TUNE_YEARS)
        rows = df[(df["YEAR_MONTH"] // 100).isin(TUNE_YEARS)].dropna(subset=M.BASELINES)
        return sum(M._scores(rows[rows["YEAR_MONTH"] // 100 == y], target, ["tuned"])[M.CHOOSE_BY].iloc[0]
                   for y in TUNE_YEARS) / len(TUNE_YEARS)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=M.SEED))
    study.enqueue_trial({k: M.PARAMS[k] for k in space(optuna.trial.FixedTrial(M.PARAMS))})  # defaults first
    study.optimize(score, n_trials=trials)

    M.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (M.RESULTS_DIR / f"{group}_tuned_params.json").write_text(json.dumps(study.best_params, indent=2))
    study.trials_dataframe().to_csv(M.RESULTS_DIR / f"{group}_tuning.csv", index=False)
    log.info("defaults %.3f -> best %.3f on %s: %s", study.trials[0].value, study.best_value,
             TUNE_YEARS, study.best_params)
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", choices=F.GROUPS, default="antihist")
    parser.add_argument("--trials", type=int, default=TRIALS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tune(args.target, args.trials)


if __name__ == "__main__":
    main()
