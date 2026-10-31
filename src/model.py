"""Forecast each practice's monthly rate 3 months ahead and score it against simple baselines.

Usage:
    python -m src.model                       # antihistamines (per1000_antihist)
    python -m src.model --target nasal        # or eye
    python -m src.model --valid-only --change "what changed"
                                              # validation only (~1 min, no test metrics),
                                              # logged to output/experiments.csv (src.experiments)
    python -m src.model --params base         # ignore TUNED_PARAMS (latest-tune: last src.tune result)

Baselines:
    last_year       the rate in the same month last year (lag12)
    last_year_lift  this year's winter baseline + last year's lift over its winter
                    (winter_base + lag12_excess): last year's season on top of today's level

LightGBM variants (all on feature.FORECAST_FEATURES), differing in loss and in what is predicted:
    lgb_l1          rate, absolute loss. Aims at the median, which sits below the mean when
                    a few practice-months spike, so totals come out low.
    lgb_l2          rate, squared loss (aims at the mean)
    lgb_poisson     rate, Poisson loss (mean, multiplicative effects)
    lgb_l2_excess   rate minus the practice's winter base, squared loss. Trees can't predict
                    beyond the levels they were trained on; adding the base back lets a rising
                    level carry into the forecast.
    lgb_l2_ratio    rate / winter base, squared loss (the season as a multiple of the base)
The chosen variant is the one with the lowest CHOOSE_BY averaged over the validation years.
The test period is never used to choose.

    lightgbm        the chosen variant retrained every month, as it would be in use: the
                    forecast for month t is made by a model trained on targets up to t - 3 only
                    (the same months its features can see), with the tree count fixed from 2023.
                    The variants above are frozen at the end of 2023 and drift low as rates
                    keep rising; monthly retraining is what removes most of that bias.

Validation: one fold per year in VALID_YEARS (2021, 2022, 2023). Each fold trains on all earlier
years and scores that year; early stopping on the same year picks its tree count. One year is
only 12 monthly regional errors, so a single year's score moves by several tenths of a point
by chance; averaging three years, and asking a change to win in each (src.experiments), keeps
choices from chasing that noise.

Time split (target months):
    valid   2021, 2022, 2023   one fold each, trained on the years before
    refit   2014 - 2023        final model with the 2023 fold's tree count
    test    Mar 2024 - Aug 2026
The test starts in March 2024, not January: features are 3 months old, so the first forecast
the end-2023 model could really have made is for March 2024 (data to Dec 2023).
Every model is scored on the same rows: those where both baselines have a value.

Tried and rejected (output/experiments.csv has the scores), on the 3-year validation:
    blend with last_year_lift    weight per year chosen on the other two years. Antihistamines
                                 3.89 -> 3.83 (better in 2021 and 2023, worse in 2022), nasal and
                                 eye worse. The best weight swung from 0.3 to 1.0 between years.
    reconcile to region model    practice forecasts scaled to a ridge regression on the
                                 region-level series. Worse for antihistamines (4.56) and eye
                                 drops (10.7); nasal level (3.89, worse in 2023). About 70 region
                                 months is too little to beat the sum of the practice forecasts.
    average the variants         all five: antihistamines 3.89 -> 3.76 and nasal 3.91 -> 3.76,
                                 but all of it from 2022 (worse in 2021 and 2023); eye worse.
                                 Top three (picked on the other years): worse or level.
    Optuna tuning                25 trials on 2021-22 per target. Antihistamines first scored
                                 3.89 -> 3.66 (better every year) but that was one lucky seed:
                                 over 5 seeds, default 3.86 vs tuned 3.78 (-2%), and worse in
                                 2023 (3.46 -> 3.60), the year the search never saw. Nasal 3.91
                                 -> 3.73 but worse in 2023; eye drops: no trial beat the defaults.
Seed noise: with subsample < 1, changing the random seed alone moves one year's score by
0.1-0.2 and the 3-year mean by ~0.05. Check any candidate across several seeds before keeping it.

Results go to output/results/: <target>_valid.csv (per year and mean, the variant choice),
and unless --valid-only: <target>_scores.csv (test, one row per model), <target>_by_month.csv,
<target>_by_group.csv (by town and list size), <target>_monthly_bias.csv,
<target>_importance.csv and <target>_predictions.csv.
"""

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import experiments
from . import feature as F
from . import io

log = logging.getLogger(__name__)

SEED = 42
VALID_YEARS = (2021, 2022, 2023)
VALID_END = VALID_YEARS[-1] * 100 + 12
TEST_START, TEST_END = 202403, 202608
PEAK_MONTHS = (5, 6, 7)
CHOOSE_BY = "regional_mape_pct"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "output" / "results"

PARAMS = {
    "learning_rate": 0.03,
    "n_estimators": 5000,           # upper limit; early stopping picks the real number
    "num_leaves": 31,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "verbose": -1,
}

# Settings from src.tune that passed the keep rule (output/experiments.csv), per target; others
# use PARAMS. Empty: no tuning has survived a check across random seeds (see "Tried and rejected").
TUNED_PARAMS = {}


def params_for(group):
    return {**PARAMS, **TUNED_PARAMS.get(group, {})}

# name -> (LightGBM objective, what the model predicts)
VARIANTS = {
    "lgb_l1": ("regression_l1", "rate"),
    "lgb_l2": ("regression", "rate"),
    "lgb_poisson": ("poisson", "rate"),
    "lgb_l2_excess": ("regression", "excess"),
    "lgb_l2_ratio": ("regression", "ratio"),
}
BASELINES = ["last_year", "last_year_lift"]


def baselines(df, g):
    return pd.DataFrame({
        "last_year": df[f"{g}_lag12"],
        "last_year_lift": df[f"{g}_winter_base"] + df[f"{g}_lag12_excess"],
    }, index=df.index)


def _encode(df, target, g, form):
    base = df[f"{g}_winter_base"]
    return {"rate": df[target], "excess": df[target] - base, "ratio": df[target] / base}[form]


def _decode(pred, df, g, form):
    base = df[f"{g}_winter_base"].to_numpy()
    return {"rate": pred, "excess": pred + base, "ratio": pred * base}[form]


def _in_year(df, year):
    return df["YEAR_MONTH"] // 100 == year


def validate(df, target, g, objective, form, params=None, years=VALID_YEARS, features=F.FORECAST_FEATURES):
    """One fold per year in ``years``: train on earlier years, early-stop on and predict that year.
    Returns predictions for every validation row and the tree count of each fold."""
    y = _encode(df, target, g, form)
    usable = y.notna() & np.isfinite(y)  # excess/ratio need a winter base
    params = {**(params or PARAMS), "objective": objective}
    pred, trees = pd.Series(np.nan, index=df.index), {}
    for year in years:
        fit = usable & (df["YEAR_MONTH"] // 100 < year)
        valid = usable & _in_year(df, year)
        probe = lgb.LGBMRegressor(**params).fit(
            df.loc[fit, features], y[fit], eval_X=(df.loc[valid, features],), eval_y=(y[valid],),
            callbacks=[lgb.early_stopping(200, verbose=False)])
        rows = _in_year(df, year)
        pred[rows] = _decode(probe.predict(df.loc[rows, features]), df[rows], g, form)
        trees[year] = probe.best_iteration_
    return pred, trees


def refit(df, target, g, objective, form, n_trees, params=None, features=F.FORECAST_FEATURES):
    """Final frozen model on everything up to VALID_END."""
    y = _encode(df, target, g, form)
    train = y.notna() & np.isfinite(y) & (df["YEAR_MONTH"] <= VALID_END)
    params = {**(params or PARAMS), "objective": objective, "n_estimators": n_trees}
    return lgb.LGBMRegressor(**params).fit(df.loc[train, features], y[train])


def rolling_forecast(df, target, g, objective, form, n_trees, months, params=None,
                     features=F.FORECAST_FEATURES):
    """Forecast each month in ``months`` with a model trained on targets up to that month - HORIZON."""
    y = _encode(df, target, g, form)
    usable = y.notna() & np.isfinite(y)
    params = {**(params or PARAMS), "objective": objective, "n_estimators": n_trees}
    pred = pd.Series(np.nan, index=df.index)
    for ym in months:
        m = F._month_index(ym) - F.HORIZON
        origin = (m // 12) * 100 + m % 12 + 1
        train = usable & (df["YEAR_MONTH"] <= origin)
        model = lgb.LGBMRegressor(**params).fit(df.loc[train, features], y[train])
        rows = df["YEAR_MONTH"] == ym
        pred[rows] = _decode(model.predict(df.loc[rows, features]), df[rows], g, form)
    return pred


def _scores(rows, target, models):
    """MAE per practice-month (all, May-Jul, Aug-Apr), and regional error: the whole West Midlands'
    monthly items, predicted vs actual, as bias over the period and mean absolute % error per month."""
    month = rows["YEAR_MONTH"] % 100
    weight = rows["PATIENTS"] / 1000
    actual = (rows[target] * weight).groupby(rows["YEAR_MONTH"]).sum()
    out = []
    for m in models:
        err = (rows[m] - rows[target]).abs()
        pred = (rows[m] * weight).groupby(rows["YEAR_MONTH"]).sum()
        out.append({
            "model": m,
            "mae": err.mean(),
            "mae_may_jul": err[month.isin(PEAK_MONTHS)].mean(),
            "mae_aug_apr": err[~month.isin(PEAK_MONTHS)].mean(),
            "regional_bias_pct": (pred.sum() / actual.sum() - 1) * 100,
            "regional_mape_pct": ((pred - actual).abs() / actual).mean() * 100,
            "rows": len(rows),
        })
    return pd.DataFrame(out).round(3)


def _breakdowns(rows, target, models):
    err = rows[models].sub(rows[target], axis=0).abs()
    by_month = err.groupby(rows["YEAR_MONTH"] % 100).mean().round(3)
    by_month.index.name = "month"
    size = pd.qcut(rows["PATIENTS"], 4, labels=["smallest 25%", "2nd", "3rd", "largest 25%"])
    by_group = pd.concat([
        err.groupby(rows["TOWN"]).mean().assign(by="town"),
        err.groupby(size, observed=True).mean().assign(by="list size"),
    ]).round(3)
    weight = rows["PATIENTS"] / 1000
    region = rows[models + [target]].mul(weight, axis=0).groupby(rows["YEAR_MONTH"]).sum()
    monthly_bias = (region[models].div(region[target], axis=0).sub(1) * 100).round(1)
    return by_month, by_group, monthly_bias


def validation_scores(df, target, models):
    """_scores for each VALID_YEARS fold, plus their mean (year == "mean")."""
    per_year = pd.concat([_scores(df[_in_year(df, y)].dropna(subset=BASELINES), target, models).assign(year=y)
                          for y in VALID_YEARS])
    mean = per_year.drop(columns="year").groupby("model", sort=False).mean().round(3).reset_index()
    mean["rows"] = mean["rows"].round().astype(int)
    return pd.concat([per_year, mean.assign(year="mean")], ignore_index=True)


def run(group="antihist", valid_only=False, params=None):
    """Validate every variant and choose one; unless ``valid_only``, then refit and score the test.
    ``params`` defaults to params_for(group). Returns (valid_scores, chosen, test results or None)."""
    target = f"per1000_{group}"
    params = params or params_for(group)
    df = pd.read_csv(io.DATA_DIR / "processed" / "features.csv")
    df[BASELINES] = baselines(df, group)

    trees = {}
    for name, (objective, form) in VARIANTS.items():
        df[name], trees[name] = validate(df, target, group, objective, form, params)
        log.info("%s: trees per fold %s", name, trees[name])

    valid_scores = validation_scores(df, target, BASELINES + list(VARIANTS))
    mean = valid_scores[(valid_scores["year"] == "mean") & valid_scores["model"].isin(VARIANTS)]
    chosen = mean.sort_values(CHOOSE_BY)["model"].iloc[0]
    log.info("chosen on mean %s over %s: %s", CHOOSE_BY, VALID_YEARS, chosen)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    valid_scores.to_csv(RESULTS_DIR / f"{group}_valid.csv", index=False)
    if valid_only:
        return valid_scores, chosen, None

    test_rows = df["YEAR_MONTH"].between(TEST_START, TEST_END)
    fitted = {}
    for name, (objective, form) in VARIANTS.items():
        fitted[name] = refit(df, target, group, objective, form, trees[name][VALID_YEARS[-1]], params)
        df.loc[test_rows, name] = _decode(fitted[name].predict(df.loc[test_rows, F.FORECAST_FEATURES]),
                                          df[test_rows], group, form)
    months = sorted(df.loc[test_rows, "YEAR_MONTH"].unique())
    df["lightgbm"] = rolling_forecast(df, target, group, *VARIANTS[chosen],
                                      trees[chosen][VALID_YEARS[-1]], months, params)
    test = df[test_rows].dropna(subset=BASELINES).copy()
    models = BASELINES + list(VARIANTS) + ["lightgbm"]
    scores = _scores(test, target, models)
    scores.insert(1, "chosen", scores["model"] == chosen)
    by_month, by_group, monthly_bias = _breakdowns(test, target, BASELINES + ["lightgbm"])
    importance = pd.Series(fitted[chosen].booster_.feature_importance("gain"), index=F.FORECAST_FEATURES)
    importance = (importance / importance.sum() * 100).round(1).sort_values(ascending=False)

    scores.to_csv(RESULTS_DIR / f"{group}_scores.csv", index=False)
    by_month.to_csv(RESULTS_DIR / f"{group}_by_month.csv")
    by_group.to_csv(RESULTS_DIR / f"{group}_by_group.csv")
    monthly_bias.to_csv(RESULTS_DIR / f"{group}_monthly_bias.csv")
    importance.rename("gain_pct").to_csv(RESULTS_DIR / f"{group}_importance.csv")
    df.loc[test_rows, ["YEAR_MONTH", "PRACTICE_CODE", "TOWN", "PATIENTS", target] + models].to_csv(
        RESULTS_DIR / f"{group}_predictions.csv", index=False)
    log.info("results in %s", RESULTS_DIR)
    return valid_scores, chosen, (scores, monthly_bias, importance)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", choices=F.GROUPS, default="antihist")
    parser.add_argument("--valid-only", action="store_true",
                        help="stop after validation: no refit, no test metrics")
    parser.add_argument("--change", help="log this run to output/experiments.csv with this description")
    parser.add_argument("--note", default="")
    parser.add_argument("--params", choices=["kept", "base", "latest-tune"], default="kept",
                        help="kept: params_for(target) (TUNED_PARAMS where kept, else PARAMS); "
                             "base: PARAMS; latest-tune: the last src.tune result for this target")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    params = {"kept": params_for(args.target), "base": PARAMS}.get(args.params)
    if args.params == "latest-tune":
        params = {**PARAMS, **json.loads((RESULTS_DIR / f"{args.target}_tuned_params.json").read_text())}
    valid_scores, chosen, test = run(args.target, args.valid_only, params)
    print("Validation:", valid_scores.to_string(index=False), sep="\n")
    if args.change:
        row = valid_scores[valid_scores["model"] == chosen].set_index("year")[CHOOSE_BY]
        label = chosen + ("" if params == PARAMS else "+tuned")
        kept = experiments.log_run(args.target, label, args.change,
                                   {y: row[y] for y in VALID_YEARS}, note=args.note)
        print(f"\nlogged: {label}, mean {CHOOSE_BY} {row['mean']:.3f}, kept={kept}")
    if test is not None:
        scores, monthly_bias, importance = test
        print("\nTest:", scores.to_string(index=False), "Regional bias by test month (%):",
              monthly_bias.to_string(), importance.head(10).to_string(), sep="\n\n")


if __name__ == "__main__":
    main()
