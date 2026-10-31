"""Forecast each practice's monthly rate 3 months ahead and score it against simple baselines.

Usage:
    python -m src.model                       # antihistamines (per1000_antihist)
    python -m src.model --target nasal        # or eye

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
The chosen variant is the one with the lowest regional error on the 2023 validation year
(CHOOSE_BY). The test period is never used to choose.

    lightgbm        the chosen variant retrained every month, as it would be in use: the
                    forecast for month t is made by a model trained on targets up to t - 3 only
                    (the same months its features can see), with the tree count fixed from 2023.
                    The variants above are frozen at the end of 2023 and drift low as rates
                    keep rising; monthly retraining is what removes most of that bias.

Time split (target months):
    fit     2014 - 2022     training
    valid   2023            early stopping picks the number of trees; variants are compared here
    refit   2014 - 2023     final model with that number of trees
    test    Mar 2024 - Aug 2026
The test starts in March 2024, not January: features are 3 months old, so the first forecast
the end-2023 model could really have made is for March 2024 (data to Dec 2023).
Every model is scored on the same test rows: those where both baselines have a value.

Results go to output/results/: <target>_scores.csv (test, one row per model), <target>_valid.csv
(2023, the variant choice), <target>_by_month.csv, <target>_by_group.csv (by town and list size),
<target>_monthly_bias.csv, <target>_importance.csv and <target>_predictions.csv.
"""

import argparse
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import feature as F
from . import io

log = logging.getLogger(__name__)

SEED = 42
FIT_END, VALID_END = 202212, 202312
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


def fit_lightgbm(df, target, g, objective, form, features=F.FORECAST_FEATURES):
    """Early-stop on 2023, then refit on 2014-2023 with the chosen number of trees.
    Returns the refit model and the 2023 predictions of the early-stopping model."""
    y = _encode(df, target, g, form)
    usable = y.notna() & np.isfinite(y)  # excess/ratio need a winter base
    fit = usable & (df["YEAR_MONTH"] <= FIT_END)
    valid = usable & df["YEAR_MONTH"].between(FIT_END + 1, VALID_END)
    params = {**PARAMS, "objective": objective}
    probe = lgb.LGBMRegressor(**params).fit(
        df.loc[fit, features], y[fit], eval_X=(df.loc[valid, features],), eval_y=(y[valid],),
        callbacks=[lgb.early_stopping(200, verbose=False)])
    n_trees = probe.best_iteration_

    train = usable & (df["YEAR_MONTH"] <= VALID_END)
    model = lgb.LGBMRegressor(**{**params, "n_estimators": n_trees}).fit(df.loc[train, features], y[train])
    in_2023 = df["YEAR_MONTH"].between(FIT_END + 1, VALID_END)
    valid_pred = pd.Series(_decode(probe.predict(df.loc[in_2023, features]), df[in_2023], g, form),
                           index=df.index[in_2023])
    return model, n_trees, valid_pred


def rolling_forecast(df, target, g, objective, form, n_trees, months, features=F.FORECAST_FEATURES):
    """Forecast each month in ``months`` with a model trained on targets up to that month - HORIZON."""
    y = _encode(df, target, g, form)
    usable = y.notna() & np.isfinite(y)
    params = {**PARAMS, "objective": objective, "n_estimators": n_trees}
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


def run(group="antihist"):
    target = f"per1000_{group}"
    df = pd.read_csv(io.DATA_DIR / "processed" / "features.csv")
    df[BASELINES] = baselines(df, group)
    test_rows = df["YEAR_MONTH"].between(TEST_START, TEST_END)
    valid_rows = df["YEAR_MONTH"].between(FIT_END + 1, VALID_END)

    fitted, trees = {}, {}
    for name, (objective, form) in VARIANTS.items():
        model, n_trees, valid_pred = fit_lightgbm(df, target, group, objective, form)
        df.loc[valid_rows, f"{name}_valid"] = valid_pred
        df.loc[test_rows, name] = _decode(model.predict(df.loc[test_rows, F.FORECAST_FEATURES]),
                                          df[test_rows], group, form)
        fitted[name], trees[name] = model, n_trees
        log.info("%s: %d trees", name, n_trees)

    # choose on 2023 only, using the early-stopping models (trained to 2022)
    valid = df[valid_rows].dropna(subset=BASELINES).copy()
    for name in VARIANTS:
        valid[name] = valid[f"{name}_valid"]
    valid_scores = _scores(valid, target, BASELINES + list(VARIANTS))
    chosen = valid_scores[valid_scores["model"].isin(VARIANTS)].sort_values(CHOOSE_BY)["model"].iloc[0]
    log.info("chosen on 2023 %s: %s; retraining it monthly over the test period", CHOOSE_BY, chosen)

    months = sorted(df.loc[test_rows, "YEAR_MONTH"].unique())
    df["lightgbm"] = rolling_forecast(df, target, group, *VARIANTS[chosen], trees[chosen], months)
    test = df[test_rows].dropna(subset=BASELINES).copy()
    models = BASELINES + list(VARIANTS) + ["lightgbm"]
    scores = _scores(test, target, models)
    scores.insert(1, "chosen", scores["model"] == chosen)
    by_month, by_group, monthly_bias = _breakdowns(test, target, BASELINES + ["lightgbm"])
    importance = pd.Series(fitted[chosen].booster_.feature_importance("gain"), index=F.FORECAST_FEATURES)
    importance = (importance / importance.sum() * 100).round(1).sort_values(ascending=False)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(RESULTS_DIR / f"{group}_scores.csv", index=False)
    valid_scores.to_csv(RESULTS_DIR / f"{group}_valid.csv", index=False)
    by_month.to_csv(RESULTS_DIR / f"{group}_by_month.csv")
    by_group.to_csv(RESULTS_DIR / f"{group}_by_group.csv")
    monthly_bias.to_csv(RESULTS_DIR / f"{group}_monthly_bias.csv")
    importance.rename("gain_pct").to_csv(RESULTS_DIR / f"{group}_importance.csv")
    df.loc[test_rows, ["YEAR_MONTH", "PRACTICE_CODE", "TOWN", "PATIENTS", target] + models].to_csv(
        RESULTS_DIR / f"{group}_predictions.csv", index=False)
    log.info("results in %s", RESULTS_DIR)
    return valid_scores, scores, monthly_bias, importance


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", choices=F.GROUPS, default="antihist")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    valid_scores, scores, monthly_bias, importance = run(args.target)
    print("2023 validation:", valid_scores.to_string(index=False), "Test:", scores.to_string(index=False),
          "Regional bias by test month (%):", monthly_bias.to_string(), importance.head(10).to_string(),
          sep="\n\n")


if __name__ == "__main__":
    main()
