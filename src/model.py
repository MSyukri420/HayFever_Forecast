"""Forecast each practice's monthly rate 3 months ahead and compare it with simple baselines.

    python -m src.model --target antihist      # or nasal, eye
    python -m src.model --valid-only --change "what changed"
    python -m src.model --evaluate-only        # redo the bootstrap ranges and coverage
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
from . import nn

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

# tuned settings that passed the keep rule (none so far)
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
INTERVAL = 0.8  # coverage of the prediction intervals
BLOCK, RESAMPLES = 3, 10_000  # block bootstrap of the test months
COMPARE = ["last_year_lift", "last_year", "pytorch"]


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
    """One fold per year, trained on the years before it."""
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
    y = _encode(df, target, g, form)
    train = y.notna() & np.isfinite(y) & (df["YEAR_MONTH"] <= VALID_END)
    params = {**(params or PARAMS), "objective": objective, "n_estimators": n_trees}
    return lgb.LGBMRegressor(**params).fit(df.loc[train, features], y[train])


def rolling_forecast(df, target, g, objective, form, n_trees, months, params=None,
                     features=F.FORECAST_FEATURES):
    # retrain for every month on what was known HORIZON months before
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


def _poisson_scale(pred, patients):
    # rough Poisson spread, so small practices get wider ranges
    return np.sqrt(np.maximum(pred, 0.5) * 1000 / patients)


def interval_quantiles(rows, target, pred):
    """Split-conformal quantiles from the validation forecasts: per calendar month, and regional."""
    lo, hi = (1 - INTERVAL) / 2, (1 + INTERVAL) / 2
    rows = rows.dropna(subset=[pred, target])
    score = (rows[target] - rows[pred]) / _poisson_scale(rows[pred], rows["PATIENTS"])
    practice = score.groupby(rows["YEAR_MONTH"] % 100).quantile([lo, hi]).unstack()
    region = rows[[target, pred]].mul(rows["PATIENTS"] / 1000, axis=0).groupby(rows["YEAR_MONTH"]).sum()
    rel = region[target] / region[pred] - 1
    return {"practice": {int(m): [round(a, 4), round(b, 4)] for m, (a, b) in practice.iterrows()},
            "region": [round(rel.quantile(lo), 4), round(rel.quantile(hi), 4)]}


def add_intervals(rows, pred, quantiles, patients="PATIENTS"):
    by_month = {int(m): q for m, q in quantiles["practice"].items()}  # JSON keys are strings
    q = np.array([by_month[m] for m in rows["YEAR_MONTH"] % 100])
    scale = _poisson_scale(rows[pred], rows[patients])
    return (rows[pred] + q[:, 0] * scale).clip(lower=0), rows[pred] + q[:, 1] * scale


def shap_in_rate_units(model, rows, g, form, features=F.FORECAST_FEATURES):
    """TreeSHAP values in items per 1,000, adding up to the forecast."""
    contrib = pd.DataFrame(model.booster_.predict(rows[features], pred_contrib=True),
                           columns=features + ["start"], index=rows.index)
    base = rows[f"{g}_winter_base"]
    if form == "ratio":
        contrib = contrib.mul(base, axis=0)
    elif form == "excess":
        contrib["start"] += base
    return contrib


def _scores(rows, target, models):
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
    per_year = pd.concat([_scores(df[_in_year(df, y)].dropna(subset=BASELINES), target, models).assign(year=y)
                          for y in VALID_YEARS])
    mean = per_year.drop(columns="year").groupby("model", sort=False).mean().round(3).reset_index()
    mean["rows"] = mean["rows"].round().astype(int)
    return pd.concat([per_year, mean.assign(year="mean")], ignore_index=True)


def monthly_ape(rows, target, models):
    region = rows[[target, *models]].mul(rows["PATIENTS"] / 1000, axis=0).groupby(rows["YEAR_MONTH"]).sum()
    return region[models].sub(region[target], axis=0).abs().div(region[target], axis=0) * 100


def block_resample(n, block=BLOCK, resamples=RESAMPLES, seed=SEED):
    # moving blocks, since neighbouring months' errors are related
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(resamples, int(np.ceil(n / block))))
    return (starts[:, :, None] + np.arange(block)).reshape(resamples, -1)[:, :n]


def gain_ci(group):
    target = f"per1000_{group}"
    rows = pd.read_csv(RESULTS_DIR / f"{group}_predictions.csv").dropna(subset=BASELINES)
    others = [m for m in COMPARE if m in rows]
    ape = monthly_ape(rows, target, ["lightgbm", *others])
    idx = block_resample(len(ape))
    out = []
    for m in others:
        ours, other = ape["lightgbm"].to_numpy()[idx].mean(axis=1), ape[m].to_numpy()[idx].mean(axis=1)
        gap, rel = other - ours, (ours / other - 1) * 100
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
    target = f"per1000_{group}"
    rows = pd.read_csv(RESULTS_DIR / f"{group}_predictions.csv").dropna(subset=["lightgbm_lo", target])
    q = json.loads((RESULTS_DIR / f"{group}_model.json").read_text())["quantiles"]
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


def run(group="antihist", valid_only=False, params=None, with_nn=True):
    target = f"per1000_{group}"
    params = params or params_for(group)
    df = pd.read_csv(io.DATA_DIR / "processed" / "features.csv")
    df[BASELINES] = baselines(df, group)

    trees = {}
    for name, (objective, form) in VARIANTS.items():
        df[name], trees[name] = validate(df, target, group, objective, form, params)
        log.info("%s: trees per fold %s", name, trees[name])
    others = []
    if with_nn:
        df["pytorch"], epochs = nn.validate(df, target, group)
        others = ["pytorch"]

    valid_scores = validation_scores(df, target, BASELINES + list(VARIANTS) + others)
    mean = valid_scores[(valid_scores["year"] == "mean") & valid_scores["model"].isin(VARIANTS)]
    chosen = mean.sort_values(CHOOSE_BY)["model"].iloc[0]
    log.info("chosen on mean %s over %s: %s", CHOOSE_BY, VALID_YEARS, chosen)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    valid_scores.to_csv(RESULTS_DIR / f"{group}_valid.csv", index=False)
    if valid_only:
        return valid_scores, chosen, None
    valid_rows = df["YEAR_MONTH"].between(VALID_YEARS[0] * 100 + 1, VALID_END)
    quantiles = interval_quantiles(df[valid_rows].dropna(subset=BASELINES), target, chosen)

    test_rows = df["YEAR_MONTH"].between(TEST_START, TEST_END)
    fitted = {}
    for name, (objective, form) in VARIANTS.items():
        fitted[name] = refit(df, target, group, objective, form, trees[name][VALID_YEARS[-1]], params)
        df.loc[test_rows, name] = _decode(fitted[name].predict(df.loc[test_rows, F.FORECAST_FEATURES]),
                                          df[test_rows], group, form)
    months = sorted(df.loc[test_rows, "YEAR_MONTH"].unique())
    n_trees = trees[chosen][VALID_YEARS[-1]]
    df["lightgbm"] = rolling_forecast(df, target, group, *VARIANTS[chosen], n_trees, months, params)
    df.loc[test_rows, "lightgbm_lo"], df.loc[test_rows, "lightgbm_hi"] = add_intervals(
        df[test_rows], "lightgbm", quantiles)
    if with_nn:
        df["pytorch"] = nn.rolling_forecast(df, target, group, epochs[VALID_YEARS[-1]], months)
    test = df[test_rows].dropna(subset=BASELINES).copy()
    models = BASELINES + list(VARIANTS) + ["lightgbm"] + others
    scores = _scores(test, target, models)
    scores.insert(1, "chosen", scores["model"] == chosen)
    by_month, by_group, monthly_bias = _breakdowns(test, target, BASELINES + ["lightgbm"] + others)

    contrib = shap_in_rate_units(fitted[chosen], test, group, VARIANTS[chosen][1])
    gain = pd.Series(fitted[chosen].booster_.feature_importance("gain"), index=F.FORECAST_FEATURES)
    shap = contrib[F.FORECAST_FEATURES].abs().mean()
    importance = pd.DataFrame({"shap_pct": shap / shap.sum() * 100, "gain_pct": gain / gain.sum() * 100})
    importance = importance.round(1).sort_values("shap_pct", ascending=False)
    shap_by_month = contrib.groupby(test["YEAR_MONTH"] % 100).mean().round(3)
    shap_by_month.index.name = "calendar_month"  # "month" is also a feature column

    scores.to_csv(RESULTS_DIR / f"{group}_scores.csv", index=False)
    by_month.to_csv(RESULTS_DIR / f"{group}_by_month.csv")
    by_group.to_csv(RESULTS_DIR / f"{group}_by_group.csv")
    monthly_bias.to_csv(RESULTS_DIR / f"{group}_monthly_bias.csv")
    importance.to_csv(RESULTS_DIR / f"{group}_importance.csv")
    shap_by_month.to_csv(RESULTS_DIR / f"{group}_shap_by_month.csv")
    keep = ["YEAR_MONTH", "PRACTICE_CODE", "TOWN", "PATIENTS", target] + models + ["lightgbm_lo", "lightgbm_hi"]
    df.loc[test_rows, keep].to_csv(RESULTS_DIR / f"{group}_predictions.csv", index=False)
    (RESULTS_DIR / f"{group}_model.json").write_text(json.dumps({
        "chosen": chosen, "n_trees": int(n_trees), "params": params, "interval": INTERVAL,
        "quantiles": quantiles}, indent=2))
    gains = evaluate(group)
    log.info("results in %s", RESULTS_DIR)
    return valid_scores, chosen, (scores, monthly_bias, importance, gains)


def evaluate(group):
    gains, cover = gain_ci(group), coverage(group)
    gains.to_csv(RESULTS_DIR / f"{group}_gain_ci.csv", index=False)
    cover.to_csv(RESULTS_DIR / f"{group}_coverage.csv", index=False)
    return gains


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", choices=F.GROUPS, default="antihist")
    parser.add_argument("--valid-only", action="store_true",
                        help="stop after validation: no refit, no test metrics")
    parser.add_argument("--change", help="log this run to output/experiments.csv with this description")
    parser.add_argument("--note", default="")
    parser.add_argument("--no-nn", action="store_true", help="skip the PyTorch comparison (src.nn)")
    parser.add_argument("--params", choices=["kept", "base", "latest-tune"], default="kept",
                        help="kept: params_for(target) (TUNED_PARAMS where kept, else PARAMS); "
                             "base: PARAMS; latest-tune: the last src.experiments --tune result")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="redo the gain ranges and coverage from the saved predictions")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.evaluate_only:
        print(evaluate(args.target).to_string(index=False))
        return
    params = {"kept": params_for(args.target), "base": PARAMS}.get(args.params)
    if args.params == "latest-tune":
        params = {**PARAMS, **json.loads((RESULTS_DIR / f"{args.target}_tuned_params.json").read_text())}
    valid_scores, chosen, test = run(args.target, args.valid_only, params, with_nn=not args.no_nn)
    print("Validation:", valid_scores.to_string(index=False), sep="\n")
    if args.change:
        row = valid_scores[valid_scores["model"] == chosen].set_index("year")[CHOOSE_BY]
        label = chosen + ("" if params == PARAMS else "+tuned")
        kept = experiments.log_run(args.target, label, args.change,
                                   {y: row[y] for y in VALID_YEARS}, note=args.note)
        print(f"\nlogged: {label}, mean {CHOOSE_BY} {row['mean']:.3f}, kept={kept}")
    if test is not None:
        scores, monthly_bias, importance, gains = test
        print("\nTest:", scores.to_string(index=False), "Regional bias by test month (%):",
              monthly_bias.to_string(), importance.head(10).to_string(),
              "Gain over each comparison, with 95% range:", gains.to_string(index=False), sep="\n\n")


if __name__ == "__main__":
    main()
