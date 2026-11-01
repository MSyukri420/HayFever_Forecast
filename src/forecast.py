"""Train on all the data and forecast the next HORIZON months for every practice.

    python -m src.forecast      # after src.model for each target
"""

import json
import logging
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import feature as F
from . import io
from . import model as M

log = logging.getLogger(__name__)

FORECAST_DIR = M.RESULTS_DIR.parent / "forecast"
DRIVERS = 3


def _add_months(ym, k):
    m = F._month_index(ym) + k
    return (m // 12) * 100 + m % 12 + 1


def future_rows(table, horizon=F.HORIZON):
    last = table["YEAR_MONTH"].max()
    current = table[table["YEAR_MONTH"] == last][["PRACTICE_CODE", "POSTCODE", "LAT", "LON", "TOWN"]]
    months = [_add_months(last, k) for k in range(1, horizon + 1)]
    return pd.concat([current.assign(YEAR_MONTH=m) for m in months], ignore_index=True)


def _drivers(contrib, n=DRIVERS):
    values = contrib[F.FORECAST_FEATURES]
    top = np.argsort(-values.abs().to_numpy(), axis=1)[:, :n]
    names = np.array(F.FORECAST_FEATURES)[top]
    vals = np.take_along_axis(values.to_numpy(), top, axis=1)
    return ["; ".join(f"{f} {v:+.2f}" for f, v in zip(fs, vs)) for fs, vs in zip(names, vals)]


def forecast_group(df, future, group):
    target = f"per1000_{group}"
    spec = json.loads((M.RESULTS_DIR / f"{group}_model.json").read_text())
    objective, form = M.VARIANTS[spec["chosen"]]
    y = M._encode(df, target, group, form)
    train = y.notna() & np.isfinite(y)
    params = {**spec["params"], "objective": objective, "n_estimators": spec["n_trees"]}
    model = lgb.LGBMRegressor(**params).fit(df.loc[train, F.FORECAST_FEATURES], y[train])

    rows = df[future].copy()
    rows["forecast"] = M._decode(model.predict(rows[F.FORECAST_FEATURES]), rows, group, form)
    rows = rows.dropna(subset=["forecast"])  # no winter base yet: practice too new to forecast
    rows["patients"] = np.exp(rows[f"log_patients_lag{F.HORIZON}"]).round()  # latest known list size
    rows = rows.dropna(subset=["patients"])
    rows["lower"], rows["upper"] = M.add_intervals(rows, "forecast", spec["quantiles"], patients="patients")
    rows["drivers"] = _drivers(M.shap_in_rate_units(model, rows, group, form))
    rows["items"] = rows["forecast"] * rows["patients"] / 1000
    out = rows[["PRACTICE_CODE", "TOWN", "YEAR_MONTH", "patients", "forecast", "lower", "upper", "items",
                "drivers"]].rename(columns=str.lower).assign(group=group)

    total = out.groupby("year_month").agg(items=("items", "sum"), patients=("patients", "sum"))
    lo, hi = spec["quantiles"]["region"]
    region = total.assign(group=group, items_lower=total["items"] * (1 + lo), items_upper=total["items"] * (1 + hi),
                          per1000=total["items"] / total["patients"] * 1000).reset_index()
    log.info("%s (%s): %d practices, trained on %s rows", group, spec["chosen"],
             out["practice_code"].nunique(), f"{train.sum():,}")
    return out, region, spec["chosen"]


def make_forecasts(out_dir=FORECAST_DIR):
    table = pd.read_csv(io.DATA_DIR / "processed" / "model_table.csv")
    future = future_rows(table)
    df = F.build_features(pd.concat([table, future], ignore_index=True))
    is_future = df["YEAR_MONTH"] > table["YEAR_MONTH"].max()

    practices, regions, chosen = [], [], {}
    for group in F.GROUPS:
        if not (M.RESULTS_DIR / f"{group}_model.json").exists():
            log.warning("skipped %s: run python -m src.model --target %s first", group, group)
            continue
        out, region, chosen[group] = forecast_group(df, is_future, group)
        practices.append(out)
        regions.append(region)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(practices).round(3).to_csv(out_dir / "forecasts.csv", index=False)
    pd.concat(regions).round(3).to_csv(out_dir / "regional.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps({
        "data_to": int(table["YEAR_MONTH"].max()),
        "forecast_months": sorted(int(m) for m in future["YEAR_MONTH"].unique()),
        "horizon": F.HORIZON, "interval": M.INTERVAL, "chosen": chosen,
        "made_on": date.today().isoformat()}, indent=2))
    log.info("forecasts in %s", out_dir)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    make_forecasts()


if __name__ == "__main__":
    main()
