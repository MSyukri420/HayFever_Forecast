"""Forecast features: everything in FORECAST_FEATURES uses data from month t - HORIZON or earlier.

    python -m src.feature      # model_table.csv -> features.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import io

log = logging.getLogger(__name__)

HORIZON = 3
GROUPS = list(io.BNF_GROUPS.values())
TARGETS = [f"per1000_{g}" for g in GROUPS]
OTC_GUIDANCE_FROM = 201803
COVID_MONTHS = (202003, 202112)
WEATHER_COLS = ["temp_mean", "temp_max_mean", "warm_days", "rain_mm", "rain_days",
                "sunshine_hours", "humidity_mean", "wind_max_mean"]
SAME_MONTH_WEATHER = WEATHER_COLS  # month t's own weather: fine for explaining, never a feature


def monthly_weather(daily):
    date = pd.to_datetime(daily["date"])
    daily = daily.assign(YEAR_MONTH=date.dt.year * 100 + date.dt.month)
    return daily.groupby(["town", "YEAR_MONTH"]).agg(
        temp_mean=("temperature_2m_mean", "mean"),
        temp_max_mean=("temperature_2m_max", "mean"),
        warm_days=("temperature_2m_max", lambda s: (s >= 20).sum()),
        rain_mm=("precipitation_sum", "sum"),
        rain_days=("precipitation_sum", lambda s: (s >= 1).sum()),
        sunshine_hours=("sunshine_duration", lambda s: s.sum() / 3600),
        humidity_mean=("relative_humidity_2m_mean", "mean"),
        wind_max_mean=("wind_speed_10m_max", "mean"),
    ).round(2).reset_index().rename(columns={"town": "TOWN"})


def forecast_features(horizon=HORIZON):
    lags = [horizon, horizon + 1, horizon + 2]
    cols = []
    for g in GROUPS:
        cols += [f"{g}_lag{k}" for k in lags] + [f"{g}_mean3", f"{g}_lag12", f"{g}_lag24",
                                                 f"{g}_winter_base", f"{g}_lag12_excess"]
    cols += [f"log_patients_lag{horizon}"] + [f"{w}_lag{horizon}" for w in WEATHER_COLS] + ["temp_mean_3m"]
    cols += ["month", "month_sin", "month_cos", "year", "otc_guidance", "covid", "LAT", "LON"]
    return cols


FORECAST_FEATURES = forecast_features()


def _month_index(ym):
    return (ym // 100) * 12 + (ym % 100) - 1


def _lagged(df, source, keys, cols, k):
    # merge on a month index rather than shift, so gaps in a practice's history stay NaN
    src = source[keys + ["_m"] + cols].assign(_m=source["_m"] + k)
    return df[keys + ["_m"]].merge(src, on=keys + ["_m"], how="left")[cols].to_numpy()


def _winter_base(df, rate, at):
    # mean Dec-Feb rate of the latest winter finished by month index `at`
    month = df["YEAR_MONTH"] % 100
    winter = df[month.isin([12, 1, 2])]
    end_year = winter["YEAR_MONTH"] // 100 + (winter["YEAR_MONTH"] % 100 == 12)
    base = (winter.assign(_end=end_year).groupby(["PRACTICE_CODE", "_end"])[rate]
                  .agg(["mean", "size"]).query("size >= 2")["mean"].rename("base").reset_index())
    at_year, at_month = at // 12, at % 12 + 1
    lookup = pd.DataFrame({"PRACTICE_CODE": df["PRACTICE_CODE"].values,
                           "_end": np.where(at_month >= 2, at_year, at_year - 1)})
    return lookup.merge(base, on=["PRACTICE_CODE", "_end"], how="left")["base"].to_numpy()


def build_features(table, horizon=HORIZON):
    df = table.copy()
    df["_m"] = _month_index(df["YEAR_MONTH"])
    keys = ["PRACTICE_CODE"]

    for g in GROUPS:
        rate = f"per1000_{g}"
        lags = [horizon, horizon + 1, horizon + 2]
        for k in lags + [12, 24]:
            df[f"{g}_lag{k}"] = _lagged(df, df, keys, [rate], k)[:, 0]
        df[f"{g}_mean3"] = df[[f"{g}_lag{k}" for k in lags]].mean(axis=1, skipna=False)
        df[f"{g}_winter_base"] = _winter_base(df, rate, df["_m"] - horizon)
        df[f"{g}_lag12_excess"] = df[f"{g}_lag12"] - _winter_base(df, rate, df["_m"] - 12)

    df[f"log_patients_lag{horizon}"] = np.log(_lagged(df, df, keys, ["PATIENTS"], horizon)[:, 0])

    # weather lags come from the town table so practice gaps don't blank them
    town = df[["TOWN", "YEAR_MONTH", "_m"] + WEATHER_COLS].drop_duplicates(["TOWN", "YEAR_MONTH"])
    df[[f"{w}_lag{horizon}" for w in WEATHER_COLS]] = _lagged(df, town, ["TOWN"], WEATHER_COLS, horizon)
    df["temp_mean_3m"] = np.column_stack(
        [_lagged(df, town, ["TOWN"], ["temp_mean"], k)[:, 0] for k in (horizon, horizon + 1, horizon + 2)]
    ).mean(axis=1)

    month = df["YEAR_MONTH"] % 100
    df["month"] = month
    df["month_sin"] = np.sin(2 * np.pi * month / 12).round(4)
    df["month_cos"] = np.cos(2 * np.pi * month / 12).round(4)
    df["year"] = df["YEAR_MONTH"] // 100
    df["otc_guidance"] = (df["YEAR_MONTH"] >= OTC_GUIDANCE_FROM).astype(int)
    df["covid"] = df["YEAR_MONTH"].between(*COVID_MONTHS).astype(int)

    new = [c for c in forecast_features(horizon) if c not in table.columns]
    df[new] = df[new].round(4)
    log.info("features: +%d columns, %s rows; rows with every lag present: %s",
             len(new), f"{len(df):,}", f"{df[new].notna().all(axis=1).sum():,}")
    return df.drop(columns="_m")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--table", type=Path, default=io.DATA_DIR / "processed" / "model_table.csv")
    parser.add_argument("--out", type=Path, default=io.DATA_DIR / "processed" / "features.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    io.save_dataset(build_features(pd.read_csv(args.table), args.horizon), args.out)


if __name__ == "__main__":
    main()
