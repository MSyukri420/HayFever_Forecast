"""Model features.

Usage:
    from src.feature import monthly_weather, build_features
    weather = monthly_weather(io.read_table("weather"))   # one row per TOWN x YEAR_MONTH
    df = build_features(model_table)                      # + forecasting features, HORIZON months ahead

    python -m src.feature            # model_table.csv -> dataset/processed/features.csv

Forecast setup. Prescribing data is published about two months late, so a forecast made
at the end of month o can only see data up to o, and predicts month t = o + HORIZON.
Every column in FORECAST_FEATURES uses data from month t - HORIZON or earlier
(tests/test_leakage.py checks this). Calendar columns and the policy/COVID flags are
known in advance, so they describe month t itself.

    <g>_lag3, _lag4, _lag5    rate per 1,000 at t-3, t-4, t-5 (latest three known months)
    <g>_mean3                 mean of those three
    <g>_lag12, _lag24         same month one and two years earlier
    <g>_winter_base           mean Dec-Feb rate of the latest winter fully known at t-3:
                              year-round use (eczema, itch, sleep), not hay fever
    <g>_lag12_excess          lag12 minus the winter base known at that time, i.e. last
                              year's hay-fever lift for this month
    log_patients_lag3         list size at t-3
    <w>_lag3, temp_mean_3m    weather at t-3, and mean temperature over t-5..t-3
    month, month_sin/cos, year
    otc_guidance              1 from Mar 2018, when NHS England told GPs to stop prescribing
                              for mild hay fever (the drop in the data shows from 2019)
    covid                     1 for Mar 2020 - Dec 2021 (lockdowns, remote appointments)
    LAT, LON                  practice location
(<g> is antihist, nasal or eye.) The lag numbers above assume HORIZON = 3.

Tried and dropped: spring warmth (degree days above 5 °C since 1 Jan, its anomaly, and its
change on last year) as a pollen stand-in. It made May-Jul forecasts slightly worse for all
three drug groups and did not improve 2023 validation for antihistamines or nasal sprays:
June prescribing follows June's own weather (e.g. warm spring 2024, cool June, moderate
season), which is not known 3 months ahead.

SAME_MONTH_WEATHER is the weather in month t itself. It is not known when forecasting,
so use it only for explaining past seasons, never in a forecast model.

Known small leak: PATIENTS between quarterly publications (2014 - Apr 2017) is
interpolated in src.preprocess, so log_patients_lag3 can borrow from the next quarter.
List sizes move by well under 1% a quarter.
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
SAME_MONTH_WEATHER = WEATHER_COLS


def monthly_weather(daily):
    """Monthly weather per town from the daily table.

    temp_mean, temp_max_mean   mean of daily mean / daily max temperature (°C)
    warm_days                  days with max >= 20 °C
    rain_mm, rain_days         total precipitation, days with >= 1 mm
    sunshine_hours             total sunshine
    humidity_mean              mean daily relative humidity (%)
    wind_max_mean              mean daily max wind speed (km/h)
    """
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
    """Names of the columns build_features adds that are safe to forecast with."""
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
    """Values of ``cols`` from ``source`` k months before each row of ``df``, matched on ``keys``.
    Merging on a month index (rather than shift) keeps gaps in a practice's history as NaN."""
    src = source[keys + ["_m"] + cols].assign(_m=source["_m"] + k)
    return df[keys + ["_m"]].merge(src, on=keys + ["_m"], how="left")[cols].to_numpy()


def _winter_base(df, rate, at):
    """Each practice's mean Dec-Feb ``rate`` for the latest winter that ended (Feb) by month index ``at``."""
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
    """Add FORECAST_FEATURES to the model table. Rows and existing columns are unchanged."""
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

    # weather lags come from the town table, so a gap in a practice's history doesn't blank them
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
