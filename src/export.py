"""Export a star schema for Power BI (or any BI tool), built with DuckDB SQL.

Usage:
    python -m src.export      # after src.model (every target) and src.forecast

Writes output/powerbi/, one CSV per table. docs/powerbi.md describes the report built on it.

    dim_practice          one row per practice: town, location, latest list size, size band
    dim_month             one row per month: date, calendar fields, period (history/test/forecast)
    dim_group             the three drug groups and their BNF codes
    fact_prescribing      actual items, patients and rate per 1,000, per practice, month, group
    fact_test_forecast    test months (Mar 2024 - Aug 2026): actual vs LightGBM with its 80% range,
                          PyTorch and the two baselines, per practice, month, group
    fact_future_forecast  the deployed forecast for the next 3 months, with range and top drivers
    fact_regional         West Midlands totals per month and group: actual items, and forecast
                          items with the 80% range for the test and future months
Keys: practice_code, year_month and drug_group join every fact to the dimensions.
"""

import json
import logging

import duckdb
import pandas as pd

from . import feature as F
from . import io
from . import model as M
from .forecast import FORECAST_DIR

log = logging.getLogger(__name__)

EXPORT_DIR = M.RESULTS_DIR.parent / "powerbi"
GROUP_LABELS = {"antihist": "Antihistamine tablets", "nasal": "Nasal sprays", "eye": "Eye drops"}
SPC_RUN = 7


def _union(template, **fields):
    return "\nUNION ALL\n".join(template.format(g=g, **fields) for g in F.GROUPS)


SQL = {
    "dim_practice": """
        WITH latest AS (
            SELECT PRACTICE_CODE, arg_max(PATIENTS, YEAR_MONTH) AS patients_latest,
                   arg_max(POSTCODE, YEAR_MONTH) AS postcode, arg_max(TOWN, YEAR_MONTH) AS town,
                   arg_max(LAT, YEAR_MONTH) AS lat, arg_max(LON, YEAR_MONTH) AS lon,
                   min(YEAR_MONTH) AS first_month, max(YEAR_MONTH) AS last_month
            FROM model_table GROUP BY PRACTICE_CODE)
        SELECT PRACTICE_CODE AS practice_code, postcode, town, lat, lon, patients_latest,
               ['1 smallest', '2', '3', '4 largest'][ntile(4) OVER (ORDER BY patients_latest)] AS size_band,
               first_month, last_month, last_month = (SELECT max(YEAR_MONTH) FROM model_table) AS active
        FROM latest ORDER BY practice_code""",

    "dim_month": f"""
        WITH months AS (
            SELECT DISTINCT YEAR_MONTH AS year_month FROM model_table
            UNION SELECT DISTINCT year_month FROM future)
        SELECT year_month,
               make_date(year_month // 100, year_month % 100, 1) AS month_start,
               year_month // 100 AS year, year_month % 100 AS month,
               strftime(make_date(year_month // 100, year_month % 100, 1), '%b') AS month_name,
               year_month % 100 IN {M.PEAK_MONTHS} AS peak_season,
               CASE WHEN year_month > (SELECT max(YEAR_MONTH) FROM model_table) THEN 'forecast'
                    WHEN year_month >= {M.TEST_START} THEN 'test' ELSE 'history' END AS period
        FROM months ORDER BY year_month""",

    "dim_group": "SELECT * FROM groups",

    "fact_prescribing": f"""
        SELECT PRACTICE_CODE AS practice_code, YEAR_MONTH AS year_month, drug_group, items,
               PATIENTS AS patients, round(items / PATIENTS * 1000, 3) AS per1000
        FROM (UNPIVOT model_table
              ON {", ".join(f"items_{g} AS '{g}'" for g in F.GROUPS)}
              INTO NAME drug_group VALUE items)
        ORDER BY year_month, practice_code, drug_group""",

    "fact_test_forecast": _union("""
        SELECT PRACTICE_CODE AS practice_code, YEAR_MONTH AS year_month, '{g}' AS drug_group,
               PATIENTS AS patients, per1000_{g} AS actual,
               round(lightgbm, 3) AS lightgbm, round(lightgbm_lo, 3) AS lower, round(lightgbm_hi, 3) AS upper,
               round(pytorch, 3) AS pytorch, last_year_lift, last_year,
               per1000_{g} BETWEEN lightgbm_lo AND lightgbm_hi AS inside_range,
               round(abs(lightgbm - per1000_{g}), 3) AS abs_error
        FROM pred_{g}
        WHERE lightgbm IS NOT NULL AND last_year IS NOT NULL AND last_year_lift IS NOT NULL"""),

    "fact_future_forecast": """
        SELECT practice_code, year_month, "group" AS drug_group, patients::INT AS patients,
               forecast, lower, upper, round(items, 1) AS items,
               split_part(drivers, '; ', 1) AS driver_1, split_part(drivers, '; ', 2) AS driver_2,
               split_part(drivers, '; ', 3) AS driver_3
        FROM future ORDER BY year_month, practice_code, drug_group""",

    # History: actual totals. Test: the same practice-months the scores use (rows with both
    # baselines), forecast total and its range from the validation quantiles. Future: src.forecast.
    "fact_regional": """
        SELECT year_month, drug_group, actual_items::BIGINT AS actual_items, patients::BIGINT AS patients,
               round(forecast_items)::BIGINT AS forecast_items,
               round(forecast_lower)::BIGINT AS forecast_lower, round(forecast_upper)::BIGINT AS forecast_upper
        FROM (""" + _union("""
        SELECT YEAR_MONTH AS year_month, '{g}' AS drug_group, sum(items_{g}) AS actual_items,
               sum(PATIENTS) AS patients, NULL::DOUBLE AS forecast_items,
               NULL::DOUBLE AS forecast_lower, NULL::DOUBLE AS forecast_upper
        FROM model_table WHERE YEAR_MONTH < {test_start} GROUP BY YEAR_MONTH""",
        test_start=M.TEST_START) + "\nUNION ALL\n" + _union("""
        SELECT p.YEAR_MONTH, '{g}', round(sum(p.per1000_{g} * p.PATIENTS / 1000)), sum(p.PATIENTS),
               sum(p.lightgbm * p.PATIENTS / 1000),
               sum(p.lightgbm * p.PATIENTS / 1000) * (1 + q.lo), sum(p.lightgbm * p.PATIENTS / 1000) * (1 + q.hi)
        FROM pred_{g} p, region_q q
        WHERE q.drug_group = '{g}' AND p.lightgbm IS NOT NULL AND p.last_year IS NOT NULL
              AND p.last_year_lift IS NOT NULL
        GROUP BY p.YEAR_MONTH, q.lo, q.hi""") + """
        UNION ALL
        SELECT year_month, "group", NULL, patients, items, items_lower, items_upper FROM future_region)
        ORDER BY drug_group, year_month""",

    # XmR control chart of the monthly regional error on the test: limits are the mean
    # +/- 2.66 x the average moving range. A signal is a month outside the limits or part of
    # a run of 7+ months on one side of the mean (a sustained bias).
    "fact_error_spc": f"""
        WITH err AS (
            SELECT drug_group, year_month,
                   round((forecast_items - actual_items) / actual_items * 100, 2) AS error_pct
            FROM fact_regional WHERE actual_items IS NOT NULL AND forecast_items IS NOT NULL),
        mr AS (
            SELECT *, abs(error_pct - lag(error_pct) OVER w) AS moving_range,
                   avg(error_pct) OVER (PARTITION BY drug_group) AS mean
            FROM err WINDOW w AS (PARTITION BY drug_group ORDER BY year_month)),
        lim AS (
            SELECT *, mean + 2.66 * avg(moving_range) OVER (PARTITION BY drug_group) AS upper_limit,
                   mean - 2.66 * avg(moving_range) OVER (PARTITION BY drug_group) AS lower_limit,
                   sign(error_pct - mean) AS side
            FROM mr),
        runs AS (
            SELECT *, row_number() OVER (PARTITION BY drug_group ORDER BY year_month)
                      - row_number() OVER (PARTITION BY drug_group, side ORDER BY year_month) AS run_id
            FROM lim),
        lengths AS (
            SELECT *, count(*) OVER (PARTITION BY drug_group, side, run_id) AS run_length FROM runs)
        SELECT drug_group, year_month, error_pct, round(moving_range, 2) AS moving_range,
               round(mean, 2) AS mean, round(upper_limit, 2) AS upper_limit,
               round(lower_limit, 2) AS lower_limit,
               CASE WHEN error_pct > upper_limit OR error_pct < lower_limit THEN 'outside limits'
                    WHEN run_length >= {SPC_RUN} AND side != 0 THEN 'run of {SPC_RUN}+'
                    ELSE 'none' END AS signal
        FROM lengths ORDER BY drug_group, year_month""",
}


def connect():
    con = duckdb.connect()
    con.execute(f"CREATE VIEW model_table AS SELECT * FROM "
                f"read_csv('{(io.DATA_DIR / 'processed' / 'model_table.csv').as_posix()}')")
    for g in F.GROUPS:
        con.execute(f"CREATE VIEW pred_{g} AS SELECT * FROM "
                    f"read_csv('{(M.RESULTS_DIR / f'{g}_predictions.csv').as_posix()}')")
    con.execute(f"CREATE VIEW future AS SELECT * FROM read_csv('{(FORECAST_DIR / 'forecasts.csv').as_posix()}')")
    con.execute(f"CREATE VIEW future_region AS SELECT * FROM "
                f"read_csv('{(FORECAST_DIR / 'regional.csv').as_posix()}')")
    con.register("groups", pd.DataFrame(
        [(g, GROUP_LABELS[g], code) for code, g in io.BNF_GROUPS.items()],
        columns=["drug_group", "label", "bnf_code"]))
    quantiles = {g: json.loads((M.RESULTS_DIR / f"{g}_model.json").read_text())["quantiles"]["region"]
                 for g in F.GROUPS}
    con.register("region_q", pd.DataFrame([(g, lo, hi) for g, (lo, hi) in quantiles.items()],
                                          columns=["drug_group", "lo", "hi"]))
    return con


def export(out_dir=EXPORT_DIR):
    con = connect()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, sql in SQL.items():
        path = (out_dir / f"{name}.csv").as_posix()
        con.execute(f"CREATE VIEW {name} AS {sql}")
        con.execute(f"COPY {name} TO '{path}' (HEADER)")
        rows, = con.execute(f"SELECT count(*) FROM read_csv('{path}')").fetchall()[0]
        log.info("%-22s %s rows", name, f"{rows:,}")
    log.info("Power BI tables in %s", out_dir)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    export()


if __name__ == "__main__":
    main()
