"""Clean the raw tables and join them into one practice-month modelling table, in DuckDB SQL.

Usage:
    from src.preprocess import build_dataset
    df = build_dataset()             # one row per YEAR_MONTH x PRACTICE_CODE

    python -m src.preprocess         # build and save to dataset/processed/model_table.csv

The raw CSVs are queried in place by DuckDB (no loading step), as the views in SQL below:
    1. rx_registered: drop prescribing codes with no registered list (walk-in, out-of-hours,
       etc.), ~1.2% of items.
    2. items: pivot to one row per practice-month: items_antihist, items_nasal, items_eye
       (0 when the practice prescribed none of that group that month).
    3. patients_monthly (pandas, see ``monthly_patients``): list sizes filled between
       publications. Practice-months under MIN_PATIENTS are dropped, where per-patient rates
       mean little.
    4. Target columns per1000_<group> = items / patients * 1000.
    5. practice_towns: each practice's location and nearest weather town, then that town's
       weather per month (weather_monthly, from src.feature.monthly_weather).
Two steps stay in pandas: filling list sizes (linear interpolation) and the monthly weather means.
DuckDB adds up a month's days in a different order, which flips the rounding of a few means
that fall exactly between two hundredths; pandas keeps the table identical to the original
pandas-only build, and so every reported result. Rounding of the rates is done in pandas too.
"""

import argparse
import logging
from pathlib import Path

import duckdb
import pandas as pd

from . import feature, io

log = logging.getLogger(__name__)

MIN_PATIENTS = 1000

GROUP_ITEMS = ",\n           ".join(
    f"coalesce(sum(ITEMS) FILTER (WHERE BNF_GROUP = '{code}'), 0)::BIGINT AS items_{name}"
    for code, name in sorted(io.BNF_GROUPS.items()))

SQL = f"""
CREATE VIEW rx_registered AS
    SELECT * FROM prescriptions
    WHERE PRACTICE_CODE IN (SELECT PRACTICE_CODE FROM patients);

CREATE VIEW items AS
    SELECT YEAR_MONTH, PRACTICE_CODE,
           {GROUP_ITEMS}
    FROM rx_registered
    GROUP BY YEAR_MONTH, PRACTICE_CODE;

CREATE VIEW practice_towns AS
    SELECT l.PRACTICE_CODE, l.POSTCODE, l.LAT, l.LON,
           arg_min(t.town, km) AS TOWN, min(km) AS TOWN_KM
    FROM locations l
    CROSS JOIN towns t
    CROSS JOIN LATERAL (SELECT sqrt(((l.LAT - t.lat) * 111) ** 2 + ((l.LON - t.lon) * 68) ** 2) AS km)
    GROUP BY ALL;

CREATE VIEW items_patients AS
    SELECT i.*, p.PATIENTS
    FROM items i
    LEFT JOIN patients_monthly p USING (YEAR_MONTH, PRACTICE_CODE);

CREATE VIEW model_table AS
    SELECT ip.YEAR_MONTH, ip.PRACTICE_CODE,
           {", ".join(f"ip.items_{n}" for _, n in sorted(io.BNF_GROUPS.items()))},
           ip.PATIENTS,
           {", ".join(f"ip.items_{n} / ip.PATIENTS * 1000 AS per1000_{n}" for n in io.BNF_GROUPS.values())},
           pt.POSTCODE, pt.LAT, pt.LON, pt.TOWN,
           {", ".join(f"w.{c}" for c in feature.WEATHER_COLS)}
    FROM items_patients ip
    LEFT JOIN practice_towns pt USING (PRACTICE_CODE)
    LEFT JOIN weather_monthly w USING (TOWN, YEAR_MONTH)
    WHERE ip.PATIENTS >= {MIN_PATIENTS}
    ORDER BY ip.YEAR_MONTH, ip.PRACTICE_CODE;
"""


def monthly_patients(patients, months):
    """Fill quarterly gaps and the two missing months by linear interpolation within each
    practice's own lifetime, plus up to 2 months either side so closing practices keep their last quarter."""
    wide = patients.pivot(index="YEAR_MONTH", columns="PRACTICE_CODE", values="PATIENTS")
    wide = wide.reindex(sorted(set(wide.index) | set(months)))
    filled = wide.interpolate(limit_area="inside").ffill(limit=2).bfill(limit=2)
    return filled.loc[months].stack().rename("PATIENTS").reset_index()


def connect(data_dir=io.DATA_DIR):
    """DuckDB connection with the raw CSVs as views, plus the weather towns."""
    con = duckdb.connect()
    for name, file in io.TABLES.items():
        path = (Path(data_dir) / file).as_posix()
        types = ", types={'BNF_GROUP': 'VARCHAR'}" if name == "prescriptions" else ""
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_csv('{path}'{types})")
    con.register("towns", pd.DataFrame(io.TOWNS.values(), columns=["town", "lat", "lon"]))
    return con


def build_dataset(data_dir=io.DATA_DIR):
    """Read, clean and join the raw tables into one row per practice-month."""
    con = connect(data_dir)
    months = con.sql("SELECT DISTINCT YEAR_MONTH FROM prescriptions "
                     "WHERE PRACTICE_CODE IN (SELECT PRACTICE_CODE FROM patients) ORDER BY 1").df()
    patients = con.sql("SELECT YEAR_MONTH, PRACTICE_CODE, PATIENTS FROM patients").df()
    con.register("patients_monthly", monthly_patients(patients, months["YEAR_MONTH"].tolist()))
    con.register("weather_monthly", feature.monthly_weather(con.sql("SELECT * FROM weather").df()))
    con.execute(SQL)

    unregistered, = con.execute(
        "SELECT 1 - (SELECT sum(ITEMS) FROM rx_registered) / sum(ITEMS) FROM prescriptions").fetchall()[0]
    dropped, = con.execute(
        f"SELECT count(*) FROM items_patients WHERE PATIENTS IS NULL OR PATIENTS < {MIN_PATIENTS}").fetchall()[0]
    towns = con.execute("SELECT median(TOWN_KM), max(TOWN_KM) FROM practice_towns").fetchall()[0]
    log.info("dropped %.1f%% of items with no registered list", unregistered * 100)
    log.info("dropped %s practice-months with no list size or under %s patients",
             f"{dropped:,}", f"{MIN_PATIENTS:,}")
    log.info("practice to weather town: median %.1f km, max %.1f km", *towns)

    df = con.sql("SELECT * FROM model_table").df()
    rates = [f"per1000_{n}" for n in io.BNF_GROUPS.values()]
    df[rates] = df[rates].round(3)
    df["PATIENTS"] = df["PATIENTS"].round().astype(int)
    log.info("model table: %s rows x %d columns", f"{len(df):,}", df.shape[1])
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=io.DATA_DIR)
    parser.add_argument("--out", type=Path, default=io.DATA_DIR / "processed" / "model_table.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    io.save_dataset(build_dataset(args.data_dir), args.out)


if __name__ == "__main__":
    main()
