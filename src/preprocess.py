"""Clean the three raw tables and join them into one practice-month modelling table.

Usage:
    from src.preprocess import build_dataset
    df = build_dataset()             # one row per YEAR_MONTH x PRACTICE_CODE

    python -m src.preprocess         # build and save to dataset/processed/model_table.csv

Steps:
    1. Drop prescribing codes with no registered list (walk-in, out-of-hours, etc.), ~1.5% of items.
    2. Pivot to one row per practice-month: items_antihist, items_nasal, items_eye
       (0 when the practice prescribed none of that group that month).
    3. Fill list sizes between publications (see ``monthly_patients``) and drop
       practice-months under MIN_PATIENTS, where per-patient rates are meaningless.
    4. Target columns per1000_<group> = items / patients * 1000.
    5. Attach each practice's location and nearest weather town, then that town's
       monthly weather (src.feature.monthly_weather).
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import feature, io

log = logging.getLogger(__name__)

MIN_PATIENTS = 1000


def monthly_patients(patients, months):
    """Fill quarterly gaps and the two missing months by linear interpolation within each
    practice's own lifetime, plus up to 2 months either side so closing practices keep their last quarter."""
    wide = patients.pivot(index="YEAR_MONTH", columns="PRACTICE_CODE", values="PATIENTS")
    wide = wide.reindex(sorted(set(wide.index) | set(months)))
    filled = wide.interpolate(limit_area="inside").ffill(limit=2).bfill(limit=2)
    return filled.loc[months].stack().rename("PATIENTS").reset_index()


def assign_towns(locations, towns=io.TOWNS):
    """Nearest weather town to each practice (straight-line distance; fine at this scale)."""
    t = pd.DataFrame(towns.values(), columns=["town", "lat", "lon"])
    km = np.hypot((locations["LAT"].values[:, None] - t["lat"].values) * 111,
                  (locations["LON"].values[:, None] - t["lon"].values) * 68)
    return locations.assign(TOWN=t["town"].values[km.argmin(axis=1)], TOWN_KM=km.min(axis=1).round(1))


def build_dataset(data_dir=io.DATA_DIR):
    """Read, clean and join the raw tables into one row per practice-month."""
    rx = io.read_table("prescriptions", data_dir)
    patients = io.read_table("patients", data_dir)

    rx = rx[rx["PRACTICE_CODE"].isin(patients["PRACTICE_CODE"])]

    items = rx.pivot_table(index=["YEAR_MONTH", "PRACTICE_CODE"], columns="BNF_GROUP",
                           values="ITEMS", aggfunc="sum", fill_value=0)
    items = items.rename(columns={g: f"items_{n}" for g, n in io.BNF_GROUPS.items()}).reset_index()
    items.columns.name = None

    df = items.merge(monthly_patients(patients, sorted(rx["YEAR_MONTH"].unique())),
                     on=["YEAR_MONTH", "PRACTICE_CODE"], how="left")
    dropped = df["PATIENTS"].isna() | (df["PATIENTS"] < MIN_PATIENTS)
    log.info("dropped %s practice-months with no list size or under %s patients",
             f"{dropped.sum():,}", f"{MIN_PATIENTS:,}")
    df = df[~dropped]

    for name in io.BNF_GROUPS.values():
        df[f"per1000_{name}"] = (df[f"items_{name}"] / df["PATIENTS"] * 1000).round(3)

    locations = assign_towns(io.read_table("locations", data_dir))
    log.info("practice to weather town: median %.1f km, max %.1f km",
             locations["TOWN_KM"].median(), locations["TOWN_KM"].max())
    df = (df.merge(locations[["PRACTICE_CODE", "POSTCODE", "LAT", "LON", "TOWN"]], on="PRACTICE_CODE", how="left")
            .merge(feature.monthly_weather(io.read_table("weather", data_dir)), on=["TOWN", "YEAR_MONTH"], how="left"))
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
