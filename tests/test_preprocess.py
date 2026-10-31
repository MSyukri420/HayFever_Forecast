"""The DuckDB table build on a tiny hand-made dataset (no downloads needed)."""

import pandas as pd
import pytest

from src import io, preprocess


@pytest.fixture
def data_dir(tmp_path):
    rx = pd.DataFrame({
        "YEAR_MONTH": [202401, 202401, 202401, 202402, 202401],
        "PRACTICE_CODE": ["A", "A", "B", "A", "WALKIN"],
        "POSTCODE": ["B1 1AA"] * 5,
        "BNF_GROUP": ["030401", "110402", "030401", "120201", "030401"],
        "ITEMS": [10, 2, 7, 4, 99],
        "QUANTITY": [1.0] * 5,
    })
    patients = pd.DataFrame({  # A has no list size for 202402, only either side of it
        "YEAR_MONTH": [202401, 202403, 202401, 202402],
        "PRACTICE_CODE": ["A", "A", "B", "B"],
        "PATIENTS": [2000.0, 3000.0, 500.0, 500.0],
    })
    days = pd.date_range("2023-10-01", "2024-02-29")
    weather = pd.DataFrame({
        "icb_code": "QHL", "town": "Birmingham", "date": days.strftime("%Y-%m-%d"),
        "temperature_2m_mean": 5.0, "temperature_2m_max": 8.0, "precipitation_sum": 2.0,
        "rain_sum": 2.0, "sunshine_duration": 3600.0, "wind_speed_10m_max": 20.0,
        "relative_humidity_2m_mean": 80,
    })
    locations = pd.DataFrame({"PRACTICE_CODE": ["A", "B"], "POSTCODE": ["B1 1AA", "CV1 1AA"],
                              "LAT": [52.48, 52.41], "LON": [-1.90, -1.51], "LOCATION_FROM": "postcode"})
    for name, df in [("prescriptions", rx), ("patients", patients), ("weather", weather),
                     ("locations", locations)]:
        df.to_csv(tmp_path / io.TABLES[name], index=False)
    return tmp_path


def test_build_dataset(data_dir):
    df = preprocess.build_dataset(data_dir)
    # WALKIN has no registered list; B is under MIN_PATIENTS
    assert df["PRACTICE_CODE"].unique().tolist() == ["A"]
    a = df.set_index("YEAR_MONTH")
    assert a.loc[202401, ["items_antihist", "items_eye", "items_nasal"]].tolist() == [10, 2, 0]
    assert a.loc[202402, "PATIENTS"] == 2500  # interpolated between publications
    assert a.loc[202401, "per1000_antihist"] == 5.0
    assert a.loc[202402, "per1000_nasal"] == 1.6
    assert a.loc[202401, "TOWN"] == "Birmingham"
    assert a.loc[202401, "temp_mean"] == 5.0 and a.loc[202401, "sunshine_hours"] == 31.0
