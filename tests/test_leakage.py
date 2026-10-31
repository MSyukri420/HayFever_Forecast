"""Leakage tests: a forecast feature for month t may only use data from t - HORIZON or earlier.

The main check is a cutoff test. Features are built twice: once on the real model table,
and once with every measured value (items, rates, patients, weather) after month C
replaced by noise. For target months up to C + HORIZON the two builds must match exactly.
Any difference means a feature looked at data it would not have had when forecasting.
"""

import numpy as np
import pandas as pd
import pytest

from src import feature as F
from src import io

TABLE = io.DATA_DIR / "processed" / "model_table.csv"
pytestmark = pytest.mark.skipif(not TABLE.exists(), reason="dataset/ not built (python -m src.run_all)")

MEASURED = ([f"items_{g}" for g in F.GROUPS] + F.TARGETS + ["PATIENTS"] + F.WEATHER_COLS)


@pytest.fixture(scope="module")
def table():
    return pd.read_csv(TABLE)


def _add_months(ym, k):
    m = F._month_index(ym) + k
    return (m // 12) * 100 + m % 12 + 1


# Feb and Mar cutoffs sit either side of a winter ending, the trickiest case for winter_base
@pytest.mark.parametrize("cutoff", [201811, 201902, 201903, 202305])
def test_features_ignore_data_after_cutoff(table, cutoff):
    noisy = table.copy()
    later = noisy["YEAR_MONTH"] > cutoff
    rng = np.random.default_rng(0)
    for col in MEASURED:
        noisy.loc[later, col] = rng.uniform(1, 100, later.sum()).round()

    real = F.build_features(table)
    fake = F.build_features(noisy)
    checked = real["YEAR_MONTH"] <= _add_months(cutoff, F.HORIZON)
    assert checked.sum() > 0
    pd.testing.assert_frame_equal(real.loc[checked, F.FORECAST_FEATURES],
                                  fake.loc[checked, F.FORECAST_FEATURES])


def test_some_rows_after_the_window_do_change(table):
    """Guards the test above: the noise must actually reach the features one month later."""
    cutoff = 202305
    noisy = table.copy()
    later = noisy["YEAR_MONTH"] > cutoff
    noisy.loc[later, F.TARGETS] = 999.0
    month = noisy["YEAR_MONTH"] == _add_months(cutoff, F.HORIZON + 1)
    lag = f"antihist_lag{F.HORIZON}"
    assert (F.build_features(noisy).loc[month, lag] == 999.0).all()


def test_same_month_values_are_not_forecast_features():
    assert not set(F.FORECAST_FEATURES) & set(F.SAME_MONTH_WEATHER + F.TARGETS + MEASURED)
