"""Forecast API on a tiny hand-made forecast folder (no model run needed)."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import api


@pytest.fixture
def client(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({
        "data_to": 202608, "forecast_months": [202609, 202610], "horizon": 3, "interval": 0.8,
        "chosen": {"antihist": "lgb_l2_excess"}}))
    pd.DataFrame({
        "practice_code": ["M81001", "M81001", "Y00001", "Y00001"],
        "town": ["Birmingham", "Birmingham", "Coventry", "Coventry"],
        "year_month": [202609, 202610, 202609, 202610],
        "patients": [10000, 10000, 5000, 5000],
        "forecast": [20.0, 19.0, 18.0, 17.0], "lower": [16.0, 15.0, 13.0, 12.0],
        "upper": [24.0, 23.0, 23.0, 22.0], "items": [200.0, 190.0, 90.0, 85.0],
        "drivers": ["month_cos -1.50; antihist_lag12_excess +0.80"] * 4,
        "group": "antihist",
    }).to_csv(tmp_path / "forecasts.csv", index=False)
    pd.DataFrame({"year_month": [202609, 202610], "items": [290.0, 275.0], "patients": [15000, 15000],
                  "group": "antihist", "items_lower": [280.0, 265.0], "items_upper": [300.0, 285.0],
                  "per1000": [19.33, 18.33]}).to_csv(tmp_path / "regional.csv", index=False)
    return TestClient(api.create_app(tmp_path))


def test_health(client):
    body = client.get("/health").json()
    assert body["data_to"] == "2026-08" and body["forecast_months"] == ["2026-09", "2026-10"]


def test_practice_forecast(client):
    body = client.get("/practices/m81001/forecast", params={"group": "antihist"}).json()
    assert [m["month"] for m in body["months"]] == ["2026-09", "2026-10"]
    first = body["months"][0]
    assert first["lower"] <= first["rate_per_1000"] <= first["upper"]
    assert first["drivers"][0] == {"feature": "month_cos", "contribution": -1.5}


def test_unknown_practice_and_group(client):
    assert client.get("/practices/NOPE/forecast").status_code == 404
    assert client.get("/practices/M81001/forecast", params={"group": "pollen"}).status_code == 422
    assert client.get("/practices/M81001/forecast", params={"group": "nasal"}).status_code == 404


def test_practices_by_town_and_region(client):
    assert [p["practice_code"] for p in client.get("/practices", params={"town": "coventry"}).json()] == ["Y00001"]
    months = client.get("/region/forecast").json()["months"]
    assert months[0]["items_lower"] <= months[0]["items"] <= months[0]["items_upper"]
