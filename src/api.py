"""API for the forecasts in output/forecast/.

    uvicorn src.api:app      # docs at http://127.0.0.1:8000/docs
"""

import json
import os
from enum import Enum
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

FORECAST_DIR = Path(os.environ.get("HAYFEVER_FORECAST_DIR",
                                   Path(__file__).resolve().parents[1] / "output" / "forecast"))


class Group(str, Enum):
    antihist = "antihist"
    nasal = "nasal"
    eye = "eye"


class Driver(BaseModel):
    feature: str
    contribution: float


class PracticeMonth(BaseModel):
    month: str
    rate_per_1000: float
    lower: float
    upper: float
    items: float
    drivers: list[Driver]


class PracticeForecast(BaseModel):
    practice_code: str
    town: str
    patients: int
    group: Group
    months: list[PracticeMonth]


class RegionMonth(BaseModel):
    month: str
    items: float
    items_lower: float
    items_upper: float
    rate_per_1000: float


class RegionForecast(BaseModel):
    group: Group
    months: list[RegionMonth]


class Practice(BaseModel):
    practice_code: str
    town: str
    patients: int


def _month(ym):
    return f"{ym // 100}-{ym % 100:02d}"


def _drivers(text):
    return [Driver(feature=f, contribution=float(v)) for f, v in (d.rsplit(" ", 1) for d in text.split("; "))]


@lru_cache
def _load(forecast_dir):
    forecast_dir = Path(forecast_dir)
    if not (forecast_dir / "meta.json").exists():
        raise HTTPException(503, f"no forecasts in {forecast_dir}: run python -m src.forecast")
    return (json.loads((forecast_dir / "meta.json").read_text()),
            pd.read_csv(forecast_dir / "forecasts.csv"),
            pd.read_csv(forecast_dir / "regional.csv"))


def create_app(forecast_dir=FORECAST_DIR):
    app = FastAPI(title="Hay fever prescribing forecast",
                  description="Monthly hay-fever prescribing forecasts for West Midlands GP practices, "
                              "three months ahead.")

    def load():
        return _load(str(forecast_dir))

    @app.get("/health")
    def health():
        meta, _, _ = load()
        return {"status": "ok", "data_to": _month(meta["data_to"]),
                "forecast_months": [_month(m) for m in meta["forecast_months"]], "models": meta["chosen"]}

    @app.get("/practices", response_model=list[Practice])
    def practices(town: str | None = Query(None, description="weather town, e.g. Coventry")):
        _, rows, _ = load()
        latest = rows.sort_values("year_month").drop_duplicates("practice_code", keep="last")
        if town:
            latest = latest[latest["town"].str.lower() == town.lower()]
        return [Practice(practice_code=r.practice_code, town=r.town, patients=int(r.patients))
                for r in latest.sort_values("practice_code").itertuples()]

    @app.get("/practices/{code}/forecast", response_model=PracticeForecast)
    def practice_forecast(code: str, group: Group = Group.antihist):
        _, rows, _ = load()
        mine = rows[(rows["practice_code"] == code.upper()) & (rows["group"] == group.value)]
        if mine.empty:
            raise HTTPException(404, f"no {group.value} forecast for practice {code}")
        mine = mine.sort_values("year_month")
        return PracticeForecast(
            practice_code=code.upper(), town=mine["town"].iloc[0], patients=int(mine["patients"].iloc[-1]),
            group=group,
            months=[PracticeMonth(month=_month(r.year_month), rate_per_1000=r.forecast, lower=r.lower,
                                  upper=r.upper, items=r.items, drivers=_drivers(r.drivers))
                    for r in mine.itertuples()])

    @app.get("/region/forecast", response_model=RegionForecast)
    def region_forecast(group: Group = Group.antihist):
        _, _, region = load()
        mine = region[region["group"] == group.value].sort_values("year_month")
        return RegionForecast(group=group, months=[
            RegionMonth(month=_month(r.year_month), items=r.items, items_lower=r.items_lower,
                        items_upper=r.items_upper, rate_per_1000=r.per1000) for r in mine.itertuples()])

    return app


app = create_app()
