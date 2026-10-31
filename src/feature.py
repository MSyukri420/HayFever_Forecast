"""Model features.

Usage:
    from src.feature import monthly_weather
    weather = monthly_weather(io.read_table("weather"))   # one row per TOWN x YEAR_MONTH

So far only weather: daily Open-Meteo values summarised to the calendar month, so they
line up with the monthly prescribing data. src.preprocess joins them onto each practice
by its nearest town.
"""

import pandas as pd


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
