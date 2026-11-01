"""Download prescribing, list size, weather and practice location data, cached per month/town.

    python -m src.io
"""

import io as _bytes_io
import json
import logging
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import certifi
import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "dataset"
CACHE_DIR = DATA_DIR / "cache"

TABLES = {
    "prescriptions": "prescriptions_wm.csv",
    "patients": "patients_wm.csv",
    "weather": "weather_wm_daily.csv",
    "locations": "practice_locations.csv",
}
_DTYPES = {"prescriptions": {"BNF_GROUP": str}}

FIRST, LAST = (2014, 1), (2026, 8)  # prescribing months to download

# West Midlands postcode areas (the digit stops B matching BA, BS...)
POSTCODE_REGEX = r"^(B|CV|DY|WS|WV|WR|HR|ST|TF|SY)[0-9]"
BNF_GROUPS = {"030401": "antihist", "120201": "nasal", "110402": "eye"}

# one weather town per ICB, plus Hereford
TOWNS = {
    "QHL": ("Birmingham", 52.48, -1.90),
    "QUA": ("Wolverhampton", 52.59, -2.13),
    "QWU": ("Coventry", 52.41, -1.51),
    "QGH": ("Worcester", 52.19, -2.22),
    "QGH_HR": ("Hereford", 52.06, -2.72),
    "QNC": ("Stoke-on-Trent", 53.00, -2.18),
    "QOC": ("Shrewsbury", 52.71, -2.75),
}

# the Windows cert store rejects the Let's Encrypt chain
_SSL = ssl.create_default_context(cafile=certifi.where())


def _get(url, data=None, retries=4, rate_limit_wait=65):
    headers = {"User-Agent": "Mozilla/5.0"}
    if data is not None:
        data, headers["Content-Type"] = json.dumps(data).encode(), "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180, context=_SSL) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404 or attempt == retries - 1:
                raise
            wait = rate_limit_wait if e.code == 429 else 5 * (attempt + 1)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            wait = 5 * (attempt + 1)
        log.warning("%s: retrying in %ds", url[:80], wait)
        time.sleep(wait)


def _months():
    y, m = FIRST
    while (y, m) <= LAST:
        yield y * 100 + m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _combine(cache, out):
    df = pd.concat([pd.read_csv(p, dtype=str) for p in sorted(cache.glob("*.csv"))], ignore_index=True)
    df.to_csv(out, index=False)
    log.info("saved %s rows to %s", f"{len(df):,}", out)


# Prescriptions (NHSBSA)

EPD_API = "https://opendata.nhsbsa.net/api/3/action/datastore_search_sql"
SNOMED_FROM = 202503  # table and column names change from here


def _prescriptions_sql(ym):
    if ym >= SNOMED_FROM:
        table, substance = f"EPD_SNOMED_{ym}", "BNF_CHEMICAL_SUBSTANCE_CODE"
    else:
        table, substance = f"EPD_{ym}", "BNF_CHEMICAL_SUBSTANCE"
    groups = ",".join(f"'{g}'" for g in BNF_GROUPS)
    sql = (
        f"SELECT PRACTICE_CODE, ANY_VALUE(POSTCODE) AS POSTCODE, "
        f"SUBSTR({substance}, 1, 6) AS BNF_GROUP, "
        f"SUM(ITEMS) AS ITEMS, SUM(TOTAL_QUANTITY) AS QUANTITY "
        f"FROM `{table}` "
        f"WHERE REGEXP_CONTAINS(POSTCODE, r'{POSTCODE_REGEX}') "
        f"AND SUBSTR({substance}, 1, 6) IN ({groups}) "
        f"GROUP BY 1, 3"
    )
    return table, sql


def fetch_prescriptions_month(ym):
    table, sql = _prescriptions_sql(ym)
    body = json.loads(_get(EPD_API + "?" + urllib.parse.urlencode({"resource_id": table, "sql": sql})))
    if not body.get("success"):
        raise RuntimeError(f"{ym}: {body.get('error')}")
    # the response is wrapped twice: result -> result -> records
    df = pd.DataFrame(body["result"]["result"]["records"])
    df.insert(0, "YEAR_MONTH", ym)  # stored as a number for every month
    return df


def download_prescriptions():
    cache = CACHE_DIR / "epd_months"
    cache.mkdir(parents=True, exist_ok=True)
    for ym in _months():
        path = cache / f"{ym}.csv"
        if path.exists():
            continue
        df = fetch_prescriptions_month(ym)
        df.to_csv(path, index=False)
        log.info("prescriptions %d: %s rows, %s items", ym, f"{len(df):,}", f"{df['ITEMS'].sum():,}")
        time.sleep(1)
    _combine(cache, DATA_DIR / TABLES["prescriptions"])


# Registered patients (NHS Digital)

PATIENTS_SERIES = ("https://digital.nhs.uk/data-and-information/publications/statistical/"
                   "patients-registered-at-a-gp-practice")
_MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july",
                "august", "september", "october", "november", "december"]
# the practice-level file has had several names
_PATIENT_FILES = [
    r"gp-reg-pat-prac-all[^/\"]*\.(?:csv|zip)",    # May 2017 onward (zipped by 2026)
    r"gp-reg-patients-\d{2}-\d{4}\.csv",            # Jan - Jul 2014
    r"gp_practice_counts\.csv",                     # Oct 2014 - Jul 2015
    r"gp-reg-pat(?:ients)?-prac-quin-age\d*\.csv",  # late 2015 - Apr 2017
]


def _patient_publications():
    html = _get(PATIENTS_SERIES).decode("utf-8", "ignore")
    for slug in set(re.findall(r"patients-registered-at-a-gp-practice/([a-z]+-\d{4})\"", html)):
        month, year = slug.split("-")
        if month in _MONTH_NAMES and int(year) >= FIRST[0]:
            yield int(year) * 100 + _MONTH_NAMES.index(month) + 1, slug


def _patient_file_url(slug):
    html = _get(f"{PATIENTS_SERIES}/{slug}").decode("utf-8", "ignore")
    links = re.findall(r"https://files\.digital\.nhs\.uk/[^\"]+", html)
    for pattern in _PATIENT_FILES:
        for link in links:
            if re.search(pattern + "$", link, re.I):
                return link
    return None


def read_practice_totals(raw, url):
    if url.lower().endswith(".zip"):
        with zipfile.ZipFile(_bytes_io.BytesIO(raw)) as z:
            raw = z.read(next(n for n in z.namelist() if n.lower().endswith(".csv")))
    df = pd.read_csv(_bytes_io.BytesIO(raw), dtype=str)
    df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")  # May 2017 has "Number of patients"

    if "TOTAL_ALL" in df.columns:  # wide format, one row per practice
        out = df[["GP_PRACTICE_CODE", "TOTAL_ALL"]]
    else:  # long format: keep the practice-level all-ages, all-sexes row
        code = "CODE" if "CODE" in df.columns else "ORG_CODE"
        org_type = "TYPE" if "TYPE" in df.columns else "ORG_TYPE"
        age = "AGE" if "AGE" in df.columns else "AGE_GROUP_5"
        keep = (df[org_type].str.upper() == "GP") & (df["SEX"].str.upper() == "ALL") & (df[age].str.upper() == "ALL")
        out = df.loc[keep, [code, "NUMBER_OF_PATIENTS"]]
    out.columns = ["PRACTICE_CODE", "PATIENTS"]
    return out.assign(PATIENTS=pd.to_numeric(out["PATIENTS"]))


def download_patients():
    cache = CACHE_DIR / "patients_months"
    cache.mkdir(parents=True, exist_ok=True)
    practices = set(read_table("prescriptions")["PRACTICE_CODE"])
    for ym, slug in sorted(_patient_publications()):
        path = cache / f"{ym}.csv"
        if path.exists():
            continue
        url = _patient_file_url(slug)
        if url is None:  # announced but not published yet, or a special-topic release
            log.info("patients %d: no practice file on %s", ym, slug)
            continue
        df = read_practice_totals(_get(url), url)
        df = df[df["PRACTICE_CODE"].isin(practices)]
        df.insert(0, "YEAR_MONTH", ym)
        df.to_csv(path, index=False)
        log.info("patients %d: %s practices <- %s", ym, f"{len(df):,}", url.rsplit("/", 1)[-1])
        time.sleep(1)
    _combine(cache, DATA_DIR / TABLES["patients"])


# Weather (Open-Meteo)

WEATHER_API = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_DAILY = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "precipitation_sum",
    "rain_sum",
    "sunshine_duration",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
]


def download_weather():
    cache = CACHE_DIR / "weather_towns"
    cache.mkdir(parents=True, exist_ok=True)
    start = f"{FIRST[0]}-{FIRST[1]:02d}-01"
    end = (pd.Timestamp(f"{LAST[0]}-{LAST[1]:02d}-01") + pd.offsets.MonthEnd()).strftime("%Y-%m-%d")
    for code, (town, lat, lon) in TOWNS.items():
        path = cache / f"{code}.csv"
        if path.exists():
            continue
        params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                  "daily": ",".join(WEATHER_DAILY), "timezone": "Europe/London"}
        # long requests count as several calls against the rate limit
        daily = json.loads(_get(WEATHER_API + "?" + urllib.parse.urlencode(params)))["daily"]
        df = pd.DataFrame(daily).rename(columns={"time": "date"})
        df.insert(0, "town", town)
        df.insert(0, "icb_code", code)
        df.to_csv(path, index=False)
        log.info("weather %s: %s days", town, f"{len(df):,}")
        time.sleep(15)
    _combine(cache, DATA_DIR / TABLES["weather"])


# Practice locations (postcodes.io)

def _postcode_fallback(postcode):
    # retired postcodes: terminated record, then the outcode centre
    for url, source in [
        (f"https://api.postcodes.io/terminated_postcodes/{postcode.replace(' ', '')}", "terminated postcode"),
        (f"https://api.postcodes.io/outcodes/{postcode.split()[0]}", "outcode centre"),
    ]:
        try:
            hit = json.loads(_get(url))["result"]
            return hit["latitude"], hit["longitude"], source
        except urllib.error.HTTPError:
            continue
    return None


def download_locations():
    rx = read_table("prescriptions")
    latest = rx.sort_values("YEAR_MONTH").groupby("PRACTICE_CODE")["POSTCODE"].last()
    unique = sorted(latest.unique())
    found = {}
    for i in range(0, len(unique), 100):
        result = json.loads(_get("https://api.postcodes.io/postcodes", {"postcodes": unique[i:i + 100]}))
        for row in result["result"]:
            if row["result"]:
                found[row["query"]] = (row["result"]["latitude"], row["result"]["longitude"], "postcode")
    for pc in set(unique) - set(found):
        if hit := _postcode_fallback(pc):
            found[pc] = hit
    loc = latest.rename("POSTCODE").reset_index()
    loc[["LAT", "LON", "LOCATION_FROM"]] = loc["POSTCODE"].map(found).apply(pd.Series)
    save_dataset(loc, DATA_DIR / TABLES["locations"])


def download_all():
    download_prescriptions()
    download_patients()
    download_weather()
    if not (DATA_DIR / TABLES["locations"]).exists():
        download_locations()


def read_table(name, data_dir=DATA_DIR):
    if name not in TABLES:
        raise ValueError(f"unknown table {name!r}, expected one of {list(TABLES)}")
    df = pd.read_csv(Path(data_dir) / TABLES[name], dtype=_DTYPES.get(name))
    log.debug("read %s: %s rows", name, f"{len(df):,}")
    return df


def save_dataset(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".pkl":
        df.to_pickle(path)
    else:
        df.to_csv(path, index=False)
    log.info("saved %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    download_all()
