← [Back to README](../README.md)

# Hay Fever Prescribing Forecast: technical details

## How the forecast works

The data is one row per practice and month. The target is items prescribed per 1,000 registered patients, for three drug groups (BNF chemical-substance prefixes):

| Group | BNF prefix | Column |
|---|---|---|
| Antihistamine tablets | `030401` | `per1000_antihist` |
| Nasal sprays (corticosteroids) | `120201` | `per1000_nasal` |
| Eye drops (anti-allergy) | `110402` | `per1000_eye` |

**Forecast setup.** Prescribing data is published about two months late. So a forecast made at the end of month *o* can only see data up to *o*, and it predicts month *t = o + 3* (`HORIZON = 3`). Every forecast feature uses data from *t − 3* or earlier. Calendar columns and the policy and COVID flags are known in advance, so they describe month *t* itself.

**Features** (42 columns; `<g>` = antihist, nasal or eye, for all three groups):

| Feature | Meaning |
|---|---|
| `<g>_lag3`, `_lag4`, `_lag5`, `_mean3` | the latest three known months, and their mean |
| `<g>_lag12`, `_lag24` | the same month one and two years earlier |
| `<g>_winter_base` | mean Dec–Feb rate of the latest winter fully known at *t − 3*: year-round use (eczema, itch, sleep), not hay fever |
| `<g>_lag12_excess` | `lag12` minus the winter base known then: last year's hay-fever rise for this month |
| `<weather>_lag3`, `temp_mean_3m` | weather at *t − 3* (8 measures), and mean temperature over *t − 5* to *t − 3* |
| `log_patients_lag3`, `LAT`, `LON` | list size and practice location |
| `month`, `month_sin`, `month_cos`, `year` | calendar |
| `otc_guidance`, `covid` | 1 from Mar 2018 (NHS guidance on over-the-counter medicines); 1 for Mar 2020 – Dec 2021 |

**Baselines:**

```
last_year       = rate in the same month last year
last_year_lift  = this year's winter base + last year's lift over its winter
                = winter_base + lag12_excess
```

**LightGBM variants.** All use the same features. They differ in loss and in what they predict:

| Variant | Loss | Predicts |
|---|---|---|
| `lgb_l1` | absolute | rate |
| `lgb_l2` | squared | rate |
| `lgb_poisson` | Poisson | rate |
| `lgb_l2_excess` | squared | rate − winter base. Trees can't predict past the levels they were trained on; adding the base back lets a rising level carry into the forecast |
| `lgb_l2_ratio` | squared | rate ÷ winter base |

**Choosing a variant.** There is one validation fold per year: 2021, 2022 and 2023. Each fold trains on all earlier years and scores its own year; early stopping on that year picks the tree count. The chosen variant has the lowest regional error averaged over the three years. The test months are never used to choose.

**Refit and test.** Every variant is refit on 2014–2023, using the tree count from the 2023 fold. The chosen variant is then retrained every month of the test (`lightgbm`): the forecast for month *t* comes from a model trained on targets up to *t − 3* only. The test starts in March 2024, because the first forecast the end-2023 data allows is for March.

**PyTorch network** (`src/nn.py`). This is a second model family on the same 42 features: a feed-forward network (42 inputs → 128 → 64 → 1, ReLU, 10% dropout). Like `lgb_l2_excess`, it predicts the rise above the practice's winter base. Inputs are standardised on the training rows; a missing value becomes 0 plus a 0/1 "was missing" column. It's trained with Adam on squared loss, with early stopping on each validation year. It goes through the same three validation folds and the same monthly retraining on the test as LightGBM. Three other training settings were tried on validation (lower learning rates, larger batches); none was clearly better, so the defaults stay. It's reported next to LightGBM but isn't part of the choice.

**Prediction intervals** (80%). These are split conformal, calibrated on the chosen variant's validation forecasts (2021–2023):

```
practice:  score = (actual − forecast) / sqrt(forecast × 1000 / patients)
           range = forecast + [10th, 90th percentile of score for that calendar month] × sqrt(...)
region:    range = regional forecast × (1 + [10th, 90th percentile of the relative error
                   of the regional total over the 36 validation months])
```

The `sqrt(rate × 1000 / patients)` term is how much a practice's rate would vary by chance if its items were Poisson. It gives a small practice a wider range than a large one with the same rate.

**SHAP.** These are TreeSHAP contributions of the chosen LightGBM model (LightGBM's own `pred_contrib`), converted to items per 1,000 so they add up to each forecast. They feed the importance chart, the drivers-by-month chart and the top 3 drivers that the API returns with each forecast.

**Metrics:**

```
regional error (%)  = mean over months of |predicted − actual| / actual × 100,
                      where both are the West Midlands total (rate × patients, summed)
regional bias (%)   = (total predicted / total actual − 1) × 100 over the period
MAE                 = mean absolute error per practice-month, in items per 1,000
```

Regional error is the headline metric and the one used to choose. Every model is scored on the same rows: those where both baselines have a value.

## Evaluation

### Test, Mar 2024 – Aug 2026 (30 months, 21,600 practice-months per group)

| Group | Model | Regional error | Regional bias | MAE | MAE May–Jul | MAE Aug–Apr |
|---|---|---|---|---|---|---|
| Antihistamines | **LightGBM** (l2_excess, monthly) | **2.88%** | −0.6% | 1.91 | 2.27 | 1.76 |
| | l2_excess, frozen at end 2023 | 3.26% | −1.5% | 1.94 | 2.27 | 1.80 |
| | last_year_lift | 3.86% | −0.7% | 2.71 | 2.83 | 2.66 |
| | PyTorch (monthly) | 3.32% | −0.3% | 1.94 | 2.35 | 1.76 |
| | last_year_lift | 3.86% | −0.7% | 2.71 | 2.83 | 2.66 |
| | last_year | 5.17% | −4.2% | 2.61 | 2.90 | 2.48 |
| Nasal sprays | **LightGBM** (l2_excess, monthly) | **2.88%** | −0.2% | 1.16 | 1.28 | 1.10 |
| | l2_excess, frozen | 2.98% | −0.6% | 1.16 | 1.28 | 1.11 |
| | PyTorch (monthly) | 3.40% | −0.4% | 1.16 | 1.26 | 1.11 |
| | last_year_lift | 3.77% | −0.2% | 1.71 | 1.77 | 1.68 |
| | last_year | 3.86% | −1.3% | 1.56 | 1.68 | 1.51 |
| Eye drops | **LightGBM** (l2_ratio, monthly) | **5.69%** | +0.5% | 0.43 | 0.57 | 0.37 |
| | l2_ratio, frozen | 5.86% | −0.1% | 0.44 | 0.58 | 0.38 |
| | PyTorch (monthly) | 6.51% | −2.7% | 0.44 | 0.57 | 0.38 |
| | last_year_lift | 5.48% | −0.2% | 0.59 | 0.70 | 0.54 |
| | last_year | 6.06% | −2.3% | 0.54 | 0.68 | 0.48 |

### How sure is the gain?

There are only 30 test months, so each gap below comes with a 95% range. The range comes from a moving-block bootstrap: 10,000 resamples of the test months, drawn in blocks of 3 consecutive months, because errors in neighbouring months are related (an early or late season moves several months at once). Both models are scored on the same resampled months. Gap = the other model's regional error minus LightGBM's, so above 0 means LightGBM is better.

| Group | Compared with | Gap (points) | 95% range | LightGBM better in |
|---|---|---|---|---|
| Antihistamines | last_year_lift | 0.98 | 0.26 to 1.93 | 16 of 30 months |
| | last_year | 2.29 | 1.48 to 3.14 | 23 of 30 |
| | PyTorch | 0.44 | −0.56 to 1.55 | 18 of 30 |
| Nasal sprays | last_year_lift | 0.89 | 0.37 to 1.53 | 19 of 30 |
| | last_year | 0.98 | 0.55 to 1.41 | 22 of 30 |
| | PyTorch | 0.52 | −0.32 to 1.32 | 18 of 30 |
| Eye drops | last_year_lift | −0.22 | −1.40 to 1.34 | 14 of 30 |
| | last_year | 0.37 | −0.92 to 1.91 | 16 of 30 |
| | PyTorch | 0.82 | −0.80 to 2.57 | 18 of 30 |

For tablets and sprays the gain over both baselines is clear: the whole range is above 0, an 8–42% lower error against the lift baseline for tablets and 11–37% for sprays. For eye drops, and for LightGBM against PyTorch, the ranges cross 0, so the test can't separate them. Also, a lower average doesn't mean winning most months: against the lift baseline, LightGBM is better in only 16 of 30 months for tablets. Its gain comes from fewer large misses.

### Prediction intervals (80%)

How often the actual value fell inside the range on the test (`output/results/<group>_coverage.csv`). Width is in items per 1,000, or % of the forecast for the regional total.

| Slice | Antihistamines | Nasal sprays | Eye drops |
|---|---|---|---|
| All practice-months | 81% (width 6.6) | 81% (3.9) | 82% (1.5) |
| Smallest 25% of practices | 84% (9.4) | 84% (5.2) | 83% (2.0) |
| Largest 25% of practices | 77% (4.1) | 77% (2.6) | 80% (0.9) |
| Calendar months, lowest to highest | 77–86% | 76–84% | 78–86% |
| Regional total, 30 months | 93% (11.5%) | 93% (11.7%) | 80% (18.7%) |

The practice ranges hold at about 80% for every practice size and month. The regional range is cautious for tablets and sprays (93%): it was set on frozen validation models, which miss by more than the monthly-retrained test model.

### Validation, regional error per year

| Group | Model | 2021 | 2022 | 2023 | Mean |
|---|---|---|---|---|---|
| Antihistamines | lgb_l2_excess (chosen) | 4.32 | 3.77 | 3.59 | 3.89 |
| | last_year_lift | 3.53 | 5.25 | 2.93 | 3.90 |
| | last_year | 3.98 | 5.03 | 3.94 | 4.32 |
| | PyTorch | 4.90 | 5.38 | 4.10 | 4.79 |
| Nasal sprays | lgb_l2_excess (chosen) | 5.17 | 3.55 | 3.01 | 3.91 |
| | last_year_lift | 7.33 | 5.18 | 3.53 | 5.35 |
| | last_year | 5.04 | 4.72 | 3.25 | 4.34 |
| | PyTorch | 5.28 | 4.25 | 3.93 | 4.49 |
| Eye drops | lgb_l2_ratio (chosen) | 6.94 | 5.74 | 6.79 | 6.49 |
| | PyTorch | 8.57 | 9.51 | 5.61 | 7.89 |
| | last_year_lift | 6.87 | 8.36 | 4.95 | 6.73 |
| | last_year | 7.46 | 8.39 | 4.74 | 6.86 |

The per-year scores for every variant are in `output/results/<group>_valid.csv`.

How to read this:

- **Practice-level error falls by 20–27% in every group.** LightGBM's MAE is lower than the better baseline's in every group (1.91 vs 2.61, 1.16 vs 1.56, 0.43 vs 0.54) and in every calendar month ([chart](figures/error_by_month.png)). That includes eye drops, where it doesn't win on the regional total.
- **PyTorch matches LightGBM per practice but not on the regional total.** Their practice MAE is within 0.03 in every group, but the network's regional error is 0.4–0.8 points higher. Its practice errors cancel out less when they're added up, so it misses the monthly total by more. It wasn't tuned beyond four settings. On a table like this, gradient-boosted trees are usually the stronger choice, and this result agrees.
- **Validation and test measure slightly different things.** Validation scores a model frozen for a year; the test scores the monthly-retrained model. Retraining is worth 0.1–0.4 points of regional error on the test. So on validation, antihistamine LightGBM only ties the lift baseline (3.89 vs 3.90), and it pulls ahead on the test (2.88 vs 3.86).
- **One year is a noisy score.** One year is only 12 monthly regional errors, with a standard error of about 0.7–0.9 points. Variants that look different in one year often swap order in the next. Read single-year columns with that in mind.
- **Error by practice size** (antihistamines, LightGBM): smallest quarter 2.54, then 2.05, 1.73, largest quarter 1.34. Error by weather town ranges from 1.46 (Hereford) to 2.05 (Birmingham).

### Feature importance

This is the share of mean |SHAP| on the test months ([chart](figures/top_features.png)). For antihistamines: winter base 22%, season (`month_cos`) 16%, latest month 13%, recent temperature (`temp_mean_3m`) 7%, last year's lift 6%. The winter base leads in all three groups (25% for nasal sprays, 31% for eye drops), because it carries the differences between practices. LightGBM's gain measure ranks the season first instead (32% for antihistamines), because it counts how much each feature helps the splits, not how much it moves the forecasts. Both measures are in `<group>_importance.csv`.

**What drives the summer peak** ([chart](figures/drivers_by_month.png)). In June the antihistamine forecast is 3.5 items per 1,000 above the model's average. The calendar month adds 3.2 of that, and the lagged weather 1.3, while the practice's own recent history pulls it down by 0.9. Weather pushes forecasts up from March to June and down from August to December. Part of that is still the season, since lagged temperatures rise and fall with it. Nasal sprays and eye drops follow the same pattern on a smaller scale (June: +1.5 and +0.8).

### Ideas tried and rejected

All were scored on the three validation years against the chosen model. To be kept, a change had to beat the reference in **every** year and cut the mean by more than 4% (about one standard error of a three-year difference). The full log is `output/experiments.csv`; [chart](figures/experiments.png).

| Idea | Result |
|---|---|
| Spring warmth (degree days since 1 Jan) as a pollen stand-in | Worse May–Jul for all groups. June follows June's own weather, which isn't known 3 months ahead |
| Blend with `last_year_lift` (weight chosen on the other two years) | Antihistamines 3.89 → 3.83 but worse in 2022; nasal and eye worse. The best weight swung from 0.3 to 1.0 between years |
| Scale practice forecasts to a region-level ridge model | Antihistamines 4.56, eye drops 10.7; nasal level. About 70 region months is too few |
| Average all five variants | Antihistamines and nasal 3.76, but all of the gain came from 2022 (worse in 2021 and 2023); eye worse |
| Average the best three variants | Worse or level |
| Optuna tuning (25 trials, searching on 2021–22 only) | Antihistamines first scored 3.66, better in every year. Over 5 random seeds the gain was 3.86 → 3.78, and 2023 (unseen by the search) got worse, 3.46 → 3.60. Rejected. Nasal worse in 2023; eye drops: no trial beat the defaults |

Changing only the random seed moves one year's score by 0.1–0.2 and the three-year mean by about 0.05. So check any candidate on several seeds before keeping it.

## Data

| Table | Source | Detail |
|---|---|---|
| Prescriptions | [NHSBSA English Prescribing Dataset](https://opendata.nhsbsa.net/dataset/english-prescribing-data-epd) | One SQL query per month through the CKAN `datastore_search_sql` API, grouped by practice and drug group. From Mar 2025 the tables are `EPD_SNOMED_*` |
| Registered patients | NHS Digital [Patients Registered at a GP Practice](https://digital.nhs.uk/data-and-information/publications/statistical/patients-registered-at-a-gp-practice) | Quarterly to Apr 2017, then monthly; Oct 2017 and Apr 2018 are missing. Five file layouts over the years |
| Weather | [Open-Meteo historical archive](https://open-meteo.com/en/docs/historical-weather-api) (ERA5) | Daily, for 7 towns: Birmingham, Wolverhampton, Coventry, Worcester, Hereford, Stoke-on-Trent, Shrewsbury. Aggregated to 8 monthly measures |
| Practice locations | [postcodes.io](https://postcodes.io/) | Latitude and longitude of each practice's latest postcode, for the nearest weather town |

The West Midlands is defined by postcode area: B, CV, DY, WS, WV, WR, HR, ST, TF and SY. The NHS data is under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/); Open-Meteo data is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Cleaning and joining (`src/preprocess.py`) is done in DuckDB SQL, which queries the raw CSVs in place. Each step is a view:

1. `rx_registered`: drop prescribing codes with no registered list (walk-in, out-of-hours and so on): 1.2% of items.
2. `items`: pivot to one row per practice-month, with `SUM(ITEMS) FILTER (WHERE BNF_GROUP = ...)`. A group a practice didn't prescribe that month counts as 0 items.
3. `items_patients`: join the list sizes. Practice-months under 1,000 patients are dropped, where per-patient rates mean little (430 practice-months).
4. `practice_towns`: the nearest weather town for each practice, with `arg_min` over a practice × town cross join (median 8.3 km, max 38.8 km). Hereford was added because rural Herefordshire is 40–55 km from Worcester.
5. `model_table`: rates per 1,000, location and that town's monthly weather.

Two steps stay in pandas and are registered into DuckDB as tables: filling list sizes between publications (linear interpolation within each practice's lifetime) and the monthly weather means. DuckDB adds up a month's days in a different order, which flipped the rounding of 84 of 120,429 weather values that fall exactly between two hundredths. Keeping these means in pandas makes the SQL build identical to the original pandas build, so no reported result moved.

The result is 120,429 practice-months from 936 practices over 152 months (Jan 2014 – Aug 2026). In the last month there are 717 practices, with 6.5 million registered patients.

**Known small leak:** before May 2017, list sizes between quarterly publications are interpolated, so `log_patients_lag3` can borrow from the next quarter. List sizes move by well under 1% a quarter.

## Running and retraining

Python 3.12. PyTorch's CPU build is enough: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

```bash
pip install -r requirements.txt

# download any missing data, then build dataset/processed/model_table.csv and features.csv
# (first run about 15 minutes, mostly API calls; then seconds from the cache)
python -m src.run_all

# tests: leakage, table build, experiment log, API
python -m pytest

# validate, choose a variant, refit, and score the test for one drug group,
# with the PyTorch comparison (about 4 minutes each; --no-nn skips PyTorch)
python -m src.model --target antihist
python -m src.model --target nasal
python -m src.model --target eye

# 95% ranges for the gains, and prediction-interval coverage on the test
python -m src.evaluate

# train on all the data and forecast the next 3 months for every practice
python -m src.forecast

# serve the forecasts, then open http://127.0.0.1:8000/docs
uvicorn src.api:app

# charts and figure_data.csv into docs/figures/
python -m src.graph

# star schema for Power BI into output/powerbi/ (after all three src.model runs and src.forecast)
python -m src.export

# rewrite the Power BI project in powerbi/ (model, measures and pages); then open HayFever.pbip and Refresh
python -m src.pbip
```

The API also runs in Docker. The image holds only the API and the forecast files (pandas, FastAPI, uvicorn; no LightGBM or PyTorch), so it is small and a new forecast means a new image:

```bash
python -m src.forecast
docker build -t hayfever-api .
docker run -p 8000:8000 hayfever-api      # then http://127.0.0.1:8000/docs
```

GitHub Actions ([`.github/workflows/tests.yml`](../.github/workflows/tests.yml)) runs the tests on every push and pull request, with the versions pinned in `requirements.txt`. The table-build, keep-rule and API tests run there. The leakage tests need the downloaded data, so they skip in CI and run locally after `src.run_all`.

The Power BI report, its pages, measures and the SPC chart of forecast error are described in [powerbi.md](powerbi.md).

API endpoints:

| Endpoint | Returns |
|---|---|
| `GET /health` | the last month of data, the forecast months and the model used per group |
| `GET /practices?town=Coventry` | practices with a forecast, optionally for one weather town |
| `GET /practices/{code}/forecast?group=antihist` | rate per 1,000 and items for the next 3 months, the 80% range, and the 3 largest SHAP drivers |
| `GET /region/forecast?group=antihist` | the West Midlands total per month, with its 80% range |

For trying ideas without looking at the test:

```bash
# validation only (about a minute), logged to output/experiments.csv
python -m src.model --target antihist --valid-only --change "what changed"

# best kept run per target, or every run
python -m src.experiments --summary
python -m src.experiments --summary --all

# Optuna search on the 2021-22 folds, then validate the result on all three years
python -m src.tune --target antihist
python -m src.model --target antihist --valid-only --params latest-tune --change "optuna settings"
```

## Project layout

```
src/
  io.py             downloads (NHSBSA, NHS Digital, Open-Meteo, postcodes.io), cached per month/town
  preprocess.py     DuckDB SQL: cleaning and joining into one practice-month table
  feature.py        forecast features, all from t-3 or earlier
  model.py          baselines, LightGBM variants, validation, monthly retraining, intervals, SHAP
  nn.py             the PyTorch network, validated and retrained the same way
  evaluate.py       block-bootstrap ranges for the gains, interval coverage
  forecast.py       the deployed model: train on everything, forecast the next 3 months
  api.py            FastAPI service for the forecasts
  export.py         DuckDB SQL: star schema and SPC table for Power BI
  pbip.py           writes the Power BI project: TMDL model, DAX measures, PBIR pages
  experiments.py    experiment log and the keep rule
  tune.py           Optuna search on the 2021-22 folds
  graph.py          charts for the README
  run_all.py        download -> table -> features
tests/
  test_leakage.py       cutoff test: noise after month C must not change features up to C + 3
  test_preprocess.py    the SQL table build on a tiny hand-made dataset
  test_experiments.py   keep-rule tests
  test_api.py           API responses on a tiny forecast folder
docs/
  technical.md      this file
  powerbi.md        the Power BI report: pages, tables, model, DAX measures, the SPC chart
  figures/          charts, Power BI screenshots and figure_data.csv (every plotted number)
powerbi/            the Power BI project (HayFever.pbip), written by src.pbip
Dockerfile          the API image (with .dockerignore)
.github/workflows/  tests on every push

produced by the scripts (not in git):
dataset/
  cache/            raw per-month and per-town downloads
  processed/        model_table.csv, features.csv
output/
  results/          per-group scores, predictions, importance, breakdowns, the model choice
  forecast/         the deployed forecasts, served by the API
  powerbi/          the star schema for the report
  experiments.csv   every logged validation run
```

## Outputs

`output/results/`, per drug group, after `python -m src.model`:

- `<group>_valid.csv`: validation scores per year and their mean, for every model
- `<group>_scores.csv`: test scores, one row per model
- `<group>_predictions.csv`: every test forecast, per practice and month
- `<group>_by_month.csv`, `<group>_by_group.csv`: test MAE by calendar month, by weather town and by list-size quarter
- `<group>_monthly_bias.csv`: regional % error for each test month
- `<group>_importance.csv`: share of mean |SHAP| and of LightGBM gain per feature
- `<group>_shap_by_month.csv`: average SHAP contribution per feature by calendar month
- `<group>_model.json`: the chosen variant, tree count, settings and interval calibration (read by `src.forecast`)
- `<group>_gain_ci.csv`, `<group>_coverage.csv`: from `src.evaluate`, the gap ranges and interval coverage above

`output/forecast/`, after `python -m src.forecast`: `forecasts.csv` (per practice, group and month: rate, items, 80% range, top 3 SHAP drivers), `regional.csv` (West Midlands totals with ranges) and `meta.json`. The API serves these.

`output/powerbi/`, after `python -m src.export`: three dimension tables (practice, month, drug group), four fact tables (prescribing history, test forecasts, next-3-month forecasts, regional totals) and `fact_error_spc.csv`, the control chart of the monthly regional error. Columns are listed in [powerbi.md](powerbi.md#tables).

Charts in [`figures/`](figures/): [forecast vs actual](figures/forecast_vs_actual.png), [regional rates](figures/regional_rates.png), [scores](figures/scores.png), [error by month](figures/error_by_month.png), [top features](figures/top_features.png), [drivers by month](figures/drivers_by_month.png), [ideas tried](figures/experiments.png), and [`figure_data.csv`](figures/figure_data.csv). The charts show three models (LightGBM, PyTorch and the lift baseline), because three colours are what stays distinct for colour-blind readers when every pair is compared. The `last_year` baseline is in the tables.

## Version history

- **v3 (current).** Built out from the forecast:
  - the table build moved to DuckDB SQL, and checked to give an identical table
  - a PyTorch network as a second model family
  - 80% prediction intervals, checked for coverage on the test
  - block-bootstrap ranges for every gain
  - SHAP explanations
  - a deployed forecast for the next 3 months, served by a FastAPI API in a Docker image, with tests run by GitHub Actions
  - a four-page Power BI report on a star schema from DuckDB SQL, with an SPC chart of the forecast error

  The LightGBM results did not change.
- **v2.** The model is chosen on three validation years (2021–2023), not one. Ideas are logged and kept only if they win in every year by more than 4%. The three-year choice changed two of the chosen variants: antihistamines from `lgb_poisson` to `lgb_l2_excess`, and eye drops from `lgb_l2_excess` to `lgb_l2_ratio`. Five ideas were tried under this rule and none was kept. One of them, Optuna tuning for antihistamines, did pass the rule at first. Re-run over five random seeds, most of its gain disappeared and 2023, the year the search never saw, got worse. So it was rejected, and any candidate is now checked on several seeds before it is kept.
- **v1 (original).** It chose the variant and the tree count on 2023 alone. Measuring the noise later showed the flaw. One year's regional error is the mean of only 12 months, with a standard error of 0.7–0.9 points, which is bigger than the gaps between the variants it was choosing between. So the chosen variant partly reflected chance. That led to the v2 rebuild of model selection.
