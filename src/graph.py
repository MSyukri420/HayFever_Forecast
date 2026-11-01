"""Charts for the README, saved to docs/figures/.

    python -m src.graph      # after src.model for each target
"""

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import experiments  # noqa: E402
from . import feature as F  # noqa: E402
from . import io  # noqa: E402
from . import model as M  # noqa: E402

log = logging.getLogger(__name__)

FIGURES_DIR = M.RESULTS_DIR.parents[1] / "docs" / "figures"

GROUP_LABELS = {"antihist": "Antihistamine tablets", "nasal": "Nasal sprays", "eye": "Eye drops"}
MODEL_LABELS = {"lightgbm": "LightGBM", "pytorch": "PyTorch network",
                "last_year_lift": "Last year's lift (baseline)"}
MODELS = list(MODEL_LABELS)
CHANGE_LABELS = {
    "blend with last_year_lift": "Blend with the baseline",
    "reconcile to region ridge forecast": "Match a region-level forecast",
    "ensemble: average all variants": "Average all 5 variants",
    "ensemble: average top3 variants": "Average the best 3 variants",
    "optuna settings (25 trials on 2021-22)": "Optuna tuning",
}

# colour-blind safe; only three models are drawn so every pair stays distinct
MODEL_COLORS = {"lightgbm": "#2a78d6", "last_year_lift": "#eb6834", "pytorch": "#1baf7a"}
RANGE_COLOR = "#cde2fb"
FAMILY_COLORS = {"Calendar & policy": "#2a78d6", "Same drug's history": "#eb6834",
                 "Other drugs' history": "#1baf7a", "Weather": "#eda100"}
OTHER_COLOR = "#c3c2b7"
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SHADE = "#f0efec"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# rows for figure_data.csv
_table = []


def _record(figure, **values):
    _table.append({"figure": figure, **values})


def _subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")


def _figure_header(fig, title, subtitle, top=0.84):
    fig.subplots_adjust(top=top)
    fig.text(0.06, 0.985, title, fontsize=12, color=INK, va="top")
    fig.text(0.06, 0.985 - 0.045 * 7 / fig.get_figheight(), subtitle, fontsize=9, color=MUTED, va="top")


def _save(fig, name):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    log.info("saved %s", path)
    return path


def _dates(year_month):
    return pd.to_datetime(pd.Series(year_month).astype(int).astype(str), format="%Y%m")


def _regional(rows, cols):
    weighted = rows[cols].mul(rows["PATIENTS"], axis=0).groupby(rows["YEAR_MONTH"]).sum()
    return weighted.div(rows.groupby("YEAR_MONTH")["PATIENTS"].sum(), axis=0)


def _result(name):
    path = M.RESULTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


def _family(feature, group):
    if feature.startswith(f"{group}_"):
        return "Same drug's history"
    if any(feature.startswith(f"{g}_") for g in F.GROUPS):
        return "Other drugs' history"
    if feature == "temp_mean_3m" or any(feature.startswith(w) for w in F.WEATHER_COLS):
        return "Weather"
    if feature in ("month", "month_sin", "month_cos", "year", "otc_guidance", "covid"):
        return "Calendar & policy"
    return "Location & list size"


def plot_regional_rates():
    """Small multiples: regional rate per 1,000 patients, every month 2014-2026, per drug group."""
    path = io.DATA_DIR / "processed" / "model_table.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path, usecols=["YEAR_MONTH", "PATIENTS", *F.TARGETS])
    region = _regional(table, F.TARGETS)
    dates = _dates(region.index)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    covid = _dates(list(F.COVID_MONTHS))
    test = _dates([M.TEST_START, M.TEST_END])
    otc = _dates([F.OTC_GUIDANCE_FROM])[0]
    for ax, group in zip(axes, F.GROUPS):
        rate = region[f"per1000_{group}"].to_numpy()
        ax.axvspan(covid[0], covid[1] + pd.offsets.MonthEnd(), color=SHADE, zorder=0, linewidth=0)
        ax.axvspan(test[0], test[1] + pd.offsets.MonthEnd(), color="#cde2fb", alpha=0.45, zorder=0,
                   linewidth=0)
        ax.axvline(otc, color=AXIS, linewidth=1, linestyle="--", zorder=1)
        ax.plot(dates, rate, color=INK, linewidth=1.6)
        ax.set_title(GROUP_LABELS[group], fontsize=10)
        ax.set_ylabel("Items per 1,000")
        ax.set_ylim(0, rate.max() * 1.15)
        ax.grid(axis="y")
        for ym, v in zip(region.index, rate):
            _record("regional_rates", group=group, year_month=int(ym), per1000=round(v, 3))
    top = axes[0]
    ymax = top.get_ylim()[1]
    top.text(otc, ymax, " NHS guidance:\n stop prescribing\n for mild hay fever", fontsize=8, color=INK_2,
             va="top")
    top.text(covid[0], ymax, " COVID", fontsize=8, color=INK_2, va="top")
    top.text(test[0], ymax, " test period", fontsize=8, color=INK_2, va="top")
    _figure_header(fig, "Hay-fever prescribing in West Midlands GP practices",
                   "Items per 1,000 registered patients, every month Jan 2014 - Aug 2026 · "
                   f"{table['YEAR_MONTH'].nunique()} months, about {len(table) // table['YEAR_MONTH'].nunique()} "
                   "practices a month", top=0.89)
    fig.subplots_adjust(hspace=0.3)
    return _save(fig, "regional_rates")


def plot_forecast_vs_actual():
    """Small multiples: regional actual vs LightGBM and the lift baseline over the test period."""
    preds = {g: _result(f"{g}_predictions.csv") for g in F.GROUPS}
    scores = {g: _result(f"{g}_scores.csv") for g in F.GROUPS}
    specs = {g: M.RESULTS_DIR / f"{g}_model.json" for g in F.GROUPS}
    if (any(p is None for p in preds.values()) or any(s is None for s in scores.values())
            or not all(p.exists() for p in specs.values())):
        return None
    models = ["lightgbm", "last_year_lift"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    for ax, group in zip(axes, F.GROUPS):
        target = f"per1000_{group}"
        region = _regional(preds[group], [target, *models])
        dates = _dates(region.index)
        mape = scores[group].set_index("model")["regional_mape_pct"]
        lo, hi = json.loads(specs[group].read_text())["quantiles"]["region"]
        ax.fill_between(dates, region["lightgbm"] * (1 + lo), region["lightgbm"] * (1 + hi), color=RANGE_COLOR,
                        linewidth=0, zorder=1, label="LightGBM 80% range")
        ax.plot(dates, region[target], color=INK, linewidth=2.4, label="Actual", zorder=3)
        for m in models:
            ax.plot(dates, region[m], color=MODEL_COLORS[m], marker="o", ms=4, markeredgecolor=SURFACE,
                    markeredgewidth=1, label=f"{MODEL_LABELS[m]}", zorder=4 if m == "lightgbm" else 2)
        ax.set_title(f"{GROUP_LABELS[group]}  ·  monthly error: LightGBM {mape['lightgbm']:.1f}%, "
                     f"baseline {mape['last_year_lift']:.1f}%", fontsize=10)
        ax.set_ylabel("Items per 1,000")
        ax.grid(axis="y")
        inside = region[target].between(region["lightgbm"] * (1 + lo), region["lightgbm"] * (1 + hi))
        ax.text(1, 1.02, f"{inside.sum()} of {len(region)} months inside the range", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=INK_2)
        for ym, row in region.iterrows():
            _record("forecast_vs_actual", group=group, year_month=int(ym), actual=round(row[target], 3),
                    **{m: round(row[m], 3) for m in models}, range_lo=round(row["lightgbm"] * (1 + lo), 3),
                    range_hi=round(row["lightgbm"] * (1 + hi), 3))
    axes[-1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=4)
    _figure_header(fig, "Forecasts three months ahead vs what was prescribed",
                   f"Test period Mar 2024 - Aug 2026, never used for any choice · LightGBM retrained each "
                   f"month on data up to 3 months before · monthly error = mean absolute % error of the "
                   f"regional total", top=0.89)
    fig.subplots_adjust(hspace=0.35)
    return _save(fig, "forecast_vs_actual")


def plot_scores():
    """Dot plots: regional error per drug group, validation (2021-23) and test, model vs baselines."""
    valid = {g: _result(f"{g}_valid.csv") for g in F.GROUPS}
    scores = {g: _result(f"{g}_scores.csv") for g in F.GROUPS}
    if any(v is None for v in valid.values()) or any(s is None for s in scores.values()):
        return None
    models = MODELS
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    y = np.arange(len(F.GROUPS))[::-1]
    for ax, part in zip(axes, ["valid", "test"]):
        for i, m in enumerate(models):
            values = []
            for group in F.GROUPS:
                s = scores[group].set_index("model")
                if part == "valid":
                    chosen = s.index[s["chosen"].astype(bool)][0]
                    v = valid[group]
                    v = v[v["year"] == "mean"].set_index("model")
                    value = v.loc[chosen if m == "lightgbm" else m, M.CHOOSE_BY]
                else:
                    value = s.loc[m, M.CHOOSE_BY]
                values.append(value)
                _record("scores", part=part, group=group, model=m, regional_mape_pct=value)
            pos = y + (1 - i) * 0.22
            ax.scatter(values, pos, s=70, color=MODEL_COLORS[m], edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label=MODEL_LABELS[m])
            for v, p in zip(values, pos):
                ax.annotate(f"{v:.2f}", (v, p), xytext=(7, 0), textcoords="offset points", va="center",
                            fontsize=8, color=INK_2)
        ax.set_xlim(0, 8)
        ax.set_xlabel("Monthly error of the regional total (%)")
        ax.grid(axis="x")
        ax.set_title("Validation, mean of 2021, 2022, 2023" if part == "valid"
                     else "Test, Mar 2024 - Aug 2026", fontsize=10)
    axes[0].set_yticks(y, [GROUP_LABELS[g] for g in F.GROUPS])
    axes[0].tick_params(axis="y", length=0)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.22), ncols=3)
    _figure_header(fig, "Clear test gains for tablets and sprays; eye drops stay level with the baseline",
                   "Lower is better · validation: one model per year, trained on earlier years · test: "
                   "retrained monthly", top=0.78)
    return _save(fig, "scores")


def plot_error_by_month():
    """Small multiples: practice-level error by calendar month over the test period."""
    by_month = {g: _result(f"{g}_by_month.csv") for g in F.GROUPS}
    if any(b is None for b in by_month.values()):
        return None
    models = MODELS
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, group in zip(axes, F.GROUPS):
        b = by_month[group].set_index("month")
        ax.axvspan(min(M.PEAK_MONTHS) - 0.5, max(M.PEAK_MONTHS) + 0.5, color=SHADE, zorder=0, linewidth=0)
        for m in models:
            ax.plot(b.index, b[m], color=MODEL_COLORS[m], marker="o", ms=5, markeredgecolor=SURFACE,
                    markeredgewidth=1, label=MODEL_LABELS[m], zorder=3 if m == "lightgbm" else 2)
            for month, v in b[m].items():
                _record("error_by_month", group=group, model=m, month=int(month), mae=v)
        ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
        ax.set_ylim(0, b[models].to_numpy().max() * 1.2)
        ax.set_title(GROUP_LABELS[group], fontsize=10)
        ax.grid(axis="y")
    axes[0].set_ylabel("Mean absolute error\n(items per 1,000, per practice)")
    axes[0].text(min(M.PEAK_MONTHS) - 0.4, axes[0].get_ylim()[1], "peak\nseason", fontsize=8, color=INK_2,
                 va="top")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=3)
    fig.subplots_adjust(wspace=0.25)
    _figure_header(fig, "Practice-level error by calendar month",
                   "Test period Mar 2024 - Aug 2026 · each practice's forecast vs its own prescribing · "
                   "both models retrained monthly", top=0.8)
    return _save(fig, "error_by_month")


def plot_top_features(n=10):
    """Horizontal bars: each drug group's most-used features, coloured by family."""
    imp = {g: _result(f"{g}_importance.csv") for g in F.GROUPS}
    if any(i is None for i in imp.values()):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    shown = set()
    for ax, group in zip(axes, F.GROUPS):
        gain = imp[group].set_index(imp[group].columns[0])["shap_pct"].head(n).iloc[::-1]
        families = [_family(f, group) for f in gain.index]
        shown.update(families)
        ax.barh(gain.index, gain.to_numpy(), height=0.7,
                color=[FAMILY_COLORS.get(fam, OTHER_COLOR) for fam in families])
        for name, v, fam in zip(gain.index, gain.to_numpy(), families):
            ax.annotate(f"{v:.0f}%" if v >= 1 else f"{v:.1f}%", (v, name), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK_2)
            _record("top_features", group=group, feature=name, family=fam, shap_pct=v)
        ax.set_xlim(0, gain.max() * 1.2)
        ax.set_xlabel("Share of mean |SHAP|")
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0, labelsize=8)
        ax.set_title(GROUP_LABELS[group], fontsize=10)
    legend = {fam: c for fam, c in [*FAMILY_COLORS.items(), ("Location & list size", OTHER_COLOR)] if fam in shown}
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in legend.values()]
    axes[1].legend(handles, legend, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=len(legend))
    fig.subplots_adjust(wspace=0.55)
    _figure_header(fig, f"What the model uses: top {n} features",
                   "TreeSHAP on the test months, model refit on 2014-2023 · month_cos = season · lag12_excess = "
                   "last year's hay-fever lift for this month · winter_base = year-round use", top=0.82)
    return _save(fig, "top_features")


def plot_drivers_by_month():
    """Stacked bars: average SHAP contribution of each feature family, by calendar month."""
    shap = {g: _result(f"{g}_shap_by_month.csv") for g in F.GROUPS}
    if any(s is None for s in shap.values()):
        return None
    families = [*FAMILY_COLORS, "Location & list size"]
    colors = {**FAMILY_COLORS, "Location & list size": OTHER_COLOR}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, group in zip(axes, F.GROUPS):
        s = shap[group].set_index("calendar_month")
        by_family = pd.DataFrame({fam: s[[f for f in F.FORECAST_FEATURES if _family(f, group) == fam]].sum(axis=1)
                                  for fam in families})
        up, down = np.zeros(len(s)), np.zeros(len(s))
        for fam in families:
            v = by_family[fam].to_numpy()
            bottom = np.where(v >= 0, up, down)
            ax.bar(s.index, v, bottom=bottom, width=0.75, color=colors[fam], edgecolor=SURFACE, linewidth=1,
                   label=fam)
            up, down = up + np.clip(v, 0, None), down + np.clip(v, None, 0)
            for month, value in zip(s.index, v):
                _record("drivers_by_month", group=group, month=int(month), family=fam, shap_per1000=round(value, 3))
        net = by_family.sum(axis=1).to_numpy()
        ax.plot(s.index, net, color=INK, marker="o", ms=4, linewidth=1.5, label="net effect", zorder=3)
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
        ax.set_title(GROUP_LABELS[group], fontsize=10)
        ax.grid(axis="y")
    axes[0].set_ylabel("Items per 1,000 above or below\nthe model's average forecast")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=6)
    fig.subplots_adjust(wspace=0.25)
    _figure_header(fig, "What drives the summer peak",
                   "Average TreeSHAP contribution by calendar month, test period Mar 2024 - Aug 2026 · above 0 "
                   "pushes the forecast above the model's average · line = all features together", top=0.8)
    return _save(fig, "drivers_by_month")


def plot_experiments():
    """Dot plots: validation change of each tried idea vs the reference run, per drug group."""
    runs = experiments.load_log()
    if runs.empty:
        return None
    years = [f"val_{y}" for y in M.VALID_YEARS]
    changes = [c for c in CHANGE_LABELS if c in set(runs["change"])]
    limit = 30
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=True)
    y = np.arange(len(changes))[::-1]
    for ax, group in zip(axes, F.GROUPS):
        mine = runs[runs["target"] == group]
        ref = mine[mine["change"] == "reference"]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        ax.axvspan(-limit, -experiments.MIN_GAIN * 100, color=SHADE, zorder=0, linewidth=0)
        ax.axvline(0, color=AXIS, linewidth=1, zorder=1)
        for yy, change in zip(y, changes):
            run = mine[mine["change"] == change]
            if run.empty:
                continue
            run = run.iloc[-1]
            per_year = (run[years].to_numpy(float) / ref[years].to_numpy(float) - 1) * 100
            mean = (run["val_mean"] / ref["val_mean"] - 1) * 100
            ax.scatter(np.clip(per_year, -limit, limit), np.full(len(years), yy), s=18, color=MUTED, zorder=2,
                       linewidth=0, label="one validation year" if yy == y[0] else None)
            rechecked = "seed check" in str(run["note"])
            ax.scatter(min(mean, limit), yy, s=70, zorder=3, linewidth=1.5,
                       color=SURFACE if rechecked else MODEL_COLORS["lightgbm"],
                       edgecolor=MODEL_COLORS["lightgbm"] if rechecked else SURFACE,
                       label="mean of the 3 years" if yy == y[0] else None)
            text = f"{mean:+.0f}%" + (" (failed seed recheck)" if rechecked else "")
            ax.annotate(text, (min(mean, limit), yy), xytext=(0, 8), textcoords="offset points",
                        ha="center" if mean < limit else "right", fontsize=8, color=INK_2)
            _record("experiments", group=group, change=change, mean_change_pct=round(mean, 1),
                    **{f"{c}_change_pct": round(v, 1) for c, v in zip(years, per_year)},
                    kept=bool(run["kept"]))
        ax.set_xlim(-limit, limit)
        ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
        ax.set_xlabel("Validation error vs the model as is (%)")
        ax.grid(axis="x")
        ax.set_title(GROUP_LABELS[group], fontsize=10)
    axes[0].set_yticks(y, [CHANGE_LABELS[c] for c in changes])
    axes[0].tick_params(axis="y", length=0)
    axes[0].text(-limit + 1, y.max() + 0.55, f"keep zone: every year\nbetter, mean -{experiments.MIN_GAIN:.0%}",
                 fontsize=7, color=INK_2, va="top")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2)
    _figure_header(fig, "Ideas tried after the main model: none kept",
                   f"Regional error on the 3 validation years, change vs the reference run · kept only if "
                   f"better in every year and by more than {experiments.MIN_GAIN:.0%} on the mean · "
                   f"points past ±{limit}% are drawn at the edge", top=0.8)
    return _save(fig, "experiments")


def make_all():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _table.clear()
    charts = {
        "regional_rates": plot_regional_rates,
        "forecast_vs_actual": plot_forecast_vs_actual,
        "scores": plot_scores,
        "error_by_month": plot_error_by_month,
        "top_features": plot_top_features,
        "drivers_by_month": plot_drivers_by_month,
        "experiments": plot_experiments,
    }
    written = []
    for name, make in charts.items():
        path = make()
        if path is None:
            log.warning("skipped %s: its inputs are missing", name)
        else:
            written.append(path)
    pd.DataFrame(_table).to_csv(FIGURES_DIR / "figure_data.csv", index=False)
    return written


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    make_all()


if __name__ == "__main__":
    main()
