"""
build_artifacts_v2.py
---------------------
Post-training utility: re-generates everything that depends on the trained
models without retraining them. Run this whenever the figures are deleted
or new statistics are needed.

Produces:
  1.  reports/figures/*.png   (all 10 visualisation figures)
  2.  models/station_travel_times_v2.pkl
        dict[(direction, current_crs, dest_crs)] -> mean minutes
        Used by conversation_live.py to estimate scheduled arrival time
        from the actual data rather than a hardcoded guess.
  3.  reports/detailed_statistics.json
        Rich descriptive statistics suitable for direct quoting in the
        written report (dataset size, delay quantiles, per-station means,
        per-month means, peak vs off-peak comparison, model rankings).

Run with:
    python -m task2.build_artifacts_v2
"""

from __future__ import annotations

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from task2.preprocess_v2 import load_and_preprocess_v2

logger = logging.getLogger(__name__)

# ─── Paths ─────────────────────────────────────────────────────────────────
_ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR    = os.path.join(_ROOT, "models")
_FIGURES_DIR   = os.path.join(_ROOT, "reports", "figures")
_STATS_PATH    = os.path.join(_ROOT, "reports", "detailed_statistics.json")
_TRAVEL_TIMES  = os.path.join(_MODELS_DIR, "station_travel_times_v2.pkl")

_DAY_NAMES   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Plain, neutral colour scheme — no fancy palettes
_BAR_COLOUR  = "steelblue"
_BEST_COLOUR = "darkorange"

os.makedirs(_FIGURES_DIR, exist_ok=True)
os.makedirs(_MODELS_DIR,  exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  1.  Station-to-station travel time lookup
# ═══════════════════════════════════════════════════════════════════════════

def _load_raw_data_for_travel_times() -> pd.DataFrame:
    """
    Load the eight raw xlsx files WITHOUT the planned_arrival/planned_departure
    null-filtering that preprocess_v2 applies. We need origin rows (no
    planned_arrival) and terminus rows (no planned_departure) to compute
    full-route travel times such as WAT -> WEY.
    """
    data_dir = os.path.join(_ROOT, "data")
    files = [
        ("2022_WEY2WAT.xlsx", 0), ("2022_WAT2WEY.xlsx", 1),
        ("2023_WEY2WAT.xlsx", 0), ("2023_WAT2WEY.xlsx", 1),
        ("2024_WEY2WAT.xlsx", 0), ("2024_WAT2WEY.xlsx", 1),
        ("2025_WEY2WAT.xlsx", 0), ("2025_WAT2WEY.xlsx", 1),
    ]
    parts = []
    for fname, direction in files:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            logger.warning("Missing data file %s — skipping", fname)
            continue
        logger.info("  Loading %s for travel-time lookup", fname)
        df_part = pd.read_excel(path)
        df_part["direction"] = direction
        parts.append(df_part)
    return pd.concat(parts, ignore_index=True)


def build_station_travel_times(df_raw: pd.DataFrame = None) -> dict:
    """
    Compute the mean planned travel time in minutes between every pair of
    stations observed together in the same service, per direction.

    For each service-instance we walk through all (earlier, later) stop
    pairs - not just consecutive ones - so long pairs like WAT -> WEY
    are captured directly with their real timetabled travel time.

    Loads the raw xlsx files internally to preserve origin and terminus
    rows (which preprocess_v2 drops because they have NaN
    planned_arrival or planned_departure respectively).

    Returns
    -------
    dict keyed by (direction:int, from_crs:str, to_crs:str) -> mean_minutes:float
    """
    logger.info("Computing pairwise planned travel times within each service…")
    df = _load_raw_data_for_travel_times()
    logger.info("  Loaded %s raw rows", f"{len(df):,}")

    def _to_min(t):
        if hasattr(t, "hour"):
            return t.hour * 60 + t.minute + t.second / 60.0
        return np.nan

    df["pa_min"] = df["planned_arrival_time"].apply(_to_min)
    df["pd_min"] = df["planned_departure_time"].apply(_to_min)

    # Group by service-instance (rid = Real-time Train Run ID, unique per train run)
    if "rid" in df.columns:
        group_key = ["rid"]
    elif "scheduled_journey_id" in df.columns:
        group_key = ["scheduled_journey_id"]
    elif "service_id" in df.columns:
        group_key = ["service_id"]
    else:
        df["service_key"] = df["date_of_service"].astype(str)
        group_key = ["service_key", "direction"]

    pair_minutes = {}  # (direction, from_crs, to_crs) -> list[minutes]

    for _, service_df in df.groupby(group_key):
        service_df = service_df.sort_values("pd_min", kind="stable")
        rows = service_df[["location", "pa_min", "pd_min", "direction"]].to_dict("records")
        n = len(rows)
        for i in range(n):
            a = rows[i]
            dep_a = a["pd_min"]
            if pd.isna(dep_a):
                continue
            for j in range(i + 1, n):
                b = rows[j]
                arr_b = b["pa_min"]
                if pd.isna(arr_b):
                    continue
                travel = arr_b - dep_a
                if travel < -720:
                    travel += 1440
                if travel > 720:
                    travel -= 1440
                # Sanity bounds: 1 minute to 6 hours
                if travel < 1 or travel > 360:
                    continue
                key = (int(a["direction"]), a["location"], b["location"])
                pair_minutes.setdefault(key, []).append(travel)

    # Mean per pair, require at least 5 observations for robustness
    mean_pair = {k: float(np.mean(v)) for k, v in pair_minutes.items() if len(v) >= 5}
    logger.info("  Computed %d pairwise travel times across the data", len(mean_pair))

    return mean_pair


# ═══════════════════════════════════════════════════════════════════════════
#  2.  Figure regeneration (10 figures)
# ═══════════════════════════════════════════════════════════════════════════

def _savefig(name: str):
    path = os.path.join(_FIGURES_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved → %s", name)


def figure_delay_distribution(df):
    delays = df["delay_minutes"].clip(-20, 60)
    mean_v   = df["delay_minutes"].mean()
    median_v = df["delay_minutes"].median()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(delays, bins=80, color=_BAR_COLOUR)
    ax.axvline(mean_v,   color="red",   lw=1.5, linestyle="--", label=f"Mean: {mean_v:.1f} min")
    ax.axvline(median_v, color="black", lw=1.5, linestyle="-.", label=f"Median: {median_v:.1f} min")
    ax.set_xlabel("Delay (minutes)")
    ax.set_ylabel("Number of stops")
    ax.set_title("Distribution of Arrival Delays (2022-2025)")
    ax.legend()
    _savefig("delay_distribution.png")


def figure_delay_by_month(df):
    data = [df.loc[df["month"] == m, "delay_minutes"].clip(-10, 40).values for m in range(1, 13)]
    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor(_BAR_COLOUR); patch.set_alpha(0.7)
    ax.set_xticklabels(_MONTH_NAMES)
    ax.set_xlabel("Month"); ax.set_ylabel("Delay (minutes)")
    ax.set_title("Arrival Delay by Month")
    ax.axhline(0, color="gray", lw=0.8, linestyle="--")
    _savefig("delay_by_month.png")


def figure_delay_by_day(df):
    data = [df.loc[df["day_of_week"] == d, "delay_minutes"].clip(-10, 40).values for d in range(7)]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor(_BAR_COLOUR); patch.set_alpha(0.7)
    ax.set_xticklabels(_DAY_NAMES)
    ax.set_xlabel("Day of Week"); ax.set_ylabel("Delay (minutes)")
    ax.set_title("Arrival Delay by Day of Week")
    ax.axhline(0, color="gray", lw=0.8, linestyle="--")
    _savefig("delay_by_day.png")


def figure_delay_by_hour(df):
    mean_by_hour = df.groupby("planned_arrival_hour")["delay_minutes"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(mean_by_hour["planned_arrival_hour"], mean_by_hour["delay_minutes"],
           color=_BAR_COLOUR, width=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Planned Arrival Hour (24h)")
    ax.set_ylabel("Mean Delay (minutes)")
    ax.set_title("Mean Arrival Delay by Hour of Day")
    ax.set_xticks(range(0, 24))
    _savefig("delay_by_hour.png")


def figure_delay_by_station(df, encoder):
    df = df.copy()
    df["station"] = encoder.inverse_transform(df["location_encoded"].astype(int))
    mean_by_st = df.groupby("station")["delay_minutes"].mean().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.barh(mean_by_st.index[::-1], mean_by_st.values[::-1], color=_BAR_COLOUR)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean Delay (minutes)")
    ax.set_title("Mean Arrival Delay by Station (Top 20)")
    _savefig("delay_by_station.png")


def figure_model_comparison(results: dict, metric: str, filename: str, title: str, lower_better: bool):
    names  = list(results.keys())
    values = [results[n][metric] for n in names]
    best_idx = (int(np.argmin(values)) if lower_better else int(np.argmax(values)))
    colours = [_BEST_COLOUR if i == best_idx else _BAR_COLOUR for i in range(len(names))]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, values, color=colours, width=0.6)
    fmt = ".2f" if metric != "R2" else ".3f"
    offset = max(values) * 0.01
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                format(val, fmt), ha="center", va="bottom", fontsize=10)
    if metric == "R2":
        ax.set_ylabel("R-squared")
        ax.set_ylim(min(values) - 0.05, 1.0)
    else:
        ax.set_ylabel(f"{metric} (minutes)")
        ax.set_ylim(0, max(values) * 1.2)
    ax.set_title(title)
    plt.xticks(rotation=15, ha="right")
    _savefig(filename)


def figure_actual_vs_predicted(y_true, y_pred, n=2000):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), size=min(n, len(y_true)), replace=False)
    y_t = np.array(y_true)[idx]
    y_p = np.array(y_pred)[idx]
    lims = [min(y_t.min(), y_p.min()) - 2, max(y_t.max(), y_p.max()) + 2]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_t, y_p, alpha=0.25, s=10, color=_BAR_COLOUR)
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Delay (minutes)")
    ax.set_ylabel("Predicted Delay (minutes)")
    ax.set_title(f"Actual vs Predicted (n={len(y_t):,}) — Random Forest")
    ax.legend()
    _savefig("actual_vs_predicted.png")


def figure_residuals(y_true, y_pred, n=2000):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), size=min(n, len(y_true)), replace=False)
    residuals = np.array(y_pred) - np.array(y_true)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(np.array(y_pred)[idx], residuals[idx], alpha=0.25, s=10, color=_BAR_COLOUR)
    ax.axhline(0, color="red", lw=1.5, linestyle="--")
    ax.set_xlabel("Predicted Delay (minutes)")
    ax.set_ylabel("Residual (Predicted - Actual)")
    ax.set_title("Residual Plot — Random Forest")
    _savefig("residuals.png")


def figure_feature_importance(rf_model, feature_names):
    importances = rf_model.feature_importances_
    order = np.argsort(importances)[::-1]
    names = [feature_names[i] for i in order]
    vals  = importances[order]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, vals, color=_BAR_COLOUR)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance (Mean Decrease Impurity)")
    ax.set_title("Random Forest — Feature Importances")
    plt.xticks(rotation=20, ha="right")
    _savefig("feature_importance.png")


# ═══════════════════════════════════════════════════════════════════════════
#  3.  Detailed statistics for the written report
# ═══════════════════════════════════════════════════════════════════════════

def build_detailed_stats(df, results: dict, encoder, travel_times: dict) -> dict:
    """
    Compute rich descriptive statistics that can be quoted directly in
    the written report. Saved as detailed_statistics.json.
    """
    stats = {}

    # ── Dataset overview ────────────────────────────────────────────────
    stats["dataset"] = {
        "total_rows_cleaned":   int(len(df)),
        "rows_by_year": {
            int(y): int((pd.to_datetime(df["date_of_service"]).dt.year == y).sum())
            for y in sorted(pd.to_datetime(df["date_of_service"]).dt.year.unique())
        },
        "rows_by_direction": {
            "WEY2WAT (0)": int((df["direction"] == 0).sum()),
            "WAT2WEY (1)": int((df["direction"] == 1).sum()),
        },
        "unique_stations": int(df["location_encoded"].nunique()),
        "date_range": {
            "earliest": str(pd.to_datetime(df["date_of_service"]).min().date()),
            "latest":   str(pd.to_datetime(df["date_of_service"]).max().date()),
        },
    }

    # ── Delay descriptive stats ─────────────────────────────────────────
    d = df["delay_minutes"]
    stats["delay_summary_minutes"] = {
        "mean":   round(float(d.mean()), 3),
        "median": round(float(d.median()), 3),
        "std":    round(float(d.std()), 3),
        "min":    round(float(d.min()), 3),
        "max":    round(float(d.max()), 3),
        "q25":    round(float(d.quantile(0.25)), 3),
        "q75":    round(float(d.quantile(0.75)), 3),
        "q95":    round(float(d.quantile(0.95)), 3),
        "q99":    round(float(d.quantile(0.99)), 3),
        "pct_on_time_within_1min":   round(float((d.abs() <= 1).mean() * 100), 2),
        "pct_within_5min":           round(float((d.abs() <= 5).mean() * 100), 2),
        "pct_late_over_10min":       round(float((d > 10).mean() * 100), 2),
        "pct_severely_late_over_30": round(float((d > 30).mean() * 100), 2),
    }

    # ── Peak vs off-peak comparison ─────────────────────────────────────
    peak_mask = df["planned_arrival_hour"].isin([7, 8, 9, 17, 18, 19])
    stats["peak_vs_offpeak"] = {
        "peak_hours_definition": "07:00-09:59 and 17:00-19:59",
        "peak_mean_delay":     round(float(d[peak_mask].mean()), 3),
        "off_peak_mean_delay": round(float(d[~peak_mask].mean()), 3),
        "peak_rows":     int(peak_mask.sum()),
        "off_peak_rows": int((~peak_mask).sum()),
    }

    # ── Weekday vs weekend ──────────────────────────────────────────────
    weekend_mask = df["day_of_week"].isin([5, 6])
    stats["weekday_vs_weekend"] = {
        "weekday_mean_delay": round(float(d[~weekend_mask].mean()), 3),
        "weekend_mean_delay": round(float(d[ weekend_mask].mean()), 3),
    }

    # ── Per-month mean delay ────────────────────────────────────────────
    stats["mean_delay_by_month"] = {
        _MONTH_NAMES[m - 1]: round(float(df.loc[df["month"] == m, "delay_minutes"].mean()), 3)
        for m in range(1, 13)
        if (df["month"] == m).any()
    }

    # ── Per-day-of-week mean delay ──────────────────────────────────────
    stats["mean_delay_by_day"] = {
        _DAY_NAMES[d_]: round(float(df.loc[df["day_of_week"] == d_, "delay_minutes"].mean()), 3)
        for d_ in range(7) if (df["day_of_week"] == d_).any()
    }

    # ── Top 10 worst stations (highest mean delay) ──────────────────────
    df_st = df.copy()
    df_st["station"] = encoder.inverse_transform(df_st["location_encoded"].astype(int))
    by_st = df_st.groupby("station")["delay_minutes"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    stats["top10_worst_stations"] = [
        {"station_crs": str(idx), "mean_delay": round(float(row["mean"]), 3),
         "sample_rows": int(row["count"])}
        for idx, row in by_st.head(10).iterrows()
    ]
    stats["top10_best_stations"] = [
        {"station_crs": str(idx), "mean_delay": round(float(row["mean"]), 3),
         "sample_rows": int(row["count"])}
        for idx, row in by_st.tail(10).iloc[::-1].iterrows()
    ]

    # ── Model ranking ───────────────────────────────────────────────────
    ranked = sorted(results.items(), key=lambda kv: kv[1]["MAE"])
    stats["model_rankings"] = [
        {"rank": i + 1, "name": name, "MAE": r["MAE"], "RMSE": r["RMSE"], "R2": r["R2"]}
        for i, (name, r) in enumerate(ranked)
    ]
    stats["best_model"] = ranked[0][0]

    # ── Travel time lookup summary ──────────────────────────────────────
    if travel_times:
        all_vals = list(travel_times.values())
        stats["station_travel_times"] = {
            "pairs_computed":     len(travel_times),
            "mean_minutes":       round(float(np.mean(all_vals)), 2),
            "median_minutes":     round(float(np.median(all_vals)), 2),
            "min_minutes":        round(float(np.min(all_vals)), 2),
            "max_minutes":        round(float(np.max(all_vals)), 2),
            "note": (
                "Computed as the mean PLANNED travel time between every "
                "consecutive station pair observed in the data, per direction. "
                "Used by conversation_live.py to derive a scheduled arrival "
                "time from actual timetable data instead of a fixed guess."
            ),
        }

    return stats


# ═══════════════════════════════════════════════════════════════════════════
#  4.  Main driver
# ═══════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Loading data via preprocess_v2…")
    X_train, X_test, y_train, y_test, le, scaler, df_clean = load_and_preprocess_v2()
    feature_names = list(X_train.columns)

    # ── (1) Travel-time lookup ─────────────────────────────────────────
    travel_times = build_station_travel_times()
    joblib.dump(travel_times, _TRAVEL_TIMES)
    logger.info("Saved travel-time lookup → %s", _TRAVEL_TIMES)

    # ── (2) Reload models and re-evaluate ──────────────────────────────
    logger.info("Re-evaluating saved models on test set…")
    X_test_scaled = scaler.transform(X_test)
    models_cfg = {
        "kNN":               ("knn_v2.pkl",                True),
        "Random Forest":     ("random_forest_v2.pkl",      False),
        "Linear Regression": ("linear_regression_v2.pkl",  False),
        "Neural Network (MLP)": ("mlp_v2.pkl",             True),
        "Gradient Boosting": ("gradient_boosting_v2.pkl",  False),
    }
    results = {}
    rf_model = None
    rf_predictions = None
    for name, (fname, uses_scaled) in models_cfg.items():
        model = joblib.load(os.path.join(_MODELS_DIR, fname))
        X_eval = X_test_scaled if uses_scaled else X_test
        y_pred = model.predict(X_eval)
        results[name] = {
            "MAE":  round(float(mean_absolute_error(y_test, y_pred)), 3),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3),
            "R2":   round(float(r2_score(y_test, y_pred)), 4),
        }
        if name == "Random Forest":
            rf_model = model
            rf_predictions = y_pred
        logger.info("  %s  MAE=%.3f", name, results[name]["MAE"])

    # Persist updated results JSON
    with open(os.path.join(_ROOT, "reports", "model_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── (3) Regenerate all figures ─────────────────────────────────────
    logger.info("Regenerating all figures…")
    figure_delay_distribution(df_clean)
    figure_delay_by_month(df_clean)
    figure_delay_by_day(df_clean)
    figure_delay_by_hour(df_clean)
    figure_delay_by_station(df_clean, le)
    figure_model_comparison(results, "MAE",  "model_comparison_mae.png",
                            "Model Comparison — MAE (lower is better)", lower_better=True)
    figure_model_comparison(results, "R2",   "model_comparison_r2.png",
                            "Model Comparison — R-squared (higher is better)", lower_better=False)
    figure_actual_vs_predicted(y_test, rf_predictions)
    figure_residuals(y_test, rf_predictions)
    figure_feature_importance(rf_model, feature_names)

    # ── (4) Detailed stats ─────────────────────────────────────────────
    logger.info("Computing detailed statistics for the report…")
    stats = build_detailed_stats(df_clean, results, le, travel_times)
    with open(_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved detailed stats → %s", _STATS_PATH)

    # ── Print headline summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ARTEFACT BUILD COMPLETE")
    print("=" * 70)
    print(f"Figures regenerated:        {len(os.listdir(_FIGURES_DIR))} files")
    print(f"Travel-time pairs:          {len(travel_times)}")
    print(f"Mean delay (whole dataset): {stats['delay_summary_minutes']['mean']} min")
    print(f"% on time within 1 min:     {stats['delay_summary_minutes']['pct_on_time_within_1min']}%")
    print(f"Best model:                 {stats['best_model']}")
    print()


if __name__ == "__main__":
    main()
