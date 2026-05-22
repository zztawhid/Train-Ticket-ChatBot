"""
train_models_v2.py
------------------
Train and evaluate five regression models on the COMBINED 2022-2025
Weymouth-Waterloo dataset (both directions).

Models
------
1. k-Nearest Neighbours (kNN) — best k from {3, 5, 7}
2. Random Forest
3. Linear Regression
4. Neural Network (MLPRegressor)
5. Gradient Boosting

Saved model artefacts (in models/)
-----------------------------------
  knn_v2.pkl
  random_forest_v2.pkl
  linear_regression_v2.pkl
  mlp_v2.pkl
  gradient_boosting_v2.pkl
  best_model_v2.pkl
  best_model_meta_v2.pkl

Saved visualisation plots (in reports/figures/)
------------------------------------------------
  delay_distribution.png
  delay_by_month.png
  delay_by_day.png
  delay_by_hour.png
  delay_by_station.png
  model_comparison_mae.png
  model_comparison_r2.png
  actual_vs_predicted.png
  residuals.png
  feature_importance.png

Run with:
    python -m task2.train_models_v2
"""

import os
import logging
import numpy as np
import joblib

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from task2.preprocess_v2 import load_and_preprocess_v2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
_FIGURES_DIR = os.path.join(_PROJECT_ROOT, "reports", "figures")

_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test) -> dict:
    """Compute MAE, RMSE, and R² for a fitted model on the test set."""
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
# Single neutral colour used throughout — simple and consistent
_BAR_COLOUR  = "steelblue"
_BEST_COLOUR = "darkorange"   # used only to highlight the best model bar

def _savefig(name: str):
    """Save current figure to reports/figures/ at 150 dpi and close it."""
    os.makedirs(_FIGURES_DIR, exist_ok=True)
    path = os.path.join(_FIGURES_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure → %s", path)


def plot_delay_distribution(df_clean):
    """Histogram of delay_minutes with mean and median lines."""
    delays = df_clean["delay_minutes"].clip(-20, 60)
    mean_val   = df_clean["delay_minutes"].mean()
    median_val = df_clean["delay_minutes"].median()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(delays, bins=80, color=_BAR_COLOUR)
    ax.axvline(mean_val,   color="red",   lw=1.5, linestyle="--",
               label=f"Mean: {mean_val:.1f} min")
    ax.axvline(median_val, color="black", lw=1.5, linestyle="-.",
               label=f"Median: {median_val:.1f} min")
    ax.set_xlabel("Delay (minutes)")
    ax.set_ylabel("Number of stops")
    ax.set_title("Distribution of Arrival Delays (2022–2025)")
    ax.legend()
    _savefig("delay_distribution.png")


def plot_delay_by_month(df_clean):
    """Boxplot of delay by month."""
    data_by_month = [
        df_clean.loc[df_clean["month"] == m, "delay_minutes"].clip(-10, 40).values
        for m in range(1, 13)
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(data_by_month, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor(_BAR_COLOUR)
        patch.set_alpha(0.7)
    ax.set_xticklabels(_MONTH_NAMES)
    ax.set_xlabel("Month")
    ax.set_ylabel("Delay (minutes)")
    ax.set_title("Arrival Delay by Month")
    ax.axhline(0, color="gray", lw=0.8, linestyle="--")
    _savefig("delay_by_month.png")


def plot_delay_by_day(df_clean):
    """Boxplot of delay by day of week."""
    data_by_day = [
        df_clean.loc[df_clean["day_of_week"] == d, "delay_minutes"].clip(-10, 40).values
        for d in range(7)
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data_by_day, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor(_BAR_COLOUR)
        patch.set_alpha(0.7)
    ax.set_xticklabels(_DAY_NAMES)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Delay (minutes)")
    ax.set_title("Arrival Delay by Day of Week")
    ax.axhline(0, color="gray", lw=0.8, linestyle="--")
    _savefig("delay_by_day.png")


def plot_delay_by_hour(df_clean):
    """Bar chart of mean delay by planned_arrival_hour."""
    mean_by_hour = (
        df_clean.groupby("planned_arrival_hour")["delay_minutes"].mean().reset_index()
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(mean_by_hour["planned_arrival_hour"], mean_by_hour["delay_minutes"],
           color=_BAR_COLOUR, width=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Planned Arrival Hour (24 h)")
    ax.set_ylabel("Mean Delay (minutes)")
    ax.set_title("Mean Arrival Delay by Hour of Day")
    ax.set_xticks(range(0, 24))
    _savefig("delay_by_hour.png")


def plot_delay_by_station(df_clean, label_encoder):
    """Horizontal bar chart of mean delay per station."""
    df_tmp = df_clean.copy()
    df_tmp["station_name"] = label_encoder.inverse_transform(
        df_tmp["location_encoded"].astype(int)
    )
    mean_by_station = (
        df_tmp.groupby("station_name")["delay_minutes"]
        .mean()
        .sort_values(ascending=False)
        .head(20)
    )
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.barh(mean_by_station.index[::-1], mean_by_station.values[::-1],
            color=_BAR_COLOUR)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean Delay (minutes)")
    ax.set_title("Mean Arrival Delay by Station (Top 20)")
    _savefig("delay_by_station.png")


def plot_model_comparison_mae(results: dict):
    """Bar chart comparing MAE of all models."""
    names    = list(results.keys())
    maes     = [results[n]["MAE"] for n in names]
    best_idx = int(np.argmin(maes))

    colours = [_BEST_COLOUR if i == best_idx else _BAR_COLOUR for i in range(len(names))]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, maes, color=colours, width=0.6)
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("MAE (minutes)")
    ax.set_title("Model Comparison — Mean Absolute Error (lower is better)")
    ax.set_ylim(0, max(maes) * 1.2)
    plt.xticks(rotation=15, ha="right")
    _savefig("model_comparison_mae.png")


def plot_model_comparison_r2(results: dict):
    """Bar chart comparing R² of all models."""
    names    = list(results.keys())
    r2s      = [results[n]["R2"] for n in names]
    best_idx = int(np.argmax(r2s))

    colours = [_BEST_COLOUR if i == best_idx else _BAR_COLOUR for i in range(len(names))]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, r2s, color=colours, width=0.6)
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("R²")
    ax.set_title("Model Comparison — R² Score (higher is better)")
    ax.set_ylim(min(r2s) - 0.05, 1.0)
    plt.xticks(rotation=15, ha="right")
    _savefig("model_comparison_r2.png")


def plot_actual_vs_predicted(model, X_test, y_test, uses_scaled: bool, scaler, n_samples=2000):
    """Scatter of actual vs predicted delay for the best model (sampled)."""
    X_eval = scaler.transform(X_test) if uses_scaled else X_test
    y_pred = model.predict(X_eval)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_test), size=min(n_samples, len(y_test)), replace=False)
    y_actual_s = np.array(y_test)[idx]
    y_pred_s   = y_pred[idx]

    lims = [min(y_actual_s.min(), y_pred_s.min()) - 2,
            max(y_actual_s.max(), y_pred_s.max()) + 2]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_actual_s, y_pred_s, alpha=0.25, s=10, color=_BAR_COLOUR)
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Delay (minutes)")
    ax.set_ylabel("Predicted Delay (minutes)")
    ax.set_title(f"Actual vs Predicted Delay — Best Model (n={len(y_actual_s):,})")
    ax.legend()
    _savefig("actual_vs_predicted.png")


def plot_residuals(model, X_test, y_test, uses_scaled: bool, scaler, n_samples=2000):
    """Residual plot (predicted − actual) for the best model."""
    X_eval = scaler.transform(X_test) if uses_scaled else X_test
    y_pred = model.predict(X_eval)
    residuals = y_pred - np.array(y_test)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_test), size=min(n_samples, len(y_test)), replace=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(y_pred[idx], residuals[idx], alpha=0.25, s=10, color=_BAR_COLOUR)
    ax.axhline(0, color="red", lw=1.5, linestyle="--")
    ax.set_xlabel("Predicted Delay (minutes)")
    ax.set_ylabel("Residual (Predicted − Actual)")
    ax.set_title("Residual Plot — Best Model")
    _savefig("residuals.png")


def plot_feature_importance(rf_model, feature_names):
    """Feature importance from the Random Forest model."""
    importances  = rf_model.feature_importances_
    sorted_idx   = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_vals  = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(sorted_names, sorted_vals, color=_BAR_COLOUR)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance (Mean Decrease Impurity)")
    ax.set_title("Random Forest — Feature Importances")
    plt.xticks(rotation=20, ha="right")
    _savefig("feature_importance.png")


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def train_all_models_v2():
    """
    Train all five model types on the combined 2022-2025 dataset.
    Save models, generate 10 visualisation plots.

    Returns
    -------
    results : dict[str, dict]  — model name → metrics dict
    """
    # 1. Load preprocessed data
    X_train, X_test, y_train, y_test, le, scaler, df_clean = load_and_preprocess_v2()

    os.makedirs(_MODELS_DIR, exist_ok=True)

    feature_names = list(X_train.columns)

    # Scaled arrays (for kNN and MLP)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # EDA plots (generated before model training)
    # ------------------------------------------------------------------
    logger.info("Generating EDA visualisations …")
    plot_delay_distribution(df_clean)
    plot_delay_by_month(df_clean)
    plot_delay_by_day(df_clean)
    plot_delay_by_hour(df_clean)
    plot_delay_by_station(df_clean, le)

    # ------------------------------------------------------------------
    # Define models
    # ------------------------------------------------------------------
    models = {}

    # (a) kNN
    logger.info("Training kNN (k=3, 5, 7) …")
    best_knn, best_knn_mae, best_k = None, float("inf"), 3
    for k in (3, 5, 7):
        knn = KNeighborsRegressor(n_neighbors=k, n_jobs=-1)
        knn.fit(X_train_scaled, y_train)
        mae = mean_absolute_error(y_test, knn.predict(X_test_scaled))
        logger.info("  k=%d  MAE=%.3f", k, mae)
        if mae < best_knn_mae:
            best_knn, best_knn_mae, best_k = knn, mae, k
    models[f"kNN (k={best_k})"] = ("knn_v2", best_knn, True)

    # (b) Random Forest
    logger.info("Training Random Forest …")
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = ("random_forest_v2", rf, False)

    # (c) Linear Regression
    logger.info("Training Linear Regression …")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["Linear Regression"] = ("linear_regression_v2", lr, False)

    # (d) MLP Neural Network
    logger.info("Training MLP Neural Network …")
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_train_scaled, y_train)
    models["Neural Network (MLP)"] = ("mlp_v2", mlp, True)

    # (e) Gradient Boosting
    logger.info("Training Gradient Boosting …")
    gb = GradientBoostingRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42
    )
    gb.fit(X_train, y_train)
    models["Gradient Boosting"] = ("gradient_boosting_v2", gb, False)

    # ------------------------------------------------------------------
    # Evaluate all models and save
    # ------------------------------------------------------------------
    results = {}
    for name, (filename, model, uses_scaled) in models.items():
        test_data = X_test_scaled if uses_scaled else X_test
        metrics = evaluate_model(model, test_data, y_test)
        results[name] = metrics

        model_path = os.path.join(_MODELS_DIR, f"{filename}.pkl")
        joblib.dump(model, model_path)
        logger.info(
            "Saved %s → %s  (MAE=%.3f, RMSE=%.3f, R²=%.4f)",
            name, model_path, metrics["MAE"], metrics["RMSE"], metrics["R2"],
        )

    # ------------------------------------------------------------------
    # Model comparison plots
    # ------------------------------------------------------------------
    logger.info("Generating model comparison plots …")
    plot_model_comparison_mae(results)
    plot_model_comparison_r2(results)

    # Feature importance from RF
    plot_feature_importance(rf, feature_names)

    # ------------------------------------------------------------------
    # Best model
    # ------------------------------------------------------------------
    best_name       = min(results, key=lambda n: results[n]["MAE"])
    best_filename   = models[best_name][0]
    best_model_obj  = models[best_name][1]
    best_uses_scaled = models[best_name][2]

    # Plots using best model
    logger.info("Generating best-model diagnostic plots …")
    plot_actual_vs_predicted(best_model_obj, X_test, y_test, best_uses_scaled, scaler)
    plot_residuals(best_model_obj, X_test, y_test, best_uses_scaled, scaler)

    # Save best model
    best_path = os.path.join(_MODELS_DIR, "best_model_v2.pkl")
    joblib.dump(best_model_obj, best_path)

    meta = {
        "model_name": best_name,
        "uses_scaler": best_uses_scaled,
        "metrics": results[best_name],
        "feature_names": feature_names,
    }
    joblib.dump(meta, os.path.join(_MODELS_DIR, "best_model_meta_v2.pkl"))

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON (v2) — Combined 2022-2025 Dataset")
    print("=" * 70)
    header = f"{'Model':<25} {'MAE':>9} {'RMSE':>9} {'R²':>9}"
    print(header)
    print("-" * 70)
    for name, m in results.items():
        marker = " *" if name == best_name else ""
        print(f"{name:<25} {m['MAE']:>9.3f} {m['RMSE']:>9.3f} {m['R2']:>9.4f}{marker}")
    print("=" * 70)
    print(f"\nBest model: {best_name}")
    print(f"  MAE  = {results[best_name]['MAE']:.3f} min")
    print(f"  RMSE = {results[best_name]['RMSE']:.3f} min")
    print(f"  R²   = {results[best_name]['R2']:.4f}")
    print(f"  Saved → {best_path}\n")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train_all_models_v2()
