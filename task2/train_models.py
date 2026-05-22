"""
train_models.py
---------------
Train and evaluate five regression models for predicting train delay at
the destination station.

Models
------
1. k-Nearest Neighbours (kNN) — best k from {3, 5, 7}
2. Random Forest
3. Linear Regression
4. Neural Network (MLPRegressor)
5. Gradient Boosting

Each model is evaluated on the hold-out test set with:
  • MAE  (Mean Absolute Error, in minutes)
  • RMSE (Root Mean Squared Error)
  • R²   (Coefficient of Determination)

All trained models are saved as .pkl files in the models/ folder.
The overall best model is also saved as models/best_model.pkl.
"""

import os
import logging
import numpy as np
import joblib
from tabulate import tabulate

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from task2.preprocess import load_and_preprocess

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test) -> dict:
    """
    Compute MAE, RMSE, and R² for a fitted model on the test set.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R²": r2}


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def train_all_models():
    """
    Train all five model types, evaluate them, save .pkl files,
    and print a comparison table.

    Returns
    -------
    results : dict[str, dict]  — model name → metrics dict
    """
    # 1. Load preprocessed data
    X_train, X_test, y_train, y_test, le, _ = load_and_preprocess()

    os.makedirs(_MODELS_DIR, exist_ok=True)

    # Scale features for kNN and MLP (tree methods don't need it)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    scaler_path = os.path.join(_MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info("Saved scaler → %s", scaler_path)

    # ------------------------------------------------------------------
    # Define models
    # ------------------------------------------------------------------
    models = {}

    # (a) kNN — try k = 3, 5, 7 and keep the best
    logger.info("Training kNN (k=3, 5, 7) …")
    best_knn, best_knn_mae = None, float("inf")
    best_k = 3
    for k in (3, 5, 7):
        knn = KNeighborsRegressor(n_neighbors=k, n_jobs=-1)
        knn.fit(X_train_scaled, y_train)
        mae = mean_absolute_error(y_test, knn.predict(X_test_scaled))
        logger.info("  k=%d  MAE=%.3f", k, mae)
        if mae < best_knn_mae:
            best_knn, best_knn_mae, best_k = knn, mae, k
    models[f"kNN (k={best_k})"] = ("knn", best_knn, True)  # True = uses scaled data

    # (b) Random Forest
    logger.info("Training Random Forest …")
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = ("random_forest", rf, False)

    # (c) Linear Regression
    logger.info("Training Linear Regression …")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["Linear Regression"] = ("linear_regression", lr, False)

    # (d) Neural Network (MLP)
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
    models["Neural Network (MLP)"] = ("mlp", mlp, True)

    # (e) Gradient Boosting
    logger.info("Training Gradient Boosting …")
    gb = GradientBoostingRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42
    )
    gb.fit(X_train, y_train)
    models["Gradient Boosting"] = ("gradient_boosting", gb, False)

    # ------------------------------------------------------------------
    # Evaluate all models
    # ------------------------------------------------------------------
    results = {}
    for name, (filename, model, uses_scaled) in models.items():
        test_data = X_test_scaled if uses_scaled else X_test
        metrics = evaluate_model(model, test_data, y_test)
        results[name] = metrics

        # Save model
        model_path = os.path.join(_MODELS_DIR, f"{filename}.pkl")
        joblib.dump(model, model_path)
        logger.info("Saved %s → %s", name, model_path)

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    table_rows = []
    for name, m in results.items():
        table_rows.append([name, f"{m['MAE']:.3f}", f"{m['RMSE']:.3f}", f"{m['R²']:.4f}"])

    print("\n" + "=" * 65)
    print("  MODEL COMPARISON — Test Set Evaluation")
    print("=" * 65)
    print(tabulate(
        table_rows,
        headers=["Model", "MAE (min)", "RMSE (min)", "R²"],
        tablefmt="grid",
    ))

    # ------------------------------------------------------------------
    # Save best model
    # ------------------------------------------------------------------
    best_name = min(results, key=lambda n: results[n]["MAE"])
    best_filename = models[best_name][0]
    best_model_obj = models[best_name][1]
    best_uses_scaled = models[best_name][2]

    best_path = os.path.join(_MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model_obj, best_path)

    # Save metadata so predict.py knows which model it loaded
    meta = {
        "model_name": best_name,
        "uses_scaler": best_uses_scaled,
        "metrics": results[best_name],
    }
    joblib.dump(meta, os.path.join(_MODELS_DIR, "best_model_meta.pkl"))

    print(f"\n✅ Best model: {best_name}")
    print(f"   MAE  = {results[best_name]['MAE']:.3f} min")
    print(f"   RMSE = {results[best_name]['RMSE']:.3f} min")
    print(f"   R²   = {results[best_name]['R²']:.4f}")
    print(f"   Saved → {best_path}\n")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        from tabulate import tabulate  # noqa: F811
    except ImportError:
        print("Installing tabulate …")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
    train_all_models()
