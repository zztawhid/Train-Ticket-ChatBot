"""
evaluate.py
-----------
Load all saved models from the models/ folder and produce a comparison
report on the test set.  Can be run standalone to regenerate the comparison
table without re-training.
"""

import os
import logging
import numpy as np
import joblib
from tabulate import tabulate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from task2.preprocess import load_and_preprocess

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

# Mapping of display name → (filename, uses_scaler)
_MODEL_REGISTRY = {
    "kNN":                ("knn.pkl",                True),
    "Random Forest":      ("random_forest.pkl",      False),
    "Linear Regression":  ("linear_regression.pkl",  False),
    "Neural Network":     ("mlp.pkl",                True),
    "Gradient Boosting":  ("gradient_boosting.pkl",  False),
}


def evaluate_all() -> dict:
    """
    Load every saved model and evaluate it on the test set.

    Returns
    -------
    results : dict[str, dict]  — model name → {MAE, RMSE, R²}
    """
    # Prepare data
    _, X_test, _, y_test, _, _ = load_and_preprocess()

    # Load scaler (needed by kNN and MLP)
    scaler_path = os.path.join(_MODELS_DIR, "scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    X_test_scaled = scaler.transform(X_test) if scaler is not None else X_test

    results = {}
    for name, (filename, uses_scaler) in _MODEL_REGISTRY.items():
        model_path = os.path.join(_MODELS_DIR, filename)
        if not os.path.exists(model_path):
            logger.warning("Model file not found: %s — skipping", model_path)
            continue

        model = joblib.load(model_path)
        data = X_test_scaled if uses_scaler else X_test
        y_pred = model.predict(data)

        results[name] = {
            "MAE":  mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R²":   r2_score(y_test, y_pred),
        }

    return results


def print_comparison(results: dict) -> None:
    """Pretty-print the results as a grid table."""
    rows = []
    for name, m in results.items():
        rows.append([name, f"{m['MAE']:.3f}", f"{m['RMSE']:.3f}", f"{m['R²']:.4f}"])

    print("\n" + "=" * 65)
    print("  MODEL COMPARISON — Test Set Evaluation")
    print("=" * 65)
    print(tabulate(rows, headers=["Model", "MAE (min)", "RMSE (min)", "R²"], tablefmt="grid"))

    best = min(results, key=lambda n: results[n]["MAE"])
    print(f"\n✅ Best model (lowest MAE): {best}  —  MAE = {results[best]['MAE']:.3f} min\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = evaluate_all()
    print_comparison(results)
