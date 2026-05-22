"""
predict.py
----------
Load the best saved model and make arrival-time predictions for a
delayed train on the Weymouth → London Waterloo route.

Usage (from chatbot)
--------------------
    from task2.predict import predict_arrival_time

    result = predict_arrival_time(
        current_station="SOU",
        destination_station="WAT",
        current_delay_minutes=10,
        planned_arrival_time_str="14:35",
        day_of_week=2,          # Wednesday
        month=4,                # April
    )
    print(result)
    # {'predicted_delay_min': 8.2,
    #  'predicted_arrival': '14:43',
    #  'planned_arrival': '14:35',
    #  'destination': 'WAT'}
"""

import os
import logging
import numpy as np
import joblib

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_model = None
_meta = None
_scaler = None
_encoder = None


def _load_artefacts():
    """Load the best model, its metadata, the scaler, and the label encoder."""
    global _model, _meta, _scaler, _encoder

    if _model is not None:
        return  # already loaded

    best_path = os.path.join(_MODELS_DIR, "best_model.pkl")
    meta_path = os.path.join(_MODELS_DIR, "best_model_meta.pkl")
    scaler_path = os.path.join(_MODELS_DIR, "scaler.pkl")
    encoder_path = os.path.join(_MODELS_DIR, "location_encoder.pkl")

    if not os.path.exists(best_path):
        raise FileNotFoundError(
            f"Best model not found at {best_path}. "
            "Run  python -m task2.train_models  first."
        )

    _model = joblib.load(best_path)
    _meta = joblib.load(meta_path) if os.path.exists(meta_path) else {}
    _scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    _encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None

    logger.info(
        "Loaded model '%s' (uses_scaler=%s)",
        _meta.get("model_name", "unknown"),
        _meta.get("uses_scaler", False),
    )


# ---------------------------------------------------------------------------
# Public prediction function
# ---------------------------------------------------------------------------
def predict_arrival_time(
    current_station: str,
    destination_station: str,
    current_delay_minutes: float,
    planned_arrival_time_str: str,
    day_of_week: int,
    month: int,
) -> dict:
    """
    Predict the actual arrival time at the destination station given
    the current delay.

    Parameters
    ----------
    current_station          : CRS code of station where the train currently is
    destination_station      : CRS code of the passenger's destination
    current_delay_minutes    : reported delay in minutes at the current station
    planned_arrival_time_str : scheduled arrival at destination, e.g. "14:35"
    day_of_week              : 0=Monday … 6=Sunday
    month                    : 1-12

    Returns
    -------
    dict with:
        predicted_delay_min  : float — predicted delay at destination (minutes)
        predicted_arrival    : str   — predicted arrival time e.g. "14:43"
        planned_arrival      : str   — the original planned time
        destination          : str   — destination CRS code
        model_name           : str   — which model was used
    """
    _load_artefacts()

    # Parse planned arrival time into hour and total minutes
    parts = planned_arrival_time_str.strip().split(":")
    planned_hour = int(parts[0])
    planned_min = int(parts[1]) if len(parts) > 1 else 0
    planned_total_min = planned_hour * 60 + planned_min

    # Encode the destination station (the location feature represents where
    # we are predicting arrival delay — i.e. the destination stop row).
    try:
        loc_encoded = _encoder.transform([destination_station.upper()])[0]
    except (ValueError, KeyError):
        # Station not in the training set — fall back to a simple heuristic:
        # predict that the delay at destination ≈ the current delay.
        logger.warning(
            "Station '%s' not in encoder — using fallback heuristic.",
            destination_station,
        )
        predicted_total = planned_total_min + current_delay_minutes
        h, m = divmod(int(round(predicted_total)) % 1440, 60)
        return {
            "predicted_delay_min": round(current_delay_minutes, 1),
            "predicted_arrival": f"{h:02d}:{m:02d}",
            "planned_arrival": planned_arrival_time_str,
            "destination": destination_station.upper(),
            "model_name": "fallback (station unknown)",
        }

    # Build feature vector — same order as preprocess.py feature_cols:
    # [planned_arrival_hour, planned_departure_hour, day_of_week,
    #  month, location_encoded, late_canc_reason, current_delay]
    #
    # We set planned_departure_hour ≈ planned_arrival_hour (reasonable
    # for intermediate stops) and late_canc_reason = 0 (unknown).
    features = np.array([[
        planned_hour,           # planned_arrival_hour
        planned_hour,           # planned_departure_hour (approx.)
        day_of_week,            # day_of_week
        month,                  # month
        loc_encoded,            # location_encoded
        0,                      # late_canc_reason (not known at query time)
        current_delay_minutes,  # current_delay
    ]])

    # Scale if the best model requires it
    if _meta.get("uses_scaler", False) and _scaler is not None:
        features = _scaler.transform(features)

    predicted_delay = float(_model.predict(features)[0])

    # Calculate predicted arrival time
    predicted_total = planned_total_min + predicted_delay
    # Clamp to valid range and handle midnight wrap
    predicted_total = int(round(predicted_total)) % 1440
    h, m = divmod(predicted_total, 60)

    return {
        "predicted_delay_min": round(predicted_delay, 1),
        "predicted_arrival": f"{h:02d}:{m:02d}",
        "planned_arrival": planned_arrival_time_str,
        "destination": destination_station.upper(),
        "model_name": _meta.get("model_name", "unknown"),
    }


# ---------------------------------------------------------------------------
# CLI quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = predict_arrival_time(
        current_station="SOU",
        destination_station="WAT",
        current_delay_minutes=10,
        planned_arrival_time_str="14:35",
        day_of_week=2,
        month=4,
    )
    print(f"\nPrediction result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
