"""
predict_v2.py
-------------
Load the best saved v2 model and make arrival-time predictions for a
delayed train on the Weymouth ↔ London Waterloo route.

Unlike predict.py, this version:
  - Uses best_model_v2.pkl, scaler_v2.pkl, location_encoder_v2.pkl
  - Accepts a 'direction' parameter (0=WEY2WAT, 1=WAT2WEY)
  - Derives day_of_week and month from datetime.now() automatically

Usage (from chatbot)
--------------------
    from task2.predict_v2 import predict_arrival_time_v2

    result = predict_arrival_time_v2(
        current_station="SOU",
        destination_station="WAT",
        current_delay_minutes=10,
        planned_arrival_time_str="14:35",
        direction=1,   # 0=WEY2WAT, 1=WAT2WEY
    )
    print(result)
    # {
    #   'predicted_delay_min': 8.2,
    #   'predicted_arrival':   '14:43',
    #   'planned_arrival':     '14:35',
    #   'destination':         'WAT',
    #   'model_name':          'Gradient Boosting',
    #   'direction':            1,
    #   'day_of_week':          2,
    #   'month':                4,
    # }
"""

import os
import logging
from datetime import datetime
import numpy as np
import joblib

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_model   = None
_meta    = None
_scaler  = None
_encoder = None


def _load_artefacts():
    """Load the best v2 model, metadata, scaler, and label encoder."""
    global _model, _meta, _scaler, _encoder

    if _model is not None:
        return  # already loaded

    best_path    = os.path.join(_MODELS_DIR, "best_model_v2.pkl")
    meta_path    = os.path.join(_MODELS_DIR, "best_model_meta_v2.pkl")
    scaler_path  = os.path.join(_MODELS_DIR, "scaler_v2.pkl")
    encoder_path = os.path.join(_MODELS_DIR, "location_encoder_v2.pkl")

    if not os.path.exists(best_path):
        raise FileNotFoundError(
            f"Best v2 model not found at {best_path}. "
            "Run  python -m task2.train_models_v2  first."
        )

    _model   = joblib.load(best_path)
    _meta    = joblib.load(meta_path) if os.path.exists(meta_path) else {}
    _scaler  = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    _encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None

    logger.info(
        "Loaded v2 model '%s' (uses_scaler=%s)",
        _meta.get("model_name", "unknown"),
        _meta.get("uses_scaler", False),
    )


# ---------------------------------------------------------------------------
# Public prediction function
# ---------------------------------------------------------------------------
def predict_arrival_time_v2(
    current_station: str,
    destination_station: str,
    current_delay_minutes: float,
    planned_arrival_time_str: str,
    direction: int,
) -> dict:
    """
    Predict the actual arrival time at the destination station given
    the current delay, using today's date and time automatically.

    Parameters
    ----------
    current_station          : CRS code of station where the train currently is
    destination_station      : CRS code of the passenger's destination
    current_delay_minutes    : reported delay in minutes at the current station
    planned_arrival_time_str : scheduled arrival at destination, e.g. "14:35"
    direction                : 0 = WEY2WAT (Weymouth → London), 1 = WAT2WEY

    Returns
    -------
    dict with:
        predicted_delay_min  : float — predicted delay at destination (minutes)
        predicted_arrival    : str   — predicted arrival time e.g. "14:43"
        planned_arrival      : str   — the original planned time
        destination          : str   — destination CRS code
        model_name           : str   — which model was used
        direction            : int   — 0 or 1
        day_of_week          : int   — 0=Monday … 6=Sunday (from today)
        month                : int   — 1-12 (from today)
    """
    _load_artefacts()

    # Derive temporal context from now
    now = datetime.now()
    day_of_week = now.weekday()   # 0=Monday … 6=Sunday
    month       = now.month       # 1-12

    # Parse planned arrival time
    parts = planned_arrival_time_str.strip().split(":")
    planned_hour = int(parts[0])
    planned_min  = int(parts[1]) if len(parts) > 1 else 0
    planned_total_min = planned_hour * 60 + planned_min

    # Encode the destination station
    try:
        loc_encoded = _encoder.transform([destination_station.upper()])[0]
    except (ValueError, KeyError):
        logger.warning(
            "Station '%s' not in v2 encoder — using fallback heuristic.",
            destination_station,
        )
        predicted_total = planned_total_min + current_delay_minutes
        h, m = divmod(int(round(predicted_total)) % 1440, 60)
        return {
            "predicted_delay_min": round(current_delay_minutes, 1),
            "predicted_arrival":   f"{h:02d}:{m:02d}",
            "planned_arrival":     planned_arrival_time_str,
            "destination":         destination_station.upper(),
            "model_name":          "fallback (station unknown)",
            "direction":           direction,
            "day_of_week":         day_of_week,
            "month":               month,
        }

    # Build feature vector — same order as preprocess_v2.py feature_cols:
    # [planned_arrival_hour, planned_departure_hour, day_of_week,
    #  month, location_encoded, late_canc_reason, current_delay, direction]
    features = np.array([[
        planned_hour,           # planned_arrival_hour
        planned_hour,           # planned_departure_hour (approx.)
        day_of_week,            # day_of_week
        month,                  # month
        loc_encoded,            # location_encoded
        0,                      # late_canc_reason (unknown at query time)
        current_delay_minutes,  # current_delay
        direction,              # direction (NEW in v2)
    ]])

    # Scale if needed
    if _meta.get("uses_scaler", False) and _scaler is not None:
        features = _scaler.transform(features)

    predicted_delay = float(_model.predict(features)[0])

    # Calculate predicted arrival time
    predicted_total = int(round(planned_total_min + predicted_delay)) % 1440
    h, m = divmod(predicted_total, 60)

    return {
        "predicted_delay_min": round(predicted_delay, 1),
        "predicted_arrival":   f"{h:02d}:{m:02d}",
        "planned_arrival":     planned_arrival_time_str,
        "destination":         destination_station.upper(),
        "model_name":          _meta.get("model_name", "unknown"),
        "direction":           direction,
        "day_of_week":         day_of_week,
        "month":               month,
    }


# ---------------------------------------------------------------------------
# CLI quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = predict_arrival_time_v2(
        current_station="SOU",
        destination_station="WAT",
        current_delay_minutes=10,
        planned_arrival_time_str="14:35",
        direction=1,
    )
    print("\nPrediction result (v2):")
    for k, v in result.items():
        print(f"  {k}: {v}")
