"""
preprocess.py
-------------
Data loading, cleaning, and feature engineering for the Weymouth-to-Waterloo
train delay prediction task.

Dataset: data/2025_WEY2WAT.xlsx  (147,900 rows of South Western Railway
stop-level punctuality data.)

Pipeline
--------
1. Load the raw Excel file.
2. Drop rows where actual_arrival_time or planned_arrival_time is null.
3. Convert time columns (stored as datetime.time objects) into total-minutes
   since midnight so they can be used numerically.
4. Engineer the feature matrix X and target vector y.
5. Split 80 / 20 train / test with a fixed random seed.
6. Fit and save a LabelEncoder for the station CRS codes.
"""

import os
import logging
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "2025_WEY2WAT.xlsx")
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")


# ---------------------------------------------------------------------------
# Helper – convert a datetime.time object to total minutes since midnight
# ---------------------------------------------------------------------------
def _time_to_minutes(t) -> float:
    """
    Convert a datetime.time object to fractional minutes since midnight.
    Returns NaN for non-time values so pandas can handle nulls naturally.
    """
    if isinstance(t, datetime.time):
        return t.hour * 60 + t.minute + t.second / 60.0
    return np.nan


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------
def load_and_preprocess(data_path: str = _DATA_PATH):
    """
    Load the raw dataset and return cleaned, feature-engineered DataFrames
    ready for model training.

    Returns
    -------
    X_train, X_test : pd.DataFrame   — feature matrices
    y_train, y_test : pd.Series      — target (delay in minutes at arrival)
    label_encoder   : LabelEncoder   — fitted encoder for station CRS codes
    df_clean        : pd.DataFrame   — the full cleaned DataFrame (for EDA)
    """
    logger.info("Loading dataset from %s …", data_path)
    df = pd.read_excel(data_path)
    logger.info("Raw dataset shape: %s", df.shape)

    # ------------------------------------------------------------------
    # 1. Drop rows missing planned or actual arrival times
    # ------------------------------------------------------------------
    df = df.dropna(subset=["planned_arrival_time", "actual_arrival_time"])
    logger.info("After dropping null arrivals: %s rows", len(df))

    # ------------------------------------------------------------------
    # 2. Convert time columns to minutes since midnight
    # ------------------------------------------------------------------
    time_cols = [
        "planned_arrival_time",
        "planned_departure_time",
        "actual_arrival_time",
        "actual_departure_time",
    ]
    for col in time_cols:
        df[f"{col}_min"] = df[col].apply(_time_to_minutes)

    # ------------------------------------------------------------------
    # 3. Target variable: delay at arrival (minutes)
    #    delay = actual_arrival - planned_arrival
    #    A positive value means the train arrived late.
    # ------------------------------------------------------------------
    df["delay_minutes"] = (
        df["actual_arrival_time_min"] - df["planned_arrival_time_min"]
    )

    # Handle midnight crossover (e.g. planned 23:55, actual 00:05)
    df.loc[df["delay_minutes"] < -720, "delay_minutes"] += 1440
    df.loc[df["delay_minutes"] > 720, "delay_minutes"] -= 1440

    # ------------------------------------------------------------------
    # 4. Feature engineering
    # ------------------------------------------------------------------

    # Hour of planned arrival (integer 0-23)
    df["planned_arrival_hour"] = df["planned_arrival_time_min"].apply(
        lambda m: int(m // 60) if pd.notna(m) else 0
    )

    # Hour of planned departure (integer 0-23)
    df["planned_departure_hour"] = df["planned_departure_time_min"].apply(
        lambda m: int(m // 60) if pd.notna(m) else 0
    )

    # Day of week (0=Monday, 6=Sunday)
    df["day_of_week"] = pd.to_datetime(df["date_of_service"]).dt.dayofweek

    # Month (1-12)
    df["month"] = pd.to_datetime(df["date_of_service"]).dt.month

    # Encode station CRS codes as integers
    le = LabelEncoder()
    df["location_encoded"] = le.fit_transform(df["location"])

    # Late/cancellation reason code (fill NaN with 0 = "no recorded cause")
    df["late_canc_reason"] = df["late_canc_reason"].fillna(0).astype(int)

    # Current delay: difference between actual departure and planned departure
    # at this stop — the key real-time signal the passenger would report.
    df["current_delay"] = (
        df["actual_departure_time_min"] - df["planned_departure_time_min"]
    )
    # Handle midnight crossover
    df.loc[df["current_delay"] < -720, "current_delay"] += 1440
    df.loc[df["current_delay"] > 720, "current_delay"] -= 1440
    # Fill NaN (e.g. first/last stops missing departure) with 0
    df["current_delay"] = df["current_delay"].fillna(0)

    # ------------------------------------------------------------------
    # 5. Assemble feature matrix X and target y
    # ------------------------------------------------------------------
    feature_cols = [
        "planned_arrival_hour",
        "planned_departure_hour",
        "day_of_week",
        "month",
        "location_encoded",
        "late_canc_reason",
        "current_delay",
    ]

    # Drop any remaining rows with NaN in the features or target
    df_clean = df.dropna(subset=feature_cols + ["delay_minutes"]).copy()
    logger.info("Final clean dataset: %s rows", len(df_clean))

    X = df_clean[feature_cols]
    y = df_clean["delay_minutes"]

    # ------------------------------------------------------------------
    # 6. Train / test split (80 / 20, reproducible)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

    # ------------------------------------------------------------------
    # 7. Save the label encoder for re-use at prediction time
    # ------------------------------------------------------------------
    os.makedirs(_MODELS_DIR, exist_ok=True)
    encoder_path = os.path.join(_MODELS_DIR, "location_encoder.pkl")
    joblib.dump(le, encoder_path)
    logger.info("Saved location encoder → %s", encoder_path)

    return X_train, X_test, y_train, y_test, le, df_clean


# ---------------------------------------------------------------------------
# CLI entry point (useful for quick testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    X_train, X_test, y_train, y_test, le, df = load_and_preprocess()
    print(f"\nFeatures:      {list(X_train.columns)}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")
    print(f"Target stats:  mean={y_train.mean():.2f}  std={y_train.std():.2f}")
    print(f"Stations:      {list(le.classes_)}")
