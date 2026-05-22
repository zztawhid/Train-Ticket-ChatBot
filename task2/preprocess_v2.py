"""
preprocess_v2.py
----------------
Data loading, cleaning, and feature engineering for the Weymouth-Waterloo
train delay prediction task — VERSION 2.

This version loads ALL 8 yearly/directional Excel files (2022-2025, both
directions) and adds a `direction` feature derived from the filename.

Dataset files (relative to project root):
  data/2022_WAT2WEY.xlsx   data/2022_WEY2WAT.xlsx
  data/2023_WAT2WEY.xlsx   data/2023_WEY2WAT.xlsx
  data/2024_WAT2WEY.xlsx   data/2024_WEY2WAT.xlsx
  data/2025_WAT2WEY.xlsx   data/2025_WEY2WAT.xlsx

Direction encoding:
  0 = WEY2WAT  (Weymouth → London Waterloo)
  1 = WAT2WEY  (London Waterloo → Weymouth)

Saved artefacts
---------------
  models/location_encoder_v2.pkl  — LabelEncoder for station CRS codes
  models/scaler_v2.pkl            — StandardScaler (fitted on training set)
"""

import os
import logging
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

# All 8 data files with their direction label
_DATA_FILES = [
    ("2022_WEY2WAT.xlsx", 0),
    ("2022_WAT2WEY.xlsx", 1),
    ("2023_WEY2WAT.xlsx", 0),
    ("2023_WAT2WEY.xlsx", 1),
    ("2024_WEY2WAT.xlsx", 0),
    ("2024_WAT2WEY.xlsx", 1),
    ("2025_WEY2WAT.xlsx", 0),
    ("2025_WAT2WEY.xlsx", 1),
]


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
def load_and_preprocess_v2():
    """
    Load all 8 raw datasets, concatenate, clean, and feature-engineer them.

    Returns
    -------
    X_train, X_test : pd.DataFrame   — feature matrices
    y_train, y_test : pd.Series      — target (delay in minutes at arrival)
    label_encoder   : LabelEncoder   — fitted encoder for station CRS codes
    scaler          : StandardScaler — fitted scaler
    df_clean        : pd.DataFrame   — the full cleaned DataFrame (for EDA)
    """
    frames = []
    for fname, direction in _DATA_FILES:
        fpath = os.path.join(_DATA_DIR, fname)
        if not os.path.exists(fpath):
            logger.warning("File not found, skipping: %s", fpath)
            continue
        logger.info("Loading %s …", fname)
        df_part = pd.read_excel(fpath)
        df_part["direction"] = direction
        frames.append(df_part)

    if not frames:
        raise FileNotFoundError(
            "No data files found. Expected files in: " + _DATA_DIR
        )

    df = pd.concat(frames, ignore_index=True)
    logger.info("Combined raw dataset shape: %s", df.shape)

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
    # ------------------------------------------------------------------
    df["delay_minutes"] = (
        df["actual_arrival_time_min"] - df["planned_arrival_time_min"]
    )
    # Handle midnight crossover
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

    # Late/cancellation reason code (fill NaN with 0)
    df["late_canc_reason"] = df["late_canc_reason"].fillna(0).astype(int)

    # Current delay: difference between actual departure and planned departure
    df["current_delay"] = (
        df["actual_departure_time_min"] - df["planned_departure_time_min"]
    )
    # Handle midnight crossover
    df.loc[df["current_delay"] < -720, "current_delay"] += 1440
    df.loc[df["current_delay"] > 720, "current_delay"] -= 1440
    df["current_delay"] = df["current_delay"].fillna(0)

    # ------------------------------------------------------------------
    # 5. Assemble feature matrix X and target y
    #    NOTE: direction is included as an additional feature vs v1
    # ------------------------------------------------------------------
    feature_cols = [
        "planned_arrival_hour",
        "planned_departure_hour",
        "day_of_week",
        "month",
        "location_encoded",
        "late_canc_reason",
        "current_delay",
        "direction",          # NEW in v2
    ]

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
    # 7. Fit and save scaler
    # ------------------------------------------------------------------
    os.makedirs(_MODELS_DIR, exist_ok=True)

    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler_path = os.path.join(_MODELS_DIR, "scaler_v2.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info("Saved scaler → %s", scaler_path)

    # ------------------------------------------------------------------
    # 8. Save the label encoder
    # ------------------------------------------------------------------
    encoder_path = os.path.join(_MODELS_DIR, "location_encoder_v2.pkl")
    joblib.dump(le, encoder_path)
    logger.info("Saved location encoder → %s", encoder_path)

    return X_train, X_test, y_train, y_test, le, scaler, df_clean


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    X_train, X_test, y_train, y_test, le, scaler, df = load_and_preprocess_v2()
    print(f"\nFeatures:      {list(X_train.columns)}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")
    print(f"Target stats:  mean={y_train.mean():.2f}  std={y_train.std():.2f}")
    print(f"Stations:      {list(le.classes_)}")
    print(f"Directions:    0=WEY2WAT  1=WAT2WEY")
