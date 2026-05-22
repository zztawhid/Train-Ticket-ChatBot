# Task 2 — Train Delay Prediction

## 1. Dataset Description

The dataset (`data/2025_WEY2WAT.xlsx`) contains **147,900 rows** of stop-level
punctuality data for South Western Railway services on the **Weymouth → London
Waterloo** route.

| Column                  | Type      | Description                                    |
|-------------------------|-----------|------------------------------------------------|
| `rid`                   | int       | Unique train service identifier                |
| `date_of_service`       | datetime  | Date the service ran                           |
| `toc_code`              | string    | Train Operating Company (always "SW")          |
| `location`              | string    | 3-letter CRS code of the station stop          |
| `planned_arrival_time`  | time      | Scheduled arrival at this stop                 |
| `planned_departure_time`| time      | Scheduled departure from this stop             |
| `actual_arrival_time`   | time      | Actual arrival at this stop                    |
| `actual_departure_time` | time      | Actual departure from this stop                |
| `late_canc_reason`      | float/int | Delay attribution code (NaN = no recorded cause)|

The route passes through **45 unique stations** including Southampton (SOU),
Bournemouth (BMH), Winchester (WIN), Basingstoke (BSK), Woking (WOK),
Clapham Junction (CLJ), and London Waterloo (WAT).

---

## 2. Data Preprocessing

Implemented in `task2/preprocess.py`.

### Cleaning steps
1. **Drop null arrivals** — rows where `planned_arrival_time` or
   `actual_arrival_time` is null are removed (reduces 147,900 → 136,439 rows).
   These correspond to origin stations where trains only depart.
2. **Time conversion** — `datetime.time` objects are converted to fractional
   minutes since midnight for numerical processing.
3. **Midnight crossover handling** — delays that cross midnight (e.g. planned
   23:55, actual 00:05) are corrected by adding/subtracting 1440 minutes.
4. **Missing values** — `late_canc_reason` nulls filled with 0 (no recorded
   delay cause); `current_delay` nulls filled with 0.

### Justification
- Dropping null arrivals is necessary because we need both planned and actual
  arrival times to compute the target variable.
- Converting times to minutes enables mathematical operations (subtraction,
  comparison) that are not possible with `datetime.time` objects.
- The midnight crossover fix prevents spurious negative delays of -1400+
  minutes for late-night services.

---

## 3. Features and Target

### Target variable (y)
**`delay_minutes`** = `actual_arrival_time` − `planned_arrival_time` (in minutes)

A positive value means the train arrived late; negative means early.

- **Mean delay**: ~2.07 minutes  
- **Std deviation**: ~8.08 minutes

### Feature matrix (X)

| Feature                  | Type    | Rationale                                                |
|--------------------------|---------|----------------------------------------------------------|
| `planned_arrival_hour`   | int     | Captures time-of-day patterns (peak vs off-peak)         |
| `planned_departure_hour` | int     | Departure timing affects downstream delays               |
| `day_of_week`            | int 0-6 | Weekend vs weekday services have different patterns       |
| `month`                  | int 1-12| Seasonal effects (weather, holiday demand)               |
| `location_encoded`       | int     | Label-encoded station CRS code — position on the route   |
| `late_canc_reason`       | int     | Delay cause code (0 = none) — indicates systemic issues  |
| `current_delay`          | float   | **Most important**: actual departure delay at this stop  |

`current_delay` is the key real-time signal — it represents the delay a
passenger would observe and report to the chatbot.

### Train/test split
- **80% train** (109,151 samples), **20% test** (27,288 samples)
- `random_state=42` for reproducibility

---

## 4. Models Trained

All models are implemented in `task2/train_models.py` using scikit-learn.

| # | Model                  | Description                                         |
|---|------------------------|-----------------------------------------------------|
| 1 | **kNN Regressor**      | Tested k=3, 5, 7; best k selected by lowest MAE    |
| 2 | **Random Forest**      | 200 trees, max depth 15                             |
| 3 | **Linear Regression**  | Ordinary least squares baseline                     |
| 4 | **Neural Network**     | MLPRegressor with layers (128, 64, 32), early stop  |
| 5 | **Gradient Boosting**  | 300 estimators, max depth 6, learning rate 0.1      |

Feature scaling (StandardScaler) is applied for kNN and MLP, which are
distance-sensitive. Tree-based models use raw features.

### Evaluation Metrics (Test Set)

| Model                | MAE (min) | RMSE (min) | R²     |
|----------------------|-----------|------------|--------|
| kNN (k=5)            | 1.339     | 3.532      | 0.8140 |
| Random Forest        | 1.029     | 3.180      | 0.8493 |
| Linear Regression    | 1.496     | 3.861      | 0.7777 |
| Neural Network (MLP) | 1.189     | 3.486      | 0.8188 |
| Gradient Boosting    | 1.029     | 3.119      | 0.8550 |

---

## 5. Best Model Selection

**Random Forest** and **Gradient Boosting** are tied on MAE (1.029 min).
The system selects the first model with the lowest MAE, which is
**Random Forest** (R² = 0.8493).

Both ensemble methods outperform the baseline Linear Regression by ~31% on
MAE. The tree-based models handle the non-linear relationships between delay
propagation, time of day, and station position well without requiring
feature scaling.

The best model is saved as `models/best_model.pkl` and used automatically
by the chatbot.

---

## 6. Chatbot Integration

### Architecture
```
app.py (Streamlit UI)
  ├── Task 1: chatbot/conversation.py  →  Ticket search (OJP API)
  └── Task 2: task2/conversation.py    →  Delay prediction (ML model)
```

### Conversation Flow (Task 2)
1. User selects "⏱️ Predict Arrival Time" in the sidebar
2. Bot asks: **Which station is your train currently at?**
3. Bot confirms the station (with alternatives if fuzzy match)
4. Bot asks: **Where is your destination?**
5. Bot confirms the destination
6. Bot asks: **How many minutes is your train currently delayed?**
7. Bot asks: **What is the planned arrival time at your destination?**
8. Bot asks: **What date are you travelling?**
9. Bot calls `predict_arrival_time()` and returns:
   > "Based on your current delay of 10 minutes at Southampton, I predict
   > your train will arrive at London Waterloo at approximately 14:45,
   > around 10 minutes late."

### Station Lookup
Task 2 reuses the `chatbot/station_lookup.py` module from Task 1 for
CRS code resolution, fuzzy matching, and station name confirmation.

---

## 7. How to Run

```bash
# 1. Train all models (only needed once)
python -m task2.train_models

# 2. (Optional) Evaluate saved models
python -m task2.evaluate

# 3. Run the chatbot
streamlit run app.py
```

### Required libraries
```
pandas, numpy, scikit-learn, joblib, openpyxl, tabulate, dateparser,
streamlit, rapidfuzz
```

---

## 8. Limitations and Possible Improvements

### Limitations
- **Route-specific**: The model is trained only on Weymouth → Waterloo data
  and cannot generalise to other routes without retraining.
- **No real-time data**: The model predicts based on historical patterns;
  it does not access live train feeds.
- **Station-level granularity**: The model predicts delay at a single station
  and does not account for intermediate stops between the current position
  and destination.
- **`late_canc_reason` unknown at query time**: At prediction time, the
  delay cause code is not available (set to 0), reducing the model's
  information.

### Possible Improvements
- **Sequence modelling**: Use the full sequence of stops on a service (via
  `rid`) to predict cumulative delay propagation with an LSTM or transformer.
- **Real-time integration**: Connect to the Network Rail DARWIN push-port
  feed for live delay data instead of relying on user-reported minutes.
- **Additional features**: Weather data, engineering work schedules, and
  platform information could improve accuracy.
- **Cross-route training**: Train on multiple routes to build a more general
  delay prediction model.
- **Hyperparameter tuning**: Use GridSearchCV or Optuna to systematically
  optimise model hyperparameters beyond the manual choices made here.

---

## File Structure

```
task2/
├── __init__.py          # Package marker
├── preprocess.py        # Data loading, cleaning, feature engineering
├── train_models.py      # Train and evaluate 5 ML models
├── evaluate.py          # Standalone model comparison
├── predict.py           # Load best model and make predictions
├── conversation.py      # Task 2 dialogue manager
└── README.md            # This file

models/
├── best_model.pkl       # Best performing model (Random Forest)
├── best_model_meta.pkl  # Metadata (name, scaler flag, metrics)
├── scaler.pkl           # StandardScaler for kNN/MLP
├── location_encoder.pkl # LabelEncoder for station CRS codes
├── knn.pkl              # k-Nearest Neighbours model
├── random_forest.pkl    # Random Forest model
├── linear_regression.pkl# Linear Regression model
├── mlp.pkl              # Neural Network (MLP) model
└── gradient_boosting.pkl# Gradient Boosting model
```
