import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. Load dataset
# ==========================================

file_path = "data/raw/train_FD001.txt"

columns = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
]

sensor_columns = [f"sensor_{i}" for i in range(1, 22)]

columns += sensor_columns

df = pd.read_csv(
    file_path,
    sep=r"\s+",
    header=None,
    names=columns
)


# ==========================================
# 2. Calculate RUL
# ==========================================

df["RUL"] = (
    df.groupby("engine_id")["cycle"]
    .transform("max")
    - df["cycle"]
)


# ==========================================
# 3. Remove constant sensors
# ==========================================

constant_sensors = [
    sensor
    for sensor in sensor_columns
    if df[sensor].std() == 0
]

print("Constant sensors:")
print(constant_sensors)


model_sensors = [
    sensor
    for sensor in sensor_columns
    if sensor not in constant_sensors
]

print("\nSensors used:")
print(model_sensors)


# ==========================================
# 4. Create temporal features
# ==========================================

feature_df = df[
    ["engine_id", "cycle", "RUL"] + model_sensors
].copy()


for sensor in model_sensors:

    # Rolling mean
    feature_df[f"{sensor}_mean"] = (
        feature_df
        .groupby("engine_id")[sensor]
        .transform(
            lambda x: x.rolling(
                window=10,
                min_periods=1
            ).mean()
        )
    )

    # Rolling standard deviation
    feature_df[f"{sensor}_std"] = (
        feature_df
        .groupby("engine_id")[sensor]
        .transform(
            lambda x: x.rolling(
                window=10,
                min_periods=1
            ).std()
        )
    )

    # Rate of change
    feature_df[f"{sensor}_diff"] = (
        feature_df
        .groupby("engine_id")[sensor]
        .diff()
    )


# Fill NaN values created by diff/std
feature_df = feature_df.fillna(0)


print("\nFeature engineering completed.")

print("Original sensor features:", len(model_sensors))

print(
    "Total ML features:",
    len(feature_df.columns) - 3
)


# ==========================================
# 5. Train/validation split by engine
# ==========================================

train_engines = np.arange(1, 81)
val_engines = np.arange(81, 101)

train_df = feature_df[
    feature_df["engine_id"].isin(train_engines)
].copy()

val_df = feature_df[
    feature_df["engine_id"].isin(val_engines)
].copy()


# ==========================================
# 6. Prepare X and y
# ==========================================

feature_columns = [
    column
    for column in feature_df.columns
    if column not in ["engine_id", "cycle", "RUL"]
]

X_train = train_df[feature_columns]
y_train = train_df["RUL"]

X_val = val_df[feature_columns]
y_val = val_df["RUL"]


print("\nTraining shape:", X_train.shape)
print("Validation shape:", X_val.shape)


# ==========================================
# 7. Scale features
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)


# ==========================================
# 8. Train Random Forest
# ==========================================

print("\nTraining improved Random Forest...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_scaled,
    y_train
)

print("Training completed!")


# ==========================================
# 9. Predictions
# ==========================================

y_pred = model.predict(X_val_scaled)


# ==========================================
# 10. Evaluation
# ==========================================

mae = mean_absolute_error(y_val, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_val, y_pred)
)

r2 = r2_score(y_val, y_pred)


print("\nImproved Model Performance")
print("==========================")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))


# ==========================================
# 11. Actual vs Predicted
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(
    y_val.values[:300],
    label="Actual RUL"
)

plt.plot(
    y_pred[:300],
    label="Predicted RUL"
)

plt.xlabel("Validation Sample")
plt.ylabel("RUL (Cycles)")

plt.title(
    "Improved Model - Actual vs Predicted RUL"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()