import pandas as pd


# ==========================================
# 1. Load the NASA FD001 training dataset
# ==========================================

file_path = "data/raw/train_FD001.txt"

columns = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
]

columns += [f"sensor_{i}" for i in range(1, 22)]

df = pd.read_csv(
    file_path,
    sep=r"\s+",
    header=None,
    names=columns
)


# ==========================================
# 2. Calculate Remaining Useful Life (RUL)
# ==========================================

df["RUL"] = (
    df.groupby("engine_id")["cycle"]
    .transform("max")
    - df["cycle"]
)


print("Dataset loaded successfully!")

print("Shape:", df.shape)

print("Number of engines:", df["engine_id"].nunique())

print("\nRUL example:")
print(df[["engine_id", "cycle", "RUL"]].head(10))


# ==========================================
# 3. Split engines into train and validation
# ==========================================

engine_ids = sorted(df["engine_id"].unique())

train_engines = engine_ids[:80]
val_engines = engine_ids[80:]

train_df = df[
    df["engine_id"].isin(train_engines)
].copy()

val_df = df[
    df["engine_id"].isin(val_engines)
].copy()


print("\nTraining engines:", len(train_engines))
print("Validation engines:", len(val_engines))

print("\nTraining shape:", train_df.shape)
print("Validation shape:", val_df.shape)

print("\nTraining engine IDs:")
print(train_engines)

print("\nValidation engine IDs:")
print(val_engines)


# ==========================================
# 4. Create training and validation datasets
# ==========================================

sensor_columns = [
    f"sensor_{i}"
    for i in range(1, 22)
]

# Remove sensors that never change
constant_sensors = [
    sensor
    for sensor in sensor_columns
    if train_df[sensor].nunique() <= 1
]

# Sensors that will be used by the model
model_sensors = [
    sensor
    for sensor in sensor_columns
    if sensor not in constant_sensors
]


# Training data
X_train = train_df[model_sensors].copy()
y_train = train_df["RUL"].copy()


# Validation data
X_val = val_df[model_sensors].copy()
y_val = val_df["RUL"].copy()


print("\nConstant sensors:")
print(constant_sensors)

print("\nModel sensors:")
print(model_sensors)

print("\nX_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("\nX_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)


# ==========================================
# 5. Scale sensor features
# ==========================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)


print("\nScaled training shape:", X_train_scaled.shape)
print("Scaled validation shape:", X_val_scaled.shape)

print("\nFirst scaled training row:")
print(X_train_scaled[0])


# ==========================================
# 6. Train baseline Random Forest model
# ==========================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(X_train_scaled, y_train)

print("Training completed!")


# ==========================================
# 7. Validate model
# ==========================================

y_pred = model.predict(X_val_scaled)

mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print("\nModel Performance:")
print("------------------")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))


# ==========================================
# 8. Save trained model
# ==========================================

import joblib

model_path = "models/random_forest_rul.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Saved to:", model_path)


# ==========================================
# 9. Actual vs Predicted RUL
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
plt.title("Actual vs Predicted RUL")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# SAVE PREPARED DATA FOR MODEL EVALUATION
# ============================================================

np.save(
    "data/processed/X_train.npy",
    X_train_scaled
)

np.save(
    "data/processed/y_train.npy",
    y_train
)

np.save(
    "data/processed/X_val.npy",
    X_val_scaled
)

np.save(
    "data/processed/y_val.npy",
    y_val
)

print("\nPrepared datasets saved successfully!")
print("X_train.npy")
print("y_train.npy")
print("X_val.npy")
print("y_val.npy")