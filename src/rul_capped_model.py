import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt


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

sensor_columns = [
    f"sensor_{i}"
    for i in range(1, 22)
]

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
# 3. Cap RUL
# ==========================================

MAX_RUL = 125

df["RUL"] = df["RUL"].clip(
    upper=MAX_RUL
)


print("Maximum RUL after capping:")
print(df["RUL"].max())


# ==========================================
# 4. Remove near-constant sensors
# ==========================================

sensor_std = df[sensor_columns].std()

constant_sensors = sensor_std[
    sensor_std < 1e-5
].index.tolist()


print("\nNear-constant sensors:")
print(constant_sensors)


model_sensors = [
    sensor
    for sensor in sensor_columns
    if sensor not in constant_sensors
]


print("\nSensors used:")
print(model_sensors)

print("\nNumber of sensors:")
print(len(model_sensors))


# ==========================================
# 5. Train / validation split by engine
# ==========================================

train_engines = np.arange(1, 81)

val_engines = np.arange(81, 101)


train_df = df[
    df["engine_id"].isin(train_engines)
].copy()


val_df = df[
    df["engine_id"].isin(val_engines)
].copy()


# ==========================================
# 6. Prepare features
# ==========================================

X_train = train_df[model_sensors]

y_train = train_df["RUL"]


X_val = val_df[model_sensors]

y_val = val_df["RUL"]


print("\nTraining shape:")
print(X_train.shape)


print("Validation shape:")
print(X_val.shape)


# ==========================================
# 7. Train Extra Trees
# ==========================================

print("\nTraining RUL-capped Extra Trees...")


model = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    max_features=1.0,
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print("Training completed!")


# ==========================================
# 8. Predictions
# ==========================================

y_pred = model.predict(X_val)


# ==========================================
# 9. Clip predictions
# ==========================================

y_pred = np.clip(
    y_pred,
    0,
    MAX_RUL
)


# ==========================================
# 10. Evaluation
# ==========================================

mae = mean_absolute_error(
    y_val,
    y_pred
)


rmse = np.sqrt(
    mean_squared_error(
        y_val,
        y_pred
    )
)


r2 = r2_score(
    y_val,
    y_pred
)


print("\nRUL-CAPPED EXTRA TREES PERFORMANCE")
print("===================================")


print("MAE :", round(mae, 2))

print("RMSE:", round(rmse, 2))

print("R²  :", round(r2, 4))


# ==========================================
# 11. Feature importance
# ==========================================

importance = pd.Series(
    model.feature_importances_,
    index=model_sensors
).sort_values(
    ascending=False
)


print("\nFeature importance:")
print(importance)


# ==========================================
# 12. Actual vs Predicted
# ==========================================

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
    "RUL-Capped Extra Trees - Actual vs Predicted"
)


plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()
