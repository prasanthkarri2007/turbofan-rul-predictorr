import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. Load the NASA C-MAPSS training dataset
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
# 2. Basic dataset information
# ==========================================

print(df.head())

print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nNumber of engines:", df["engine_id"].nunique())


# ==========================================
# 3. Cycles per engine
# ==========================================

print("\nCycles per engine:")
print(
    df.groupby("engine_id")["cycle"]
    .max()
    .head(10)
)


# ==========================================
# 4. Sensor statistics
# ==========================================

print("\nSensor statistics:")

sensor_columns = [
    f"sensor_{i}"
    for i in range(1, 22)
]

print(
    df[sensor_columns]
    .describe()
    .T
)


# ==========================================
# 5. Calculate Remaining Useful Life (RUL)
# ==========================================

df["RUL"] = (
    df.groupby("engine_id")["cycle"]
    .transform("max")
    - df["cycle"]
)

print("\nRUL example:")

print(
    df[
        ["engine_id", "cycle", "RUL"]
    ].head(10)
)


# ==========================================
# 6. Select Engine 1
# ==========================================

engine_1 = df[
    df["engine_id"] == 1
].copy()


# ==========================================
# 7. Sensor 3 degradation trend
# ==========================================

engine_1["sensor_3_smooth"] = (
    engine_1["sensor_3"]
    .rolling(window=10)
    .mean()
)

plt.figure(figsize=(8, 5))

plt.plot(
    engine_1["cycle"],
    engine_1["sensor_3_smooth"]
)

plt.xlabel("Flight Cycle")
plt.ylabel("Sensor 3 Value")

plt.title(
    "Engine 1 - Sensor 3 Degradation Trend"
)

plt.tight_layout()

plt.show()


# ==========================================
# 8. RUL vs Flight Cycle
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    engine_1["cycle"],
    engine_1["RUL"]
)

plt.xlabel("Flight Cycle")

plt.ylabel(
    "Remaining Useful Life (RUL)"
)

plt.title(
    "Engine 1 - RUL vs Flight Cycle"
)

plt.tight_layout()

plt.show()

print("\nSensor correlation with RUL:")

correlations = (
    df[sensor_columns + ["RUL"]]
    .corr()["RUL"]
    .drop("RUL")
    .sort_values()
)

print(correlations)

constant_sensors = [
    sensor
    for sensor in sensor_columns
    if df[sensor].nunique() <= 1
]

print("\nConstant sensors:")
print(constant_sensors)

model_sensors = [
    sensor
    for sensor in sensor_columns
    if sensor not in constant_sensors
]

print("\nSensors used for ML:")
print(model_sensors)

print("\nNumber of sensors used:", len(model_sensors))


# ==========================================
# 11. Create ML features and target
# ==========================================

X = df[model_sensors].copy()

y = df["RUL"].copy()

print("\nML feature shape:", X.shape)

print("Target shape:", y.shape)

print("\nFirst 5 feature rows:")
print(X.head())

print("\nFirst 5 target values:")
print(y.head())