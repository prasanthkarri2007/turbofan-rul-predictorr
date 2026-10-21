import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 1. CONFIGURATION
# ============================================================

FILE_PATH = "data/raw/train_FD001.txt"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "gru_rul_model.keras"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "gru_scaler.pkl"
)

SEQUENCE_LENGTH = 30

MAX_RUL = 125

RANDOM_SEED = 42

TRAIN_ENGINES = 80
VALIDATION_ENGINES = 20


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)


# ============================================================
# 3. CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 4. COLUMN NAMES
# ============================================================

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

columns.extend(sensor_columns)


# ============================================================
# 5. LOAD DATA
# ============================================================

print("=" * 60)
print("GRU TURBOFAN RUL MODEL")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(
    FILE_PATH,
    sep=r"\s+",
    header=None,
    names=columns
)

print("Dataset shape:", df.shape)


# ============================================================
# 6. CALCULATE RUL
# ============================================================

print("\nCalculating RUL...")

max_cycles = (
    df.groupby("engine_id")["cycle"]
    .max()
    .reset_index()
)

max_cycles.columns = [
    "engine_id",
    "max_cycle"
]

df = df.merge(
    max_cycles,
    on="engine_id",
    how="left"
)

df["RUL"] = (
    df["max_cycle"] - df["cycle"]
)

# Cap RUL at MAX_RUL
df["RUL"] = df["RUL"].clip(
    upper=MAX_RUL
)

df.drop(
    columns=["max_cycle"],
    inplace=True
)


# ============================================================
# 7. SELECT 15 MODEL SENSORS
# ============================================================

# These are the same 15 sensors used by
# the Random Forest / original GRU pipeline.

MODEL_SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21"
]

print("\nModel sensors:")

for sensor in MODEL_SENSORS:
    print(sensor)

print(
    "\nNumber of sensors:",
    len(MODEL_SENSORS)
)


# ============================================================
# 8. SPLIT BY ENGINE
# ============================================================

print("\nSplitting engines...")

engine_ids = sorted(
    df["engine_id"].unique()
)

np.random.seed(RANDOM_SEED)

np.random.shuffle(
    engine_ids
)

train_engine_ids = engine_ids[
    :TRAIN_ENGINES
]

validation_engine_ids = engine_ids[
    TRAIN_ENGINES:
    TRAIN_ENGINES + VALIDATION_ENGINES
]

train_df = df[
    df["engine_id"].isin(
        train_engine_ids
    )
].copy()

val_df = df[
    df["engine_id"].isin(
        validation_engine_ids
    )
].copy()

print(
    "\nTraining engines:",
    len(train_engine_ids)
)

print(
    "Validation engines:",
    len(validation_engine_ids)
)


# ============================================================
# 9. CREATE SEQUENCES
# ============================================================

def create_sequences(
    data,
    sensors,
    sequence_length
):

    X = []
    y = []
    engine_ids = []

    for engine_id, engine_data in data.groupby(
        "engine_id"
    ):

        engine_data = engine_data.sort_values(
            "cycle"
        )

        sensor_values = engine_data[
            sensors
        ].values

        rul_values = engine_data[
            "RUL"
        ].values

        for i in range(
            sequence_length,
            len(engine_data) + 1
        ):

            X.append(
                sensor_values[
                    i - sequence_length:i
                ]
            )

            y.append(
                rul_values[i - 1]
            )

            engine_ids.append(
                engine_id
            )

    return (
        np.array(X),
        np.array(y),
        np.array(engine_ids)
    )


# ============================================================
# 10. CREATE TRAINING SEQUENCES
# ============================================================

print("\nCreating training sequences...")

(
    X_train,
    y_train,
    train_sequence_engines
) = create_sequences(
    train_df,
    MODEL_SENSORS,
    SEQUENCE_LENGTH
)


# ============================================================
# 11. CREATE VALIDATION SEQUENCES
# ============================================================

print("Creating validation sequences...")

(
    X_val,
    y_val,
    val_sequence_engines
) = create_sequences(
    val_df,
    MODEL_SENSORS,
    SEQUENCE_LENGTH
)


# ============================================================
# 12. DISPLAY SEQUENCE SHAPES
# ============================================================

print("\nSequence shapes:")

print(
    "X_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "X_val:",
    X_val.shape
)

print(
    "y_val:",
    y_val.shape
)


# ============================================================
# 13. SCALE SEQUENCE DATA
# ============================================================

print("\nScaling GRU sequence data...")

n_train = X_train.shape[0]
n_val = X_val.shape[0]

n_features = X_train.shape[2]

print(
    "Number of features:",
    n_features
)


# ------------------------------------------------------------
# Flatten only for scaler
# ------------------------------------------------------------

X_train_2d = X_train.reshape(
    -1,
    n_features
)

X_val_2d = X_val.reshape(
    -1,
    n_features
)


# ------------------------------------------------------------
# Create scaler
# ------------------------------------------------------------

scaler = MinMaxScaler()


# ------------------------------------------------------------
# FIT ONLY ON TRAINING DATA
# ------------------------------------------------------------

X_train_scaled_2d = scaler.fit_transform(
    X_train_2d
)


# ------------------------------------------------------------
# TRANSFORM VALIDATION DATA
# ------------------------------------------------------------

X_val_scaled_2d = scaler.transform(
    X_val_2d
)


# ------------------------------------------------------------
# Convert back to sequence format
# ------------------------------------------------------------

X_train_scaled = X_train_scaled_2d.reshape(
    n_train,
    SEQUENCE_LENGTH,
    n_features
)

X_val_scaled = X_val_scaled_2d.reshape(
    n_val,
    SEQUENCE_LENGTH,
    n_features
)


# ============================================================
# 14. VERIFY SCALER
# ============================================================

print(
    "\nScaler expects features:",
    scaler.n_features_in_
)

print(
    "Scaled training shape:",
    X_train_scaled.shape
)

print(
    "Scaled validation shape:",
    X_val_scaled.shape
)


# ============================================================
# 15. SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    SCALER_PATH
)

print(
    "\nGRU scaler saved to:",
    SCALER_PATH
)


# ============================================================
# 16. BUILD GRU MODEL
# ============================================================

print("\nBuilding GRU model...")

model = Sequential()


# First GRU layer
model.add(
    GRU(
        64,
        return_sequences=True,
        input_shape=(
            SEQUENCE_LENGTH,
            n_features
        )
    )
)


# Dropout
model.add(
    Dropout(0.2)
)


# Second GRU layer
model.add(
    GRU(
        32,
        return_sequences=False
    )
)


# Dropout
model.add(
    Dropout(0.2)
)


# Dense layer
model.add(
    Dense(
        16,
        activation="relu"
    )
)


# Output layer
model.add(
    Dense(
        1
    )
)


# ============================================================
# 17. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


print("\nModel summary:")

model.summary()


# ============================================================
# 18. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)


# ============================================================
# 19. TRAIN GRU
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING GRU")
print("=" * 60)

history = model.fit(
    X_train_scaled,
    y_train,

    validation_data=(
        X_val_scaled,
        y_val
    ),

    epochs=50,

    batch_size=128,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


print("\nGRU training completed!")


# ============================================================
# 20. PREDICTION
# ============================================================

print("\nGenerating validation predictions...")

y_pred = model.predict(
    X_val_scaled,
    verbose=0
).flatten()


# ============================================================
# 21. CALCULATE METRICS
# ============================================================

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


# ============================================================
# 22. DISPLAY PERFORMANCE
# ============================================================

print("\n")
print("=" * 60)
print("GRU PERFORMANCE")
print("=" * 60)

print(
    "MAE  :",
    round(mae, 2)
)

print(
    "RMSE :",
    round(rmse, 2)
)

print(
    "R2   :",
    round(r2, 4)
)


# ============================================================
# 23. SAVE MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print(
    "\nGRU model saved successfully!"
)

print(
    "Saved to:",
    MODEL_PATH
)

print(
    "Scaler saved to:",
    SCALER_PATH
)


# ============================================================
# 24. PLOT TRAINING LOSS
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "GRU Training vs Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "MSE Loss"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 25. PLOT ACTUAL VS PREDICTED RUL
# ============================================================

plot_count = min(
    300,
    len(y_val)
)

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    y_val[:plot_count],
    label="Actual RUL"
)

plt.plot(
    y_pred[:plot_count],
    label="Predicted RUL"
)

plt.title(
    "GRU - Actual vs Predicted RUL"
)

plt.xlabel(
    "Validation Sequence"
)

plt.ylabel(
    "RUL (Cycles)"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 26. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 60)
print("GRU TRAINING COMPLETE")
print("=" * 60)

print(
    f"Final MAE  : {mae:.2f} cycles"
)

print(
    f"Final RMSE : {rmse:.2f} cycles"
)

print(
    f"Final R2   : {r2:.4f}"
)

print(
    "\nModel:",
    MODEL_PATH
)

print(
    "Scaler:",
    SCALER_PATH
)