import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import joblib
import os


# ==========================================
# 1. Configuration
# ==========================================

FILE_PATH = "data/raw/train_FD001.txt"

MAX_RUL = 125

SEQUENCE_LENGTH = 30

RANDOM_SEED = 42


# ==========================================
# 2. Reproducibility
# ==========================================

np.random.seed(RANDOM_SEED)


# ==========================================
# 3. Load dataset
# ==========================================

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
    FILE_PATH,
    sep=r"\s+",
    header=None,
    names=columns
)


print("Dataset loaded successfully!")

print("Shape:", df.shape)

print("Number of engines:", df["engine_id"].nunique())


# ==========================================
# 4. Calculate RUL
# ==========================================

df["RUL"] = (
    df.groupby("engine_id")["cycle"]
    .transform("max")
    - df["cycle"]
)


# ==========================================
# 5. Cap RUL
# ==========================================

df["RUL"] = df["RUL"].clip(
    upper=MAX_RUL
)


print("\nMaximum RUL:")
print(df["RUL"].max())


# ==========================================
# 6. Remove constant sensors
# ==========================================

sensor_std = df[sensor_columns].std()

constant_sensors = sensor_std[
    sensor_std < 1e-5
].index.tolist()


model_sensors = [
    sensor
    for sensor in sensor_columns
    if sensor not in constant_sensors
]


print("\nConstant sensors:")
print(constant_sensors)


print("\nSensors used:")
print(model_sensors)


print("\nNumber of sensors:")
print(len(model_sensors))


# ==========================================
# 7. Train / Validation engine split
# ==========================================

train_engines = np.arange(1, 81)

val_engines = np.arange(81, 101)


train_df = df[
    df["engine_id"].isin(train_engines)
].copy()


val_df = df[
    df["engine_id"].isin(val_engines)
].copy()


print("\nTraining engines:", len(train_engines))

print("Validation engines:", len(val_engines))


# ==========================================
# 8. Scale sensor data
# ==========================================

scaler = MinMaxScaler()


train_df[model_sensors] = scaler.fit_transform(
    train_df[model_sensors]
)


val_df[model_sensors] = scaler.transform(
    val_df[model_sensors]
)


# Save scaler
os.makedirs("models", exist_ok=True)

joblib.dump(
    scaler,
    "models/lstm_scaler.pkl"
)


print("\nSensor scaling completed.")


# ==========================================
# 9. Create sequences
# ==========================================

def create_sequences(data, sensors, sequence_length):

    X = []
    y = []

    engine_ids = []

    for engine_id, engine_data in data.groupby("engine_id"):

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


# ==========================================
# 10. Generate training sequences
# ==========================================

X_train, y_train, train_sequence_engines = (
    create_sequences(
        train_df,
        model_sensors,
        SEQUENCE_LENGTH
    )
)


# ==========================================
# 11. Generate validation sequences
# ==========================================

X_val, y_val, val_sequence_engines = (
    create_sequences(
        val_df,
        model_sensors,
        SEQUENCE_LENGTH
    )
)


print("\nSequence shapes:")

print("X_train:", X_train.shape)

print("y_train:", y_train.shape)

print("X_val:", X_val.shape)

print("y_val:", y_val.shape)


# ==========================================
# 12. Build LSTM model
# ==========================================

print("\nBuilding LSTM model...")


model = Sequential([

    LSTM(
        64,
        return_sequences=True,
        input_shape=(
            SEQUENCE_LENGTH,
            len(model_sensors)
        )
    ),

    Dropout(0.2),

    LSTM(
        32
    ),

    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        1
    )
])


model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


model.summary()


# ==========================================
# 13. Early stopping
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)


# ==========================================
# 14. Train LSTM
# ==========================================

print("\nTraining LSTM...")


history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=50,

    batch_size=128,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


print("\nLSTM training completed!")


# ==========================================
# 15. Predict validation RUL
# ==========================================

y_pred = model.predict(
    X_val,
    verbose=0
).flatten()


# ==========================================
# 16. Clip predictions
# ==========================================

y_pred = np.clip(
    y_pred,
    0,
    MAX_RUL
)


# ==========================================
# 17. Calculate metrics
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


print("\nLSTM PERFORMANCE")
print("================")


print("MAE :", round(mae, 2))

print("RMSE:", round(rmse, 2))

print("R²  :", round(r2, 4))


# ==========================================
# 18. Save model
# ==========================================

model_path = (
    "models/lstm_rul_model.keras"
)


model.save(
    model_path
)


print("\nLSTM model saved successfully!")

print(
    "Saved to:",
    model_path
)


# ==========================================
# 19. Training history
# ==========================================

plt.figure(
    figsize=(10, 5)
)


plt.plot(
    history.history["loss"],
    label="Training Loss"
)


plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)


plt.xlabel("Epoch")

plt.ylabel("MSE Loss")

plt.title(
    "LSTM Training vs Validation Loss"
)


plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()


# ==========================================
# 20. Actual vs Predicted RUL
# ==========================================

plt.figure(
    figsize=(10, 5)
)


plt.plot(
    y_val[:300],
    label="Actual RUL"
)


plt.plot(
    y_pred[:300],
    label="Predicted RUL"
)


plt.xlabel(
    "Validation Sequence"
)


plt.ylabel(
    "RUL (Cycles)"
)


plt.title(
    "LSTM - Actual vs Predicted RUL"
)


plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()