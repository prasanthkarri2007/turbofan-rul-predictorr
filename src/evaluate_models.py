import os
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import load_model


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "models"
DATA_DIR = "data/processed"
RAW_DATA_PATH = "data/raw/train_FD001.txt"

SEQUENCE_LENGTH = 30
MAX_RUL = 125


# ============================================================
# IMPORTANT:
# THESE ARE THE 15 SENSORS USED BY THE ORIGINAL
# GRU/LSTM TRAINING PIPELINE
# ============================================================

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


# ============================================================
# VALIDATION ENGINES
# ============================================================

VAL_ENGINE_IDS = range(81, 101)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, rmse, r2


# ============================================================
# CREATE SEQUENCES
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
# LOAD RANDOM FOREST VALIDATION DATA
# ============================================================

print("=" * 60)
print("LOADING RANDOM FOREST VALIDATION DATA")
print("=" * 60)

X_val_rf = np.load(
    os.path.join(
        DATA_DIR,
        "X_val.npy"
    )
)

y_val_rf = np.load(
    os.path.join(
        DATA_DIR,
        "y_val.npy"
    )
)

print(
    "RF X_val shape:",
    X_val_rf.shape
)

print(
    "RF y_val shape:",
    y_val_rf.shape
)


# ============================================================
# LOAD RAW DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING RAW DATA FOR GRU/LSTM")
print("=" * 60)

column_names = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21"
]

raw_df = pd.read_csv(
    RAW_DATA_PATH,
    sep=r"\s+",
    header=None,
    names=column_names
)

print(
    "Raw dataset shape:",
    raw_df.shape
)


# ============================================================
# CALCULATE RUL
# ============================================================

max_cycles = (
    raw_df
    .groupby("engine_id")["cycle"]
    .transform("max")
)

raw_df["RUL"] = (
    max_cycles - raw_df["cycle"]
)

# Same RUL cap used during training
raw_df["RUL"] = raw_df["RUL"].clip(
    upper=MAX_RUL
)


# ============================================================
# VALIDATION DATA
# ============================================================

val_df = raw_df[
    raw_df["engine_id"].isin(
        VAL_ENGINE_IDS
    )
].copy()

print(
    "Validation engines:",
    val_df["engine_id"].nunique()
)


# ============================================================
# CREATE 30-TIMESTEP SEQUENCES
# ============================================================

print("\n" + "=" * 60)
print("CREATING VALIDATION SEQUENCES")
print("=" * 60)

X_val_seq, y_val_seq, val_sequence_engines = (
    create_sequences(
        val_df,
        MODEL_SENSORS,
        SEQUENCE_LENGTH
    )
)

print(
    "Sequence X shape:",
    X_val_seq.shape
)

print(
    "Sequence y shape:",
    y_val_seq.shape
)

print(
    "Expected format: "
    "(samples, 30, 15)"
)


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# RANDOM FOREST
# ============================================================

rf_path = os.path.join(
    MODEL_DIR,
    "random_forest_rul.pkl"
)

if os.path.exists(rf_path):

    print("\n" + "=" * 60)
    print("EVALUATING RANDOM FOREST")
    print("=" * 60)

    rf_model = joblib.load(
        rf_path
    )

    rf_pred = rf_model.predict(
        X_val_rf
    )

    mae, rmse, r2 = calculate_metrics(
        y_val_rf,
        rf_pred
    )

    print(
        "MAE :",
        round(mae, 2)
    )

    print(
        "RMSE:",
        round(rmse, 2)
    )

    print(
        "R2  :",
        round(r2, 4)
    )

    results.append([
        "Random Forest",
        mae,
        rmse,
        r2
    ])


# ============================================================
# LSTM
# ============================================================

lstm_path = os.path.join(
    MODEL_DIR,
    "lstm_rul_model.keras"
)

lstm_scaler_path = os.path.join(
    MODEL_DIR,
    "lstm_scaler.pkl"
)

if (
    os.path.exists(lstm_path)
    and os.path.exists(lstm_scaler_path)
):

    print("\n" + "=" * 60)
    print("EVALUATING LSTM")
    print("=" * 60)

    lstm_model = load_model(
        lstm_path
    )

    lstm_scaler = joblib.load(
        lstm_scaler_path
    )

    print(
        "LSTM scaler features:",
        lstm_scaler.n_features_in_
    )

    # Flatten the time dimension temporarily
    # 3D: (samples, 30, 15)
    # -> 2D: (samples*30, 15)

    X_lstm_2d = X_val_seq.reshape(
        -1,
        X_val_seq.shape[-1]
    )

    X_lstm_scaled_2d = (
        lstm_scaler.transform(
            X_lstm_2d
        )
    )

    # Restore sequence structure
    X_lstm_scaled = (
        X_lstm_scaled_2d.reshape(
            X_val_seq.shape
        )
    )

    print(
        "LSTM input shape:",
        X_lstm_scaled.shape
    )

    lstm_pred = lstm_model.predict(
        X_lstm_scaled,
        verbose=0
    ).flatten()

    mae, rmse, r2 = calculate_metrics(
        y_val_seq,
        lstm_pred
    )

    print(
        "LSTM MAE :",
        round(mae, 2)
    )

    print(
        "LSTM RMSE:",
        round(rmse, 2)
    )

    print(
        "LSTM R2  :",
        round(r2, 4)
    )

    results.append([
        "LSTM",
        mae,
        rmse,
        r2
    ])


# ============================================================
# GRU
# ============================================================

gru_path = os.path.join(
    MODEL_DIR,
    "gru_rul_model.keras"
)

gru_scaler_path = os.path.join(
    MODEL_DIR,
    "gru_scaler.pkl"
)

if (
    os.path.exists(gru_path)
    and os.path.exists(gru_scaler_path)
):

    print("\n" + "=" * 60)
    print("EVALUATING GRU")
    print("=" * 60)

    gru_model = load_model(
        gru_path
    )

    gru_scaler = joblib.load(
        gru_scaler_path
    )

    print(
        "GRU scaler features:",
        gru_scaler.n_features_in_
    )

    # Flatten time dimension
    X_gru_2d = X_val_seq.reshape(
        -1,
        X_val_seq.shape[-1]
    )

    X_gru_scaled_2d = (
        gru_scaler.transform(
            X_gru_2d
        )
    )

    # Restore 3D sequence
    X_gru_scaled = (
        X_gru_scaled_2d.reshape(
            X_val_seq.shape
        )
    )

    print(
        "GRU input shape:",
        X_gru_scaled.shape
    )

    gru_pred = gru_model.predict(
        X_gru_scaled,
        verbose=0
    ).flatten()

    mae, rmse, r2 = calculate_metrics(
        y_val_seq,
        gru_pred
    )

    print(
        "GRU MAE :",
        round(mae, 2)
    )

    print(
        "GRU RMSE:",
        round(rmse, 2)
    )

    print(
        "GRU R2  :",
        round(r2, 4)
    )

    results.append([
        "GRU",
        mae,
        rmse,
        r2
    ])


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "RMSE",
        "R2"
    ]
)


# ============================================================
# SORT BY MAE
# ============================================================

results_df = results_df.sort_values(
    by="MAE"
).reset_index(
    drop=True
)


# ============================================================
# DISPLAY
# ============================================================

print("\n")
print("=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.2f}".format,
            "RMSE": "{:.2f}".format,
            "R2": "{:.4f}".format
        }
    )
)


# ============================================================
# BEST MODEL
# ============================================================

if len(results_df) > 0:

    best = results_df.iloc[0]

    print("\n")
    print("=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(
        "Model :",
        best["Model"]
    )

    print(
        "MAE   :",
        round(best["MAE"], 2)
    )

    print(
        "RMSE  :",
        round(best["RMSE"], 2)
    )

    print(
        "R2    :",
        round(best["R2"], 4)
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = os.path.join(
    DATA_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\nResults saved to:")
print(output_path)

print("\nEvaluation completed successfully.")