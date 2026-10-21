
import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model


# ============================================================
# 1. CONFIGURATION
# ============================================================

TEST_FILE = "data/raw/test_FD001.txt"
MODEL_PATH = "models/gru_rul_model.keras"
SCALER_PATH = "models/gru_scaler.pkl"

OUTPUT_DIR = "data/processed"
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "gru_rul_predictions.csv"
)

SEQUENCE_LENGTH = 30
MAX_RUL = 125


# ============================================================
# 2. EXACT SENSORS USED DURING GRU TRAINING
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
# 3. CHECK FILES
# ============================================================

for path in [TEST_FILE, MODEL_PATH, SCALER_PATH]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 4. LOAD MODEL AND SCALER
# ============================================================

print("=" * 65)
print("TURBOFAN REMAINING USEFUL LIFE PREDICTION")
print("=" * 65)

print("\nLoading GRU model...")
model = load_model(MODEL_PATH)

print("Loading GRU scaler...")
scaler = joblib.load(SCALER_PATH)


# ============================================================
# 5. VERIFY FEATURE COUNT
# ============================================================

if hasattr(scaler, "n_features_in_"):
    expected_features = scaler.n_features_in_

    if expected_features != len(MODEL_SENSORS):
        raise ValueError(
            f"Scaler expects {expected_features} features, "
            f"but MODEL_SENSORS contains {len(MODEL_SENSORS)}."
        )

if hasattr(scaler, "feature_names_in_"):
    saved_features = list(scaler.feature_names_in_)

    if saved_features != MODEL_SENSORS:
        raise ValueError(
            "Sensor names or order differ from the scaler's "
            "training features.\n"
            f"Scaler features: {saved_features}\n"
            f"Prediction features: {MODEL_SENSORS}"
        )

print(f"Sensor count: {len(MODEL_SENSORS)}")
print(f"Sequence length: {SEQUENCE_LENGTH}")


# ============================================================
# 6. LOAD FD001 TEST DATA
# ============================================================

print("\nLoading test dataset...")

columns = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3"
] + [
    f"sensor_{i}" for i in range(1, 22)
]

test_df = pd.read_csv(
    TEST_FILE,
    sep=r"\s+",
    header=None,
    names=columns
)

if test_df.empty:
    raise ValueError("The test dataset is empty.")

if test_df[MODEL_SENSORS].isnull().any().any():
    raise ValueError(
        "Missing values found in selected sensor columns."
    )

print(f"Rows loaded: {len(test_df)}")
print(f"Engines found: {test_df['engine_id'].nunique()}")


# ============================================================
# 7. CREATE ONE LATEST SEQUENCE PER ENGINE
# ============================================================

print("\nCreating latest engine sequences...")

sequences = []
engine_ids = []
last_cycles = []

for engine_id, engine_data in test_df.groupby("engine_id"):
    engine_data = engine_data.sort_values("cycle")

    sensor_values = engine_data[MODEL_SENSORS].to_numpy(
        dtype=np.float32
    )

    if len(sensor_values) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Engine {engine_id} has only "
            f"{len(sensor_values)} cycles; "
            f"{SEQUENCE_LENGTH} are required."
        )

    sequences.append(sensor_values[-SEQUENCE_LENGTH:])
    engine_ids.append(int(engine_id))
    last_cycles.append(int(engine_data["cycle"].iloc[-1]))

X_test = np.asarray(sequences, dtype=np.float32)

expected_shape = (
    len(engine_ids),
    SEQUENCE_LENGTH,
    len(MODEL_SENSORS)
)

if X_test.shape != expected_shape:
    raise ValueError(
        f"Unexpected sequence shape: {X_test.shape}; "
        f"expected {expected_shape}"
    )

print(f"Sequence shape: {X_test.shape}")


# ============================================================
# 8. SCALE USING THE TRAINED GRU SCALER
# ============================================================

print("\nScaling sensor sequences...")

n_engines, n_steps, n_features = X_test.shape

X_scaled = scaler.transform(
    X_test.reshape(-1, n_features)
).reshape(
    n_engines,
    n_steps,
    n_features
).astype(np.float32)

print(f"Scaled shape: {X_scaled.shape}")


# ============================================================
# 9. PREDICT RUL
# ============================================================

print("\nPredicting RUL...")

predictions = model.predict(
    X_scaled,
    verbose=0
).reshape(-1)

if len(predictions) != len(engine_ids):
    raise ValueError(
        "Prediction count does not match engine count."
    )

if not np.isfinite(predictions).all():
    raise ValueError(
        "Model returned invalid predictions (NaN or infinity)."
    )

predictions = np.clip(predictions, 0, MAX_RUL)


# ============================================================
# 10. HEALTH STATUS
# ============================================================

def get_health_status(rul):
    if rul > 80:
        return "HEALTHY"
    elif rul > 40:
        return "WARNING"
    elif rul > 20:
        return "CRITICAL"
    else:
        return "HIGH RISK"


# ============================================================
# 11. BUILD RESULTS
# ============================================================

results_df = pd.DataFrame({
    "engine_id": engine_ids,
    "last_cycle": last_cycles,
    "predicted_RUL": np.round(predictions, 2)
})

results_df["health_status"] = (
    results_df["predicted_RUL"].apply(get_health_status)
)

results_df = results_df.sort_values(
    "engine_id"
).reset_index(drop=True)


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("GRU RUL PREDICTIONS")
print("=" * 65)

print(results_df.to_string(index=False))

print("\n" + "=" * 65)
print("PREDICTION SUMMARY")
print("=" * 65)

print(f"Engines predicted: {len(results_df)}")
print(
    f"Mean predicted RUL: "
    f"{results_df['predicted_RUL'].mean():.2f} cycles"
)
print(
    f"Minimum predicted RUL: "
    f"{results_df['predicted_RUL'].min():.2f} cycles"
)
print(
    f"Maximum predicted RUL: "
    f"{results_df['predicted_RUL'].max():.2f} cycles"
)
print(
    f"Unique RUL predictions: "
    f"{results_df['predicted_RUL'].nunique()}"
)

print("\nHealth status counts:")
print(results_df["health_status"].value_counts().to_string())


# ============================================================
# 13. SAVE CSV
# ============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 65)
print("PREDICTION COMPLETED")
print("=" * 65)
print(f"Results saved to: {OUTPUT_PATH}")
