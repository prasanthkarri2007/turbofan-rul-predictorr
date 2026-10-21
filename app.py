
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from tensorflow.keras.models import load_model


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "gru_rul_model.keras"
SCALER_PATH = ROOT / "models" / "gru_scaler.pkl"

SEQUENCE_LENGTH = 30
MAX_RUL = 125

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
    "sensor_21",
]


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Turbofan RUL Predictor",
    page_icon="⚙️",
    layout="wide",
)

st.title("Turbofan RUL Predictor")
st.caption(
    "Predictive maintenance dashboard powered by a trained GRU model"
)


# =========================================================
# LOAD MODEL AND SCALER
# =========================================================

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"GRU model not found: {MODEL_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"GRU scaler not found: {SCALER_PATH}"
        )

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


try:
    model, scaler = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the GRU model or scaler: {exc}")
    st.info(
        "Run src/gru_model.py first and verify the saved "
        "model and scaler filenames."
    )
    st.stop()


# Check that the saved artifacts match the intended input.
expected_shape = (None, SEQUENCE_LENGTH, len(MODEL_SENSORS))
actual_shape = tuple(model.input_shape)

if (
    len(actual_shape) != 3
    or actual_shape[1] != SEQUENCE_LENGTH
    or actual_shape[2] != len(MODEL_SENSORS)
):
    st.error(
        f"Model input shape is {actual_shape}; expected "
        f"(None, {SEQUENCE_LENGTH}, {len(MODEL_SENSORS)}). "
        "Check the trained model before proceeding."
    )
    st.stop()

if (
    hasattr(scaler, "n_features_in_")
    and scaler.n_features_in_ != len(MODEL_SENSORS)
):
    st.error(
        f"The scaler expects {scaler.n_features_in_} features, "
        f"but the dashboard uses {len(MODEL_SENSORS)}. "
        "Use the matching GRU scaler."
    )
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Dashboard Settings")
st.sidebar.write("Model: GRU")
st.sidebar.write(f"Sequence length: {SEQUENCE_LENGTH} cycles")
st.sidebar.write(f"Sensor features: {len(MODEL_SENSORS)}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Health thresholds")
st.sidebar.caption(
    "These are configurable demonstration thresholds, "
    "not validated maintenance limits."
)

critical_threshold = st.sidebar.number_input(
    "Critical: below this RUL",
    min_value=0.0,
    max_value=float(MAX_RUL),
    value=30.0,
)

warning_threshold = st.sidebar.number_input(
    "Warning: below this RUL",
    min_value=0.0,
    max_value=float(MAX_RUL),
    value=60.0,
)

high_risk_threshold = st.sidebar.number_input(
    "High risk: below this RUL",
    min_value=0.0,
    max_value=float(MAX_RUL),
    value=15.0,
)

if high_risk_threshold > critical_threshold:
    st.sidebar.error(
        "High-risk threshold must not exceed the critical threshold."
    )
    st.stop()

if critical_threshold > warning_threshold:
    st.sidebar.error(
        "Critical threshold must not exceed the warning threshold."
    )
    st.stop()


# =========================================================
# FILE UPLOAD
# =========================================================

st.subheader("Upload Engine Sensor Data")

st.write(
    "Upload a CSV containing sensor readings for each engine. "
    "Each engine needs at least 30 chronological cycles."
)

uploaded_file = st.file_uploader(
    "Choose a sensor CSV file",
    type=["csv"],
)

with st.expander("Required CSV format"):
    st.write(
        "Include the 15 sensor columns listed below. "
        "An engine_id column is recommended when the file "
        "contains multiple engines. A cycle column is also "
        "recommended for chronological ordering."
    )

    st.code(
        "engine_id,cycle,"
        + ",".join(MODEL_SENSORS),
        language="text",
    )

    st.caption(
        "The sensor values must use the same units and "
        "preprocessing conventions as the GRU training data."
    )


# =========================================================
# PREPROCESS AND PREDICT
# =========================================================

def predict_engines(data):
    required = set(MODEL_SENSORS)
    missing = sorted(required - set(data.columns))

    if missing:
        raise ValueError(
            "Missing required sensor columns: "
            + ", ".join(missing)
        )

    df = data.copy()

    # Support a single engine if no ID column is supplied.
    if "engine_id" not in df.columns:
        df["engine_id"] = 1

    # Sort cycles when available.
    if "cycle" in df.columns:
        df["cycle"] = pd.to_numeric(
            df["cycle"], errors="raise"
        )
        df = df.sort_values(
            ["engine_id", "cycle"]
        )

    for sensor in MODEL_SENSORS:
        df[sensor] = pd.to_numeric(
            df[sensor], errors="raise"
        )

    if not np.isfinite(
        df[MODEL_SENSORS].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Sensor columns contain missing or infinite values. "
            "Clean the CSV before prediction."
        )

    sequences = []
    engine_ids = []

    for engine_id, engine_data in df.groupby(
        "engine_id", sort=False
    ):
        if len(engine_data) < SEQUENCE_LENGTH:
            continue

        # Use the latest 30 cycles per engine.
        sensor_values = engine_data[
            MODEL_SENSORS
        ].to_numpy(dtype=float)

        latest_window = sensor_values[-SEQUENCE_LENGTH:]
        sequences.append(latest_window)
        engine_ids.append(engine_id)

    if not sequences:
        raise ValueError(
            "No engines have at least 30 valid cycles."
        )

    X = np.asarray(sequences, dtype=float)

    n_engines, n_steps, n_features = X.shape

    # Transform each sensor row using the saved training scaler.
    X_scaled = scaler.transform(
        X.reshape(-1, n_features)
    )

    X_scaled = X_scaled.reshape(
        n_engines, n_steps, n_features
    )

    predictions = model.predict(
        X_scaled,
        verbose=0,
    ).reshape(-1)

    if not np.isfinite(predictions).all():
        raise ValueError(
            "The model returned non-finite predictions."
        )

    # RUL is capped to the configured training target range.
    predictions = np.clip(
        predictions, 0, MAX_RUL
    )

    return pd.DataFrame({
        "engine_id": engine_ids,
        "predicted_rul_cycles": predictions,
    })


def classify_health(rul):
    if rul <= high_risk_threshold:
        return "HIGH RISK"
    if rul <= critical_threshold:
        return "CRITICAL"
    if rul <= warning_threshold:
        return "WARNING"
    return "HEALTHY"


# =========================================================
# DASHBOARD RESULTS
# =========================================================

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)

        with st.expander("Preview uploaded data"):
            st.dataframe(
                raw_df.head(10),
                use_container_width=True,
            )

        if st.button(
            "Predict Remaining Useful Life",
            type="primary",
        ):
            with st.spinner("Analyzing engine sensor sequences..."):
                results = predict_engines(raw_df)

                results["health_status"] = (
                    results["predicted_rul_cycles"]
                    .apply(classify_health)
                )

                results = results.sort_values(
                    "predicted_rul_cycles"
                ).reset_index(drop=True)

                st.session_state["rul_results"] = results

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


# Render saved results after prediction or Streamlit reruns.
if "rul_results" in st.session_state:
    results = st.session_state["rul_results"]

    st.markdown("---")
    st.subheader("Prediction Overview")

    total = len(results)
    average_rul = results["predicted_rul_cycles"].mean()
    minimum_rul = results["predicted_rul_cycles"].min()
    high_risk_count = (
        results["health_status"] == "HIGH RISK"
    ).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Engines Analyzed", total)
    col2.metric("Average RUL", f"{average_rul:.1f} cycles")
    col3.metric("Lowest RUL", f"{minimum_rul:.1f} cycles")
    col4.metric("High-Risk Engines", int(high_risk_count))

    st.subheader("Engine Health Distribution")

    status_counts = (
        results["health_status"]
        .value_counts()
        .rename_axis("Health Status")
        .reset_index(name="Engines")
    )

    fig_status = px.pie(
        status_counts,
        names="Health Status",
        values="Engines",
        hole=0.55,
        title="Distribution by Predicted Health Status",
    )

    st.plotly_chart(
        fig_status,
        use_container_width=True,
    )

    st.subheader("Predicted RUL by Engine")

    fig_rul = px.bar(
        results,
        x="engine_id",
        y="predicted_rul_cycles",
        color="health_status",
        hover_data=["health_status"],
        title="Estimated Remaining Useful Life",
        labels={
            "engine_id": "Engine ID",
            "predicted_rul_cycles": "RUL (cycles)",
            "health_status": "Health Status",
        },
    )

    fig_rul.update_layout(
        xaxis_type="category",
    )

    st.plotly_chart(
        fig_rul,
        use_container_width=True,
    )

    st.subheader("Engine-Level Predictions")

    st.dataframe(
        results.style.format({
            "predicted_rul_cycles": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Predictions CSV",
        data=csv_bytes,
        file_name="gru_rul_predictions.csv",
        mime="text/csv",
        type="primary",
    )

    st.caption(
        "RUL values are model estimates, not guarantees of "
        "remaining operating time. Health labels are based "
        "on the configurable thresholds in the sidebar."
    )
else:
    st.info(
        "Upload a sensor CSV and click 'Predict Remaining "
        "Useful Life' to see your dashboard results."
    )
