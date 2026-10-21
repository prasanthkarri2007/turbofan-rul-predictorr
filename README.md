# Turbofan Engine Remaining Useful Life (RUL) Prediction

A machine learning and deep learning project that predicts the **Remaining Useful Life (RUL)** of turbofan engines using NASA's C-MAPSS dataset. The project includes model training, performance comparison, RUL prediction, and an interactive Streamlit dashboard.

## Project Overview

Predictive maintenance helps identify when equipment may fail so maintenance can be planned before a failure occurs. This project uses engine sensor measurements collected over operating cycles to estimate how many useful cycles remain before an engine reaches its end of life.

## Features

- Data preparation and feature engineering
- Machine learning and deep learning model training
- Model performance evaluation using MAE, RMSE, and R²
- Remaining Useful Life prediction for individual engines
- Interactive Streamlit dashboard
- CSV upload and downloadable prediction results
- Visual comparison of model performance

## Models

The repository contains the trained model artifacts for:

- Random Forest
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

The GRU model is used by the current dashboard for RUL predictions.

## Dataset

This project uses the **NASA C-MAPSS FD001 turbofan engine degradation dataset**.

The dataset contains multivariate sensor measurements recorded over successive operating cycles. The training data includes run-to-failure trajectories, while the test data contains truncated trajectories.

Dataset source: [NASA Prognostics Center of Excellence — Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

Please review the dataset's applicable usage and redistribution terms before redistributing the raw data.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Plotly
- Joblib
- Matplotlib

## Project Structure

```text
turbofan-rul-predictor/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── app/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── gru_rul_model.keras
│   ├── gru_scaler.pkl
│   ├── lstm_rul_model.keras
│   ├── lstm_scaler.pkl
│   └── random_forest_rul.pkl
├── notebooks/
└── src/
    ├── evaluate_models.py
    ├── extra_trees_model.py
    ├── feature_engineering.py
    ├── gru_model.py
    ├── load_data.py
    ├── lstm_model.py
    ├── predict_rul.py
    ├── prepare_dashboard_data.py
    ├── prepare_data.py
    └── rul_capped_model.py
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/turbofan-rul-predictor.git
cd turbofan-rul-predictor
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Dashboard

From the project root, execute:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## Using the Dashboard

1. Launch the Streamlit application.
2. Upload a CSV containing engine IDs, operating cycles, and the required sensor measurements.
3. Review the predicted RUL values and engine health indicators.
4. Explore the visualizations and download the prediction results.

Use the sample CSV provided in `data/processed/` to test the dashboard if it matches the application's expected input format.

## Evaluation

The models are evaluated using:

- **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted RUL.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more strongly.
- **R² Score:** Measures how much variation in the target is explained by the model.

Evaluation results depend on the data preparation, train-validation split, and evaluation procedure. Report only metrics produced by the corresponding evaluation scripts, and distinguish validation performance from test-set performance.

## Limitations

- Predictions are estimates and should not be treated as guarantees of actual engine failure timing.
- The test dataset does not provide the true RUL labels directly in its sensor data files.
- Model performance depends on the selected operating conditions, sensor inputs, and preprocessing pipeline.

## Future Improvements

- Deploy the dashboard online
- Improve validation and generalization testing
- Add model monitoring and prediction uncertainty estimates
- Enhance visualizations and user guidance

## Author

**Karri Prasanth**

Computer Science and Engineering — Artificial Intelligence and Machine Learning

---

If you find this project useful, feel free to explore the code and suggest improvements.
