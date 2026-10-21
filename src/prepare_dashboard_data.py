
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT / "data" / "raw" / "test_FD001.txt"
OUTPUT_PATH = ROOT / "data" / "processed" / "dashboard_test.csv"

# NASA C-MAPSS FD001 columns
columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

df = pd.read_csv(
    INPUT_PATH,
    sep=r"\s+",
    header=None,
    names=columns,
    engine="python",
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Keep the exact 15 sensors used by the GRU.
model_sensors = [
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

df = df[["engine_id", "cycle"] + model_sensors]
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved dashboard test data to: {OUTPUT_PATH}")
print(f"Engines: {df['engine_id'].nunique()}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
