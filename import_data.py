import sqlite3
from pathlib import Path

import pandas as pd

CSV_PATH = Path("dataset.csv")
DB_PATH = Path("anpr_demo.db")

REQUIRED_COLUMNS = [
    "sim_time",
    "camera_id",
    "camera_lon",
    "camera_lat",
    "plate",
    "vehicle_lon",
    "vehicle_lat",
    "speed_mps",
    "camera_name",
    "is_hub",
]


def main():
    if not CSV_PATH.exists():
        print("Error: dataset.csv is missing. Put the dataset in this folder.")
        raise FileNotFoundError(
            "dataset.csv is missing. Put the dataset in this folder."
        )

    df = pd.read_csv(CSV_PATH)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing these columns: {sorted(missing)}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    df["sim_time"] = pd.to_datetime(df["sim_time"], errors="coerce")
    df["camera_lon"] = pd.to_numeric(df["camera_lon"], errors="coerce")
    df["camera_lat"] = pd.to_numeric(df["camera_lat"], errors="coerce")
    df["vehicle_lon"] = pd.to_numeric(df["vehicle_lon"], errors="coerce")
    df["vehicle_lat"] = pd.to_numeric(df["vehicle_lat"], errors="coerce")
    df["speed_mps"] = pd.to_numeric(df["speed_mps"], errors="coerce")

    df = df.dropna(subset=["sim_time", "camera_id", "plate"])

    df["sim_time"] = df["sim_time"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("events", conn, if_exists="replace", index=False)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_plate_time
            ON events (plate, sim_time)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_camera_time
            ON events (camera_id, sim_time)
        """)

    print(f"Done: imported {len(df):,} rows into {DB_PATH}")


if __name__ == "__main__":
    main()