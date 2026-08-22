import argparse
import sqlite3
from pathlib import Path

import pandas as pd

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

parser = argparse.ArgumentParser(
    description="Import ANPR CSV into SQLite while preserving SUMO sim_time as numeric seconds."
)
parser.add_argument("--csv", required=True, help="Path to anpr_hits_final.csv")
parser.add_argument("--db", default="anpr_demo_fixed.db", help="Output SQLite DB path")
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Replace the target database if it already exists.",
)
args = parser.parse_args()

csv_path = Path(args.csv).expanduser().resolve()
db_path = Path(args.db).expanduser().resolve()

if not csv_path.is_file():
    raise SystemExit(f"CSV not found: {csv_path}")

if db_path.exists() and not args.overwrite:
    raise SystemExit(
        f"Target DB already exists: {db_path}\n"
        "Use a new --db name, or explicitly add --overwrite."
    )

df = pd.read_csv(csv_path)

missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
if missing:
    raise SystemExit(f"Missing required CSV columns: {missing}")

df = df[REQUIRED_COLUMNS].copy()
df["sim_time"] = pd.to_numeric(df["sim_time"], errors="coerce")

print(f"Source CSV: {csv_path}")
print(f"Rows read: {len(df):,}")
print("First 10 sim_time values:", df["sim_time"].head(10).tolist())
print(f"sim_time min: {df['sim_time'].min()}")
print(f"sim_time max: {df['sim_time'].max()}")
print(f"Distinct sim_time: {df['sim_time'].nunique():,}")
print(f"Null/invalid sim_time: {df['sim_time'].isna().sum():,}")

if df["sim_time"].isna().any():
    raise SystemExit("Aborting: sim_time contains non-numeric values.")

if df["sim_time"].nunique() <= 1:
    raise SystemExit("Aborting: sim_time must contain multiple unique values.")

for column in [
    "camera_lon",
    "camera_lat",
    "vehicle_lon",
    "vehicle_lat",
    "speed_mps",
    "is_hub",
]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

if db_path.exists():
    db_path.unlink()

schema = """
CREATE TABLE events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sim_time REAL NOT NULL,
    camera_id TEXT NOT NULL,
    camera_lon REAL,
    camera_lat REAL,
    plate TEXT NOT NULL,
    vehicle_lon REAL,
    vehicle_lat REAL,
    speed_mps REAL,
    camera_name TEXT,
    is_hub INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_events_plate_time ON events (plate, sim_time);
CREATE INDEX idx_events_camera_time ON events (camera_id, sim_time);
"""

insert_sql = """
INSERT INTO events (
    sim_time, camera_id, camera_lon, camera_lat, plate,
    vehicle_lon, vehicle_lat, speed_mps, camera_name, is_hub
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

rows = []
for values in df.itertuples(index=False, name=None):
    row = [None if pd.isna(value) else value for value in values]
    row[0] = float(row[0])
    row[9] = int(row[9] or 0)
    rows.append(tuple(row))

with sqlite3.connect(db_path) as connection:
    connection.executescript(schema)
    connection.executemany(insert_sql, rows)
    connection.commit()

    summary = connection.execute("""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT sim_time) AS distinct_times,
            MIN(sim_time) AS min_time,
            MAX(sim_time) AS max_time,
            SUM(CASE WHEN sim_time IS NULL THEN 1 ELSE 0 END) AS null_times
        FROM events
    """).fetchone()

    print("\nSQLite verification:")
    print(f"Rows: {summary[0]:,}")
    print(f"Distinct sim_time: {summary[1]:,}")
    print(f"Min sim_time: {summary[2]}")
    print(f"Max sim_time: {summary[3]}")
    print(f"Null sim_time: {summary[4]}")

    print("\nFirst 10 events by numeric sim_time:")
    for event in connection.execute("""
        SELECT sim_time, plate, camera_id, camera_name, speed_mps
        FROM events
        ORDER BY sim_time ASC, event_id ASC
        LIMIT 10
    """):
        print(event)

print(f"\nCreated repaired database: {db_path}")
