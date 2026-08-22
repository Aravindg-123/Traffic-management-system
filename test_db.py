import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "anpr_demo_fixed.db"

if not DB_PATH.exists():
    raise SystemExit(f"Database not found: {DB_PATH}")

with sqlite3.connect(DB_PATH) as conn:
    result = conn.execute(
        """
        SELECT
            COUNT(*) AS vehicle_count,
            COALESCE(SUM(hit_count), 0) AS raw_event_count
        FROM (
            SELECT plate, COUNT(*) AS hit_count
            FROM events
            GROUP BY plate
            HAVING COUNT(DISTINCT camera_id) = 3
        )
        """
    ).fetchone()

    samples = conn.execute(
        """
        SELECT
            plate,
            COUNT(*) AS raw_event_count,
            GROUP_CONCAT(camera_id, ' -> ') AS observed_camera_hits
        FROM events
        GROUP BY plate
        HAVING COUNT(DISTINCT camera_id) = 2
        ORDER BY raw_event_count DESC, plate ASC
        LIMIT 10
        """
    ).fetchall()

print(f"Database: {DB_PATH}")
print(f"Vehicles with exactly 2 distinct camera locations: {result[0]:,}")
print(f"Raw detection events belonging to those vehicles: {result[1]:,}")
print("\nTop 10 examples by raw event count:")
for plate, raw_events, observed_hits in samples:
    print(f"- {plate}: {raw_events:,} raw events | {observed_hits}")