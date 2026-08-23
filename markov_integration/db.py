"""
markov_integration.db
=====================
Read-only SQLite access layer.

Only Python built-in ``sqlite3`` is used.  The connection is opened in
immutable / read-only URI mode so neither file is ever modified.

Public API
----------
get_plate_events(plate: str) -> list[dict]
    Returns all events for *plate* from the ``events`` table, ordered by
    (sim_time ASC, event_id ASC).  Returns an empty list for an unknown plate.
"""

import sqlite3
from typing import List, Dict, Any

from markov_integration.config import DB_PATH

# ---------------------------------------------------------------------------
# Column constants (matches schema confirmed via PRAGMA table_info)
# ---------------------------------------------------------------------------
_COLUMNS = (
    "event_id",
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
)

_SELECT_SQL = """
    SELECT
        rowid AS event_id,
        sim_time,
        camera_id,
        camera_lon,
        camera_lat,
        plate,
        vehicle_lon,
        vehicle_lat,
        speed_mps,
        camera_name,
        is_hub
    FROM events
    WHERE plate = ?
    ORDER BY sim_time ASC, event_id ASC
"""


def _open_readonly() -> sqlite3.Connection:
    """Open *anpr_demo_fixed.db* in read-only mode via URI."""
    uri = DB_PATH.as_posix()
    # Convert to file URI for read-only mode
    file_uri = f"file:{uri}?mode=ro"
    conn = sqlite3.connect(file_uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_plate_events(plate: str) -> List[Dict[str, Any]]:
    """
    Return all events for *plate*, ordered by (sim_time ASC, event_id ASC).

    Parameters
    ----------
    plate:
        Exact licence-plate string (case-sensitive as stored in DB).

    Returns
    -------
    list[dict]
        One dict per event row.  Empty list if the plate is not found.
        ``sim_time`` is a raw SUMO simulation float; it is **not** a
        calendar timestamp and must never be treated as one.
    """
    conn = _open_readonly()
    try:
        cur = conn.cursor()
        cur.execute(_SELECT_SQL, (plate,))
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_db_path() -> str:
    """Return the resolved absolute DB path as a string (for diagnostics)."""
    return str(DB_PATH.resolve())


def get_schema_info() -> List[Dict[str, Any]]:
    """Return PRAGMA table_info for the *events* table (diagnostics only)."""
    conn = _open_readonly()
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(events)")
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
