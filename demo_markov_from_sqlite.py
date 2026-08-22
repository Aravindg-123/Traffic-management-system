# -*- coding: utf-8 -*-
"""
demo_markov_from_sqlite.py
==========================
Demonstration script: SQLite → Markov trajectory prediction.

Usage
-----
  python .\\demo_markov_from_sqlite.py
  python .\\demo_markov_from_sqlite.py --plate TN03QH4950

Tests run automatically
-----------------------
  1. A valid plate with a multi-camera latest segment (2nd-order if found).
  2. UNKNOWN_PLATE_XX0000   → plate_not_found response.
  3. A single-camera plate  → prediction with only 1-camera history.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# ── Ensure the project root is importable regardless of CWD ────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

# Fix Windows console Unicode issues before any print
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from markov_integration.config import (
    DB_PATH,
    MODEL_DIR,
    TRANSITION_1ST_CSV,
    TRANSITION_2ND_JSON,
    CAMERA_NAMES_JSON,
    CAMERA_LOCATIONS_JSON,
    MAX_TRAJECTORY_GAP_SECONDS,
    TOP_K,
)
from markov_integration.db import get_db_path, get_schema_info
from markov_integration.markov_adapter import MarkovAdapter
from markov_integration.trajectory_service import (
    TrajectoryPredictionService,
    _split_into_segments,
    _collapse_consecutive_cameras,
)

# ── Helpers ─────────────────────────────────────────────────────────────────

SEPARATOR = "-" * 70


def _print_sep(title: str = "") -> None:
    if title:
        pad = max(0, 68 - len(title))
        print(f"\n-- {title} {'-' * pad}")
    else:
        print(SEPARATOR)


def _pjson(obj: object) -> None:
    """Pretty-print a JSON-serialisable object."""
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _assert_prob_sum(result: dict, label: str) -> None:
    """Assert that the probability distribution sums to 1 ± 1e-4.

    Tolerance is 1e-4 (not 1e-6) because the 1st-order CSV stores 3–4
    decimal places; after renormalisation, floating-point summation of
    30 values cannot guarantee sub-1e-6 accuracy.
    """
    mr = result.get("markov_result") or {}
    dist = mr.get("probability_distribution", {})
    if not dist:
        print(f"  [SKIP] {label}: no probability distribution to check.")
        return
    total = sum(dist.values())
    assert abs(total - 1.0) <= 1e-4, (
        f"[FAIL] {label}: probability sum = {total:.8f} (expected 1.0 +/- 1e-4)"
    )
    print(f"  [OK]  {label}: probability distribution sums to {total:.8f}")


# ── Path and DB diagnostics ──────────────────────────────────────────────────

def print_diagnostics() -> None:
    _print_sep("RESOLVED PATHS & CONFIGURATION")
    print(f"  DB path       : {DB_PATH.resolve()}")
    print(f"  DB exists     : {DB_PATH.exists()}")
    print(f"  Model dir     : {MODEL_DIR.resolve()}")
    print(f"  1st-order CSV : {TRANSITION_1ST_CSV.name}  exists={TRANSITION_1ST_CSV.exists()}")
    print(f"  2nd-order JSON: {TRANSITION_2ND_JSON.name} exists={TRANSITION_2ND_JSON.exists()}")
    print(f"  camera_names  : {CAMERA_NAMES_JSON.name}   exists={CAMERA_NAMES_JSON.exists()}")
    print(f"  cam_locations : {CAMERA_LOCATIONS_JSON.name} exists={CAMERA_LOCATIONS_JSON.exists()}")
    print(f"  Max gap (s)   : {MAX_TRAJECTORY_GAP_SECONDS}")
    print(f"  Top-k default : {TOP_K}")

    _print_sep("DATABASE SCHEMA — events table")
    schema = get_schema_info()
    for col in schema:
        print(f"  col {col['cid']:>2}: {col['name']:<16} {col['type']}")

    _print_sep("SIM_TIME VALIDATION")
    # Quick numeric validation on sim_time
    conn_path = str(DB_PATH.resolve())
    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM events")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM events WHERE typeof(sim_time) NOT IN ('integer','real')")
    bad = cur.fetchone()[0]
    cur.execute("SELECT MIN(sim_time), MAX(sim_time) FROM events")
    mn, mx = cur.fetchone()
    conn.close()
    print(f"  Total events  : {total}")
    print(f"  Non-numeric sim_time rows: {bad}  (expected 0)")
    print(f"  sim_time range: {mn} → {mx} SUMO simulation seconds")
    print(f"  NOTE: sim_time is raw SUMO sim time, NOT calendar/wall-clock time.")


# ── Plate discovery ──────────────────────────────────────────────────────────

def _load_second_order_states() -> set:
    import json
    with open(TRANSITION_2ND_JSON, encoding="utf-8") as f:
        return set(json.load(f).keys())


def discover_valid_plate(require_second_order: bool = True) -> dict:
    """
    Scan the DB and return the first plate whose latest segment has
    ≥2 distinct collapsed cameras and (optionally) a 2nd-order state.

    Returns dict with keys: plate, history, has_second_order.
    """
    second_order_states = _load_second_order_states() if require_second_order else set()

    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT plate FROM events ORDER BY plate")
    plates = [r["plate"] for r in cur.fetchall()]

    for plate in plates:
        cur.execute(
            "SELECT sim_time, camera_id FROM events "
            "WHERE plate=? ORDER BY sim_time ASC, event_id ASC",
            (plate,),
        )
        evts = [dict(r) for r in cur.fetchall()]
        if len(evts) < 2:
            continue
        segs = _split_into_segments(evts, MAX_TRAJECTORY_GAP_SECONDS)
        latest = segs[-1]
        history = _collapse_consecutive_cameras(latest)
        if len(history) < 2:
            continue
        if require_second_order:
            state = f"{history[-2]}_{history[-1]}"
            if state in second_order_states:
                conn.close()
                return {"plate": plate, "history": history, "has_second_order": True}
        else:
            conn.close()
            return {"plate": plate, "history": history, "has_second_order": False}

    conn.close()
    return {}


def discover_single_camera_plate() -> str:
    """
    Return the first plate whose latest segment collapses to exactly 1 camera.
    """
    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT plate FROM events ORDER BY plate")
    plates = [r["plate"] for r in cur.fetchall()]

    for plate in plates:
        cur.execute(
            "SELECT sim_time, camera_id FROM events "
            "WHERE plate=? ORDER BY sim_time ASC, event_id ASC",
            (plate,),
        )
        evts = [dict(r) for r in cur.fetchall()]
        segs = _split_into_segments(evts, MAX_TRAJECTORY_GAP_SECONDS)
        latest = segs[-1]
        history = _collapse_consecutive_cameras(latest)
        if len(history) == 1:
            conn.close()
            return plate

    conn.close()
    return ""


# ── Per-test runner ──────────────────────────────────────────────────────────

def run_test(svc: TrajectoryPredictionService, plate: str, label: str) -> dict:
    _print_sep(f"TEST: {label}")
    print(f"  Plate: {plate}")
    result = svc.predict_for_plate(plate, top_k=TOP_K)

    # Summarise (avoid printing raw_event_count lines of events)
    summary = {k: v for k, v in result.items() if k != "markov_result"}
    print("\n  Service summary:")
    _pjson(summary)

    mr = result.get("markov_result")
    if mr:
        print("\n  Markov result:")
        # Truncate probability_distribution to top 5 for readability
        mr_display = {k: v for k, v in mr.items() if k != "probability_distribution"}
        mr_display["probability_distribution"] = dict(
            list(mr.get("probability_distribution", {}).items())[:5]
        )
        mr_display["probability_distribution"]["..."] = "(truncated)"
        _pjson(mr_display)

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo: SQLite → Markov trajectory prediction."
    )
    parser.add_argument(
        "--plate",
        metavar="PLATE",
        default=None,
        help="Run a single plate lookup instead of the full demo suite.",
    )
    args = parser.parse_args()

    print_diagnostics()

    svc = TrajectoryPredictionService()

    if args.plate:
        # ── Single plate mode ────────────────────────────────────────────
        _print_sep(f"DIRECT LOOKUP — {args.plate}")
        result = svc.predict_for_plate(args.plate, top_k=TOP_K)
        _pjson(result)
        _assert_prob_sum(result, args.plate)
        return

    # ── Full demo suite ──────────────────────────────────────────────────
    _print_sep("DISCOVERING TEST PLATES")

    # Test 1: valid multi-camera plate (prefer 2nd-order state)
    info = discover_valid_plate(require_second_order=True)
    if not info:
        print("  No 2nd-order plate found; falling back to any multi-camera plate.")
        info = discover_valid_plate(require_second_order=False)
    if not info:
        print("  ERROR: Could not find any valid multi-camera plate in DB.")
        sys.exit(1)

    test1_plate = info["plate"]
    print(f"  Test-1 plate (multi-camera): {test1_plate}  history={info['history']}")
    print(f"  2nd-order state available  : {info['has_second_order']}")

    # Test 3: single-camera plate
    test3_plate = discover_single_camera_plate()
    if test3_plate:
        print(f"  Test-3 plate (single-cam)  : {test3_plate}")
    else:
        print("  Test-3 plate (single-cam)  : none found, skipping.")

    # ── Run tests ────────────────────────────────────────────────────────
    results = {}

    results["test1"] = run_test(svc, test1_plate, f"Valid multi-camera plate [{test1_plate}]")
    _assert_prob_sum(results["test1"], "Test-1")

    results["test2"] = run_test(svc, "UNKNOWN_PLATE_XX0000", "Unknown plate")
    assert results["test2"]["status"] == "plate_not_found", (
        f"[FAIL] Test-2: expected status='plate_not_found', "
        f"got '{results['test2']['status']}'"
    )
    print("  [OK]  Test-2: status=plate_not_found confirmed.")

    if test3_plate:
        results["test3"] = run_test(svc, test3_plate, f"Single-camera plate [{test3_plate}]")
        _assert_prob_sum(results["test3"], "Test-3")

    # ── Summary ──────────────────────────────────────────────────────────
    _print_sep("SUMMARY")
    t1_mr = results["test1"].get("markov_result", {}) or {}
    top3 = t1_mr.get("top_predictions", [])
    print(f"  Selected plate  : {test1_plate}")
    print(f"  Active history  : {results['test1']['active_history']}")
    print(f"  Model order used: {t1_mr.get('model_order_used')}")
    print(f"  2nd-order state : {t1_mr.get('second_order_state_seen')}")
    print(f"  Top-{TOP_K} predictions:")
    for i, p in enumerate(top3, 1):
        print(
            f"    {i}. {p['camera_id']} ({p['camera_name']})  "
            f"prob={p['probability']:.4f}  hub={p['is_hub']}"
        )

    _print_sep()
    print("All assertions passed.  Demo complete.")


if __name__ == "__main__":
    main()
