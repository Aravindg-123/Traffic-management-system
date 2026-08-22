# MARKOV SQLITE INTEGRATION

Read-only Python integration that accepts a licence-plate number,
constructs the plate's **latest valid camera-history segment** from the
`anpr_demo_fixed.db` SQLite database, and returns a Markov next-camera
prediction.

---

## Quick Start (Windows PowerShell)

```powershell
# From the project root
cd C:\Users\aravi\OneDrive\Desktop\proj\sih26\anpr-trajectory-dashboard

# Run the full demo suite (auto-discovers plates)
.\.venv\Scripts\python.exe .\demo_markov_from_sqlite.py

# Query a specific plate
.\.venv\Scripts\python.exe .\demo_markov_from_sqlite.py --plate TN03QH4950
```

---

## Architecture

```
anpr_demo_fixed.db
       │
       ▼
markov_integration/db.py          ← read-only sqlite3, get_plate_events()
       │
       ▼
markov_integration/trajectory_service.py
  ┌─────────────────────────────────────────────┐
  │  1. Fetch ALL raw events for plate           │
  │  2. Validate sim_time (numeric, SUMO secs)   │
  │  3. Split into 900-second segments           │
  │  4. Take the LATEST segment only             │
  │  5. Collapse consecutive duplicate cameras   │
  └─────────────────────────────────────────────┘
       │
       ▼
markov_integration/markov_adapter.py
  ┌─────────────────────────────────────────────┐
  │  Loaded once at startup:                     │
  │    transition_probs_1st.csv  (30×30 matrix)  │
  │    transition_probs_2nd.json (bigram dict)   │
  │    camera_names.json                         │
  │    camera_locations.json                     │
  │                                              │
  │  If last two cameras form a known 2nd-order  │
  │  state → use 2nd-order distribution.         │
  │  Otherwise → 1st-order from latest camera.  │
  │                                              │
  │  Returns top-k predictions + full dist.     │
  └─────────────────────────────────────────────┘
```

---

## Key Design Decisions

### sim_time is SUMO simulation seconds — not calendar time

`sim_time` is a raw numeric float produced by the SUMO traffic simulator.
It represents seconds elapsed since the start of the simulation.
It is **never** converted to a `datetime` object or treated as a Unix
timestamp.  Use it only for chronological ordering and for computing gaps
between events (e.g., 900-second segment threshold).

### Zero-speed events are retained

Consecutive repeated camera hits, including hits with `speed_mps = 0.0`,
are **retained in the database** and counted in `raw_event_count` and
`active_segment_event_count`.  These records carry value for congestion
analytics (a vehicle parked or idling at a camera is a real observation).

However, **consecutive duplicate camera IDs are collapsed** before the
Markov model sees the history.  Route prediction does not benefit from
knowing a vehicle visited the same camera three times in a row; the
Markov model needs distinct transitions.

Example:
```
Raw DB events   : CAM_01 → CAM_01 → CAM_01 → CAM_11 → CAM_02
Collapsed input : CAM_01 → CAM_11 → CAM_02
```

### 900-second segment gap

If the time gap between two consecutive events for the same plate exceeds
900 SUMO simulation seconds, a new trajectory segment begins.  Only the
**latest segment** is used for prediction — older trips are irrelevant to
where the vehicle is heading now.

---

## File Structure

```
anpr-trajectory-dashboard/
├── anpr_demo_fixed.db          ← source DB (READ-ONLY, never modified)
├── handoff_package_v2/         ← Markov artefacts (READ-ONLY, never modified)
│   ├── transition_probs_1st.csv
│   ├── transition_probs_2nd.json
│   ├── camera_names.json
│   └── camera_locations.json
├── markov_integration/
│   ├── __init__.py             ← exposes TrajectoryPredictionService
│   ├── config.py               ← paths + env overrides
│   ├── db.py                   ← sqlite3 read-only layer
│   ├── markov_adapter.py       ← Markov model loader + predictor
│   └── trajectory_service.py  ← orchestration
├── demo_markov_from_sqlite.py  ← runnable demo + test suite
├── .env.example                ← environment variable template
└── MARKOV_SQLITE_INTEGRATION.md
```

---

## Environment Variable Overrides

Set in your PowerShell session **before** running any script:

```powershell
$env:ANPR_DB_PATH                = "anpr_demo_fixed.db"
$env:MARKOV_MODEL_DIR            = "handoff_package_v2"
$env:MAX_TRAJECTORY_GAP_SECONDS  = "900"
$env:TOP_K                       = "3"
```

All paths are relative to the project root unless you supply an absolute path.
These variables are read with `os.getenv()` only — no `python-dotenv` dependency.

---

## Python API

```python
from markov_integration import TrajectoryPredictionService

svc = TrajectoryPredictionService()
result = svc.predict_for_plate("TN03QH4950", top_k=3)
```

### Response fields

| Field | Type | Description |
|---|---|---|
| `plate` | str | Input plate |
| `status` | str | See status values below |
| `latest_sim_time` | float \| None | sim_time of last event in active segment |
| `active_history` | list[str] | Deduplicated camera IDs for latest segment |
| `raw_event_count` | int | Total events for this plate in DB |
| `active_segment_event_count` | int | Events in the latest segment (pre-dedup) |
| `unique_camera_count` | int | Distinct cameras in latest segment |
| `max_gap_seconds` | int | Configured gap threshold |
| `markov_result` | dict \| None | Nested Markov prediction (see below) |

### Status values

| Status | Meaning |
|---|---|
| `plate_not_found` | Plate has no events in DB |
| `invalid_sim_time` | One or more sim_time values are non-numeric |
| `prediction_ready` | markov_result is populated (may itself be invalid if history has no known cameras) |

### markov_result fields

| Field | Type | Description |
|---|---|---|
| `input_valid` | bool | False if history is empty or has no known cameras |
| `history` | list[str] | The deduplicated camera history that was passed in |
| `model_order_used` | int \| None | 1 or 2 |
| `second_order_state_seen` | str \| None | e.g. `"CAM_11_CAM_02"` |
| `top_predictions` | list[dict] | Up to top_k predictions |
| `probability_distribution` | dict | Full {camera_id: probability} |
| `explanation` | str | Human-readable explanation of the prediction |

### top_predictions entry fields

| Field | Type | Description |
|---|---|---|
| `camera_id` | str | e.g. `"CAM_22"` |
| `camera_name` | str | Human-readable name e.g. `"Adyar"` |
| `latitude` | float | Camera latitude |
| `longitude` | float | Camera longitude |
| `is_hub` | bool | Whether this is a major hub camera |
| `probability` | float | Transition probability |

---

## Baseline Accuracy (offline evaluation)

These figures were computed over the full dataset in `handoff_package_v2/`:

| Model order | Top-1 accuracy | Top-3 accuracy |
|---|---|---|
| 1st-order Markov | **24.8 %** | **46.8 %** |
| 2nd-order Markov | **25.4 %** | **46.6 %** |

The model predicts the **next observed camera**, not the final destination.

---

## Compile Check

```powershell
.\.venv\Scripts\python.exe -m py_compile markov_integration\__init__.py `
    markov_integration\config.py `
    markov_integration\db.py `
    markov_integration\markov_adapter.py `
    markov_integration\trajectory_service.py `
    demo_markov_from_sqlite.py
echo "All files compile OK"
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: anpr_demo_fixed.db` | Wrong CWD or env override points to wrong path | Run from project root; check `$env:ANPR_DB_PATH` |
| `FileNotFoundError: transition_probs_1st.csv` | Model dir not found | Check `$env:MARKOV_MODEL_DIR`; default is `handoff_package_v2` |
| `status: plate_not_found` | Plate not in DB | Check spelling; plate is case-sensitive |
| `input_valid: false` | All cameras in history are unknown | History contains cameras not in the 30-camera set |
| Probability sum assertion fails | Floating-point drift beyond 1e-6 | Should not occur; `_normalise()` corrects drift up to 0.01 |

---

## Constraints (DO NOT violate)

- **Never modify** `anpr_demo_fixed.db` or `anpr_demo.db`.
- **Never modify, move, or regenerate** anything under `handoff_package_v2/`.
- **Never** convert `sim_time` to a `datetime` object.
- **Never** filter or delete zero-speed events from the DB.
- **No new dependencies** — only Python standard library is used.
- **No absolute paths** hardcoded in source files.
