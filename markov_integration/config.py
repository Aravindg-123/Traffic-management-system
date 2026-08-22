"""
markov_integration.config
=========================
Central configuration resolved from pathlib + environment overrides.

Environment variables (PowerShell session):
  ANPR_DB_PATH                 – path to the SQLite database
  MARKOV_MODEL_DIR             – directory containing Markov artifacts
  MAX_TRAJECTORY_GAP_SECONDS   – seconds gap that starts a new segment
  TOP_K                        – default number of top predictions

All paths are relative to the project root, which is derived from
this file's own location (two directories up).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root: markov_integration/ lives directly inside the project root.
# ---------------------------------------------------------------------------
_MODULE_DIR = Path(__file__).resolve().parent   # …/anpr-trajectory-dashboard/markov_integration
PROJECT_ROOT = _MODULE_DIR.parent               # …/anpr-trajectory-dashboard

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
_db_env = os.getenv("ANPR_DB_PATH")
DB_PATH: Path = (
    Path(_db_env) if _db_env and not Path(_db_env).is_absolute()
    else PROJECT_ROOT / "anpr_demo_fixed.db"
    if not _db_env
    else Path(_db_env)
)

# ---------------------------------------------------------------------------
# Model artefacts directory
# ---------------------------------------------------------------------------
_model_env = os.getenv("MARKOV_MODEL_DIR")
MODEL_DIR: Path = (
    PROJECT_ROOT / _model_env if _model_env and not Path(_model_env).is_absolute()
    else PROJECT_ROOT / "handoff_package_v2"
    if not _model_env
    else Path(_model_env)
)

# ---------------------------------------------------------------------------
# Artefact filenames (fixed, inside MODEL_DIR)
# ---------------------------------------------------------------------------
TRANSITION_1ST_CSV = MODEL_DIR / "transition_probs_1st.csv"
TRANSITION_2ND_JSON = MODEL_DIR / "transition_probs_2nd.json"
CAMERA_NAMES_JSON = MODEL_DIR / "camera_names.json"
CAMERA_LOCATIONS_JSON = MODEL_DIR / "camera_locations.json"

# ---------------------------------------------------------------------------
# Trajectory segmentation
# ---------------------------------------------------------------------------
MAX_TRAJECTORY_GAP_SECONDS: int = int(
    os.getenv("MAX_TRAJECTORY_GAP_SECONDS", "900")
)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
TOP_K: int = int(os.getenv("TOP_K", "3"))
