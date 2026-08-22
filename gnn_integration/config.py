"""
gnn_integration.config
=======================
Paths for the Part 2 (GNN) route model. Mirrors markov_integration/config.py
so both models read the same raw sources and write into a sibling artifact
directory, keeping the two pipelines independent but symmetric.
"""

import os
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _MODULE_DIR.parent

_model_env = os.getenv("MARKOV_MODEL_DIR")
MODEL_DIR: Path = (
    PROJECT_ROOT / _model_env if _model_env and not Path(_model_env).is_absolute()
    else PROJECT_ROOT / "handoff_package_v2"
    if not _model_env
    else Path(_model_env)
)

# ---------------------------------------------------------------------------
# Raw source files (same sources the Markov baseline was built from)
# ---------------------------------------------------------------------------
ALL_TRANSITIONS_JSON = MODEL_DIR / "all_transitions.json"
ANPR_HITS_CSV = MODEL_DIR / "anpr_hits_final.csv"
CAMERA_NAMES_JSON = MODEL_DIR / "camera_names.json"
GROUND_TRUTH_ROUTES_JSON = MODEL_DIR / "ground_truth_routes.json"

# ---------------------------------------------------------------------------
# GNN artifact directory (trained weights + graph + metrics)
# ---------------------------------------------------------------------------
GNN_ARTIFACT_DIR = MODEL_DIR / "gnn_artifacts"
GNN_WEIGHTS_NPZ = GNN_ARTIFACT_DIR / "gnn_weights.npz"
GNN_NODE_FEATURES_NPY = GNN_ARTIFACT_DIR / "node_features.npy"
GNN_CAMERA_INDEX_JSON = GNN_ARTIFACT_DIR / "camera_index.json"
GNN_TRAIN_GRAPH_JSON = GNN_ARTIFACT_DIR / "train_graph.json"
GNN_METRICS_JSON = GNN_ARTIFACT_DIR / "metrics.json"
GNN_TRAIN_PLATES_JSON = GNN_ARTIFACT_DIR / "train_plates.json"
GNN_TEST_PLATES_JSON = GNN_ARTIFACT_DIR / "test_plates.json"

# ---------------------------------------------------------------------------
# Train/test + model hyperparameters
# ---------------------------------------------------------------------------
TEST_RATIO: float = 0.2
RANDOM_SEED: int = 42
TOP_K: int = int(os.getenv("TOP_K", "3"))

HIDDEN1 = 16          # node-embedding projection width
DECODER_HIDDEN = 32   # decoder MLP hidden width
EPOCHS = 2500
LEARNING_RATE = 0.3
L2_REG = 1e-4
