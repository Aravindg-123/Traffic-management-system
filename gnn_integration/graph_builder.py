"""
gnn_integration.graph_builder
==============================
Builds the directed camera graph and per-node feature matrix used by the
Part 2 GNN route model.

Pipeline
--------
1. Reconstruct per-plate camera trajectories from ``all_transitions.json``
   (same source used for the Markov baseline's 2nd-order table).
2. Split **by plate** (not by row) into train/test, so the same vehicle
   never leaks between the two sets. This is stricter than the existing
   Markov baseline split and keeps the two models' evaluation numbers
   honest and comparable.
3. Build a directed camera graph from TRAIN trajectories only
   (CAM_i -> CAM_j edge weight = observed transition count).
4. Build a per-camera feature matrix (geography, connectivity, traffic,
   hub flag) and a fixed 2-hop mean-aggregation ("message passing")
   operator over the train graph.
"""

import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from gnn_integration.config import (
    ALL_TRANSITIONS_JSON,
    ANPR_HITS_CSV,
    CAMERA_NAMES_JSON,
    RANDOM_SEED,
    TEST_RATIO,
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_transitions() -> List[Dict]:
    with open(ALL_TRANSITIONS_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def load_camera_meta() -> Dict[str, Dict]:
    with open(CAMERA_NAMES_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def load_hits() -> pd.DataFrame:
    return pd.read_csv(ANPR_HITS_CSV)


# ---------------------------------------------------------------------------
# Trajectory reconstruction + plate-level split
# ---------------------------------------------------------------------------

def build_trajectories(transitions: List[Dict]) -> Dict[str, List[str]]:
    """plate -> ordered camera sequence, reconstructed from consecutive edges."""
    by_plate: Dict[str, List[Dict]] = defaultdict(list)
    for t in transitions:
        by_plate[t["plate"]].append(t)

    trajectories: Dict[str, List[str]] = {}
    for plate, trans_list in by_plate.items():
        seq = [trans_list[0]["from"]]
        for t in trans_list:
            seq.append(t["to"])
        trajectories[plate] = seq
    return trajectories


def split_plates(
    trajectories: Dict[str, List[str]],
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[List[str], List[str]]:
    """Plate-level train/test split (prevents any single vehicle's route
    from appearing in both sets)."""
    plates = sorted(trajectories.keys())
    rng = random.Random(seed)
    rng.shuffle(plates)
    split_idx = int((1 - test_ratio) * len(plates))
    return plates[:split_idx], plates[split_idx:]


def build_examples(
    trajectories: Dict[str, List[str]], plates: List[str]
) -> List[Dict]:
    """(prev2, prev1) -> actual examples for the given plate subset.

    prev2 is None for the first hop of a trajectory (no second-order
    context yet available).
    """
    plate_set = set(plates)
    examples = []
    for plate, seq in trajectories.items():
        if plate not in plate_set:
            continue
        for i in range(len(seq) - 1):
            prev1 = seq[i]
            actual = seq[i + 1]
            prev2 = seq[i - 1] if i - 1 >= 0 else None
            examples.append(
                {"plate": plate, "prev2": prev2, "prev1": prev1, "actual": actual}
            )
    return examples


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_train_graph(examples: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Directed adjacency: from_cam -> {to_cam: transition_count}, built
    from TRAIN examples only (no test-plate leakage into graph edges)."""
    graph: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ex in examples:
        graph[ex["prev1"]][ex["actual"]] += 1
    return {k: dict(v) for k, v in graph.items()}


# ---------------------------------------------------------------------------
# Node features + fixed message-passing operator
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "norm_lon",
    "norm_lat",
    "norm_out_degree",
    "norm_in_degree",
    "norm_traffic_count",
    "norm_mean_speed",
    "is_hub",
]


def _minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def build_camera_index(camera_meta: Dict[str, Dict]) -> Tuple[List[str], Dict[str, int]]:
    camera_list = sorted(camera_meta.keys())
    camera_index = {cam: i for i, cam in enumerate(camera_list)}
    return camera_list, camera_index


def build_node_features(
    camera_list: List[str],
    camera_meta: Dict[str, Dict],
    hits_df: pd.DataFrame,
    train_graph: Dict[str, Dict[str, int]],
) -> np.ndarray:
    """Returns an (N_cameras x F) raw feature matrix, in camera_list order."""
    n = len(camera_list)

    lons = np.array([camera_meta[c]["lon"] for c in camera_list])
    lats = np.array([camera_meta[c]["lat"] for c in camera_list])
    is_hub = np.array([float(camera_meta[c].get("is_hub", False)) for c in camera_list])

    out_deg = np.array(
        [sum(train_graph.get(c, {}).values()) for c in camera_list], dtype=float
    )
    in_deg = np.zeros(n)
    for src, targets in train_graph.items():
        if src not in camera_meta:
            continue
        for dst, cnt in targets.items():
            if dst in camera_meta:
                in_deg[camera_list.index(dst)] += cnt

    mean_speed = np.zeros(n)
    if "camera_id" in hits_df.columns and "speed_mps" in hits_df.columns:
        speed_by_cam = hits_df.groupby("camera_id")["speed_mps"].mean()
        mean_speed = np.array(
            [speed_by_cam.get(c, 0.0) for c in camera_list], dtype=float
        )

    traffic_count = out_deg + in_deg

    X = np.stack(
        [
            _minmax(lons),
            _minmax(lats),
            _minmax(out_deg),
            _minmax(in_deg),
            _minmax(traffic_count),
            _minmax(mean_speed),
            is_hub,
        ],
        axis=1,
    )
    return X


def build_smoothing_operator(
    camera_list: List[str], train_graph: Dict[str, Dict[str, int]]
) -> np.ndarray:
    """Row-normalised (self-loop) adjacency, applied twice, giving a fixed
    2-hop mean "message passing" operator S = A_hat @ A_hat.

    This is the non-trainable propagation step of a Simplified-GCN-style
    model: it smooths each camera's raw features over its 1- and 2-hop
    neighbourhood in the *train* graph before the trainable projection.
    """
    n = len(camera_list)
    idx = {c: i for i, c in enumerate(camera_list)}
    A = np.eye(n)  # self-loops
    for src, targets in train_graph.items():
        if src not in idx:
            continue
        i = idx[src]
        for dst, cnt in targets.items():
            if dst in idx:
                A[i, idx[dst]] += cnt

    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    A_hat = A / row_sums
    return A_hat @ A_hat
