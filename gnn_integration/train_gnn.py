"""
gnn_integration.train_gnn
==========================
Offline training + evaluation script for the Part 2 GNN route model.

Usage
-----
    python -m gnn_integration.train_gnn

Produces (under handoff_package_v2/gnn_artifacts/):
    gnn_weights.npz     trained W1/b1/Wd1/bd1/Wd2/bd2
    camera_index.json   camera_id -> row index (fixes embedding ordering)
    node_features.npy   raw (N x F) node feature matrix
    train_graph.json    directed adjacency built from TRAIN plates only
    train_plates.json / test_plates.json
    metrics.json        top-1 / top-3 / MRR on the held-out TEST plates,
                         computed the same way as the Markov baseline's
                         evaluate_markov.py so the two are comparable.

Non-negotiable rule this script enforces: train_graph, node feature
normalisation stats, and model weights are all derived from TRAIN plates
only. Test plates are used purely for scoring.
"""

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np

from gnn_integration.config import (
    GNN_ARTIFACT_DIR,
    GNN_WEIGHTS_NPZ,
    GNN_NODE_FEATURES_NPY,
    GNN_CAMERA_INDEX_JSON,
    GNN_TRAIN_GRAPH_JSON,
    GNN_METRICS_JSON,
    GNN_TRAIN_PLATES_JSON,
    GNN_TEST_PLATES_JSON,
    TEST_RATIO,
    RANDOM_SEED,
    HIDDEN1,
    DECODER_HIDDEN,
    EPOCHS,
    LEARNING_RATE,
    L2_REG,
    TOP_K,
)
from gnn_integration.graph_builder import (
    load_transitions,
    load_camera_meta,
    load_hits,
    build_trajectories,
    split_plates,
    build_examples,
    build_train_graph,
    build_camera_index,
    build_node_features,
    build_smoothing_operator,
    FEATURE_NAMES,
)
from gnn_integration.gnn_model import SimpleGNN


def _speed_lookup(camera_list, X, feature_names):
    speed_idx = feature_names.index("norm_mean_speed")
    return {cam: float(X[i, speed_idx]) for i, cam in enumerate(camera_list)}


def evaluate(model: SimpleGNN, test_examples, train_graph, speed_lookup, top_k=TOP_K):
    """Top-1 / Top-3 / MRR on held-out plates. Unseen `curr` (no train-graph
    out-edges) falls back to a uniform prediction over all cameras, mirroring
    the Markov baseline's behaviour for sparse contexts."""
    n = 0
    top1_hits = 0
    top3_hits = 0
    rr_sum = 0.0
    all_cameras = model.camera_list

    for ex in test_examples:
        curr, prev, actual = ex["prev1"], ex["prev2"], ex["actual"]
        if curr not in model.camera_index:
            continue
        candidates = sorted(train_graph.get(curr, {}).keys())
        if not candidates:
            candidates = all_cameras

        speed_norm = speed_lookup.get(curr, 0.5)
        probs = model.score_candidates(curr, prev, candidates, speed_norm, dt_norm=0.5)
        ranked = sorted(zip(candidates, probs), key=lambda kv: kv[1], reverse=True)
        ranked_cams = [c for c, _ in ranked]

        n += 1
        if actual in ranked_cams:
            rank = ranked_cams.index(actual) + 1
            rr_sum += 1.0 / rank
            if rank == 1:
                top1_hits += 1
            if rank <= min(top_k, len(ranked_cams)):
                top3_hits += 1
        # actual outside candidate set entirely -> counts as a miss (rr=0)

    if n == 0:
        return {"n_test": 0, "top1_accuracy": None, "top3_accuracy": None, "mrr": None}

    return {
        "n_test": n,
        "top1_accuracy": top1_hits / n,
        "top3_accuracy": top3_hits / n,
        "mrr": rr_sum / n,
    }


def main():
    GNN_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw sources...")
    transitions = load_transitions()
    camera_meta = load_camera_meta()
    hits_df = load_hits()

    trajectories = build_trajectories(transitions)
    train_plates, test_plates = split_plates(trajectories, TEST_RATIO, RANDOM_SEED)
    print(f"Plates: {len(trajectories)} total -> {len(train_plates)} train / {len(test_plates)} test")

    train_examples = build_examples(trajectories, train_plates)
    test_examples = build_examples(trajectories, test_plates)
    print(f"Examples: {len(train_examples)} train, {len(test_examples)} test")

    train_graph = build_train_graph(train_examples)
    camera_list, camera_index = build_camera_index(camera_meta)
    X = build_node_features(camera_list, camera_meta, hits_df, train_graph)
    S = build_smoothing_operator(camera_list, train_graph)

    model = SimpleGNN(
        camera_list, X, S,
        hidden1=HIDDEN1, decoder_hidden=DECODER_HIDDEN, seed=RANDOM_SEED,
    )

    speed_lookup = _speed_lookup(camera_list, X, FEATURE_NAMES)

    print("Training GNN (full-batch numpy backprop)...")
    losses = model.train(
        train_examples, train_graph,
        epochs=EPOCHS, lr=LEARNING_RATE, l2=L2_REG,
        speed_lookup=speed_lookup, verbose_every=50,
    )
    print(f"Final training loss: {losses[-1]:.4f}")

    print("Evaluating on held-out TEST plates...")
    metrics = evaluate(model, test_examples, train_graph, speed_lookup, top_k=TOP_K)
    metrics["n_train"] = len(train_examples)
    metrics["top_k"] = TOP_K
    metrics["random_seed"] = RANDOM_SEED
    print(json.dumps(metrics, indent=2))

    # ---- Save artifacts ---------------------------------------------------
    model.save_weights(GNN_WEIGHTS_NPZ)
    np.save(GNN_NODE_FEATURES_NPY, X)
    with open(GNN_CAMERA_INDEX_JSON, "w", encoding="utf-8") as fh:
        json.dump(camera_index, fh, indent=2)
    with open(GNN_TRAIN_GRAPH_JSON, "w", encoding="utf-8") as fh:
        json.dump(train_graph, fh, indent=2)
    with open(GNN_METRICS_JSON, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    with open(GNN_TRAIN_PLATES_JSON, "w", encoding="utf-8") as fh:
        json.dump(train_plates, fh, indent=2)
    with open(GNN_TEST_PLATES_JSON, "w", encoding="utf-8") as fh:
        json.dump(test_plates, fh, indent=2)

    print(f"\nSaved artifacts to {GNN_ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
