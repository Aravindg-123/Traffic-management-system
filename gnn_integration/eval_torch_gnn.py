"""
gnn_integration.eval_torch_gnn
================================
Evaluates the already-trained trajectory_project weights (gnn.pt/head.pt)
using trajectory_project's own scorer.py, unmodified, and persists the
result to handoff_package_v2/trajectory_project/metrics.json so the
dashboard can display real numbers instead of hardcoding them.

This does not retrain and does not change anything inside
trajectory_project/ -- it only calls its existing functions the same way
main.py does, then writes a metrics file next to the model weights.
"""

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from gnn_integration.config import MODEL_DIR

TRAJECTORY_PROJECT_DIR = MODEL_DIR / "trajectory_project"


def main():
    sys.path.insert(0, str(TRAJECTORY_PROJECT_DIR))
    import utils
    import graph_builder
    import predictor
    import scorer

    hits, cam_loc, gt_routes = utils.load_data(str(MODEL_DIR))
    trajs = utils.build_trajectories(hits)
    train_trajs, test_trajs = utils.train_test_split_by_vehicle(trajs, test_ratio=0.2)

    G = graph_builder.build_graph(hits, cam_loc)
    data, node_idx, nodes = graph_builder.graph_to_pyg(G)

    gnn = predictor.CameraGNN(in_dim=3, hidden=32, out_dim=16)
    head = predictor.NextNodePredictor(emb_dim=16)
    gnn.load_state_dict(torch.load(TRAJECTORY_PROJECT_DIR / "gnn.pt", map_location="cpu"))
    head.load_state_dict(torch.load(TRAJECTORY_PROJECT_DIR / "head.pt", map_location="cpu"))

    # Mirrors main.py exactly: samples/stats are built from test_trajs (the
    # same data gnn.pt/head.pt were trained on).
    samples, order2_counts, time_stats, speed_stats = predictor.build_training_samples(
        test_trajs, G, node_idx
    )

    metrics = scorer.evaluate_next_node_accuracy(
        gnn, head, data, node_idx, G, test_trajs, time_stats, speed_stats, order2_counts
    )
    baseline = scorer.evaluate_baseline_accuracy(G, test_trajs)

    result = {
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "n_train_vehicles": len(train_trajs),
        "n_test_vehicles": len(test_trajs),
        "n_samples_scored": metrics["num_test_transitions"],
        "top1_accuracy": metrics["top1_accuracy"],
        "top3_accuracy": metrics["top3_accuracy"],
        "baseline_top1_accuracy": baseline["baseline_top1_accuracy"],
        "note": (
            "Computed with trajectory_project's own main.py/scorer.py logic "
            "unmodified, which builds training samples from the same split "
            "used for evaluation (see build_training_samples(test_trajs, ...) "
            "in main.py) -- so this reflects fit-to-that-split performance, "
            "not accuracy on vehicles unseen during training."
        ),
    }
    print(json.dumps(result, indent=2))

    out_path = TRAJECTORY_PROJECT_DIR / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
