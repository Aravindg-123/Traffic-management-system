"""
gnn_integration.torch_gnn_adapter
===================================
Wraps the pre-built PyTorch Geometric GraphSAGE model from
``handoff_package_v2/trajectory_project/`` (graph_builder.py, predictor.py,
utils.py, scorer.py, gnn.pt, head.pt) for use as the Part 2 model in the
dashboard.

Per explicit instruction: nothing inside trajectory_project/ is modified.
This module only imports those files as a library and adapts their
existing ``predict_next`` output into the same response shape used by
``markov_integration.markov_adapter.MarkovAdapter``, so the frontend can
treat both models interchangeably.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from gnn_integration.config import MODEL_DIR
from markov_integration.markov_adapter import MarkovAdapter  # reuse camera metadata loader

TRAJECTORY_PROJECT_DIR = MODEL_DIR / "trajectory_project"


class TorchGnnNotAvailableError(RuntimeError):
    """Raised when trajectory_project/ or its trained weights aren't present."""


class TorchGnnAdapter:
    def __init__(self) -> None:
        if not TRAJECTORY_PROJECT_DIR.exists():
            raise TorchGnnNotAvailableError(
                f"{TRAJECTORY_PROJECT_DIR} not found."
            )
        gnn_path = TRAJECTORY_PROJECT_DIR / "gnn.pt"
        head_path = TRAJECTORY_PROJECT_DIR / "head.pt"
        if not gnn_path.exists() or not head_path.exists():
            raise TorchGnnNotAvailableError(
                f"Trained weights not found in {TRAJECTORY_PROJECT_DIR} "
                "(expected gnn.pt and head.pt)."
            )

        # trajectory_project's modules use bare `import utils` / `from predictor
        # import ...` style relative imports, so its directory must be on sys.path.
        sys_path_entry = str(TRAJECTORY_PROJECT_DIR)
        if sys_path_entry not in sys.path:
            sys.path.insert(0, sys_path_entry)

        import utils  # type: ignore
        import graph_builder as tp_graph_builder  # type: ignore
        import predictor as tp_predictor  # type: ignore

        self._predict_next = tp_predictor.predict_next

        hits, cam_loc, gt_routes = utils.load_data(str(MODEL_DIR))
        trajs = utils.build_trajectories(hits)

        self._G = tp_graph_builder.build_graph(hits, cam_loc)
        self._data, self._node_idx, self._nodes = tp_graph_builder.graph_to_pyg(self._G)

        self._gnn = tp_predictor.CameraGNN(in_dim=3, hidden=32, out_dim=16)
        self._head = tp_predictor.NextNodePredictor(emb_dim=16)
        self._gnn.load_state_dict(torch.load(gnn_path, map_location="cpu"))
        self._head.load_state_dict(torch.load(head_path, map_location="cpu"))
        self._gnn.eval()
        self._head.eval()

        # Stats needed by predict_next(); computed over ALL trajectories so
        # every camera the demo can select has usable order-2/time/speed context.
        _, self._order2_counts, self._time_stats, self._speed_stats = (
            tp_predictor.build_training_samples(trajs, self._G, self._node_idx)
        )

        # Per-camera mean speed, used as a fallback when no real speed_mps
        # is available for a rolled-out (predicted, not observed) camera.
        self._mean_speed_by_cam: Dict[str, float] = (
            hits.groupby("camera_id")["speed_mps"].mean().to_dict()
        )

        self._cam_meta = MarkovAdapter()._cam_meta

    # ------------------------------------------------------------------

    def _enrich(self, cam_id: str, prob: float) -> Dict[str, Any]:
        meta = self._cam_meta.get(cam_id, {})
        return {
            "camera_id": cam_id,
            "camera_name": meta.get("camera_name", cam_id),
            "latitude": meta.get("latitude"),
            "longitude": meta.get("longitude"),
            "is_hub": meta.get("is_hub", False),
            "probability": round(float(prob), 6),
        }

    def predict(
        self,
        history: List[str],
        top_k: int = 3,
        speed_mps: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not history:
            return self._invalid_response(history, "Empty history provided; cannot predict.")

        valid_history = [c for c in history if c in self._node_idx]
        if not valid_history:
            return self._invalid_response(
                history,
                f"None of the provided cameras {history} are in the trained "
                f"camera graph ({len(self._nodes)} cameras).",
            )

        curr = valid_history[-1]
        prev = valid_history[-2] if len(valid_history) >= 2 else None
        speed = speed_mps if speed_mps is not None else self._mean_speed_by_cam.get(curr, 8.0)

        top1, probs = self._predict_next(
            self._gnn, self._head, self._data, self._node_idx, self._G,
            curr, speed, self._time_stats, self._speed_stats, self._order2_counts,
            prev_cam=prev,
        )

        if top1 is None:
            return self._invalid_response(
                history, f"Camera '{curr}' has no outgoing edges in the trained graph."
            )

        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top_predictions = [self._enrich(cam, p) for cam, p in ranked[:top_k]]
        prob_dist = {cam: round(float(p), 6) for cam, p in ranked}

        state_desc = f"{prev} -> {curr}" if prev else curr
        explanation = (
            f"2-layer GraphSAGE (SAGEConv) node embeddings for '{curr}', decoded by a head "
            f"that combines learned edge features (transition count, 2nd-order count, travel "
            f"time, speed) with the embeddings, over its {len(probs)} legal outgoing "
            f"neighbours, conditioned on history [{state_desc}]."
        )

        return {
            "input_valid": True,
            "history": history,
            "model_order_used": "gnn_sage",
            "second_order_state_seen": f"{prev}_{curr}" if prev else None,
            "top_predictions": top_predictions,
            "probability_distribution": prob_dist,
            "explanation": explanation,
        }

    def roll_out_future_path(
        self, history: List[str], steps: int = 2, top_k: int = 3,
        speed_mps: Optional[float] = None,
    ) -> List[str]:
        working = list(history)
        path: List[str] = []
        for _ in range(steps):
            result = self.predict(working, top_k=top_k, speed_mps=speed_mps)
            if not result["input_valid"] or not result["top_predictions"]:
                break
            next_cam = result["top_predictions"][0]["camera_id"]
            path.append(next_cam)
            working.append(next_cam)
            speed_mps = None  # subsequent hops fall back to per-camera mean speed
        return path

    def _invalid_response(self, history: List[str], reason: str) -> Dict[str, Any]:
        return {
            "input_valid": False,
            "history": history,
            "model_order_used": None,
            "second_order_state_seen": None,
            "top_predictions": [],
            "probability_distribution": {},
            "explanation": reason,
        }
