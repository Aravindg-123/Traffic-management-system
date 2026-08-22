"""
gnn_integration.gnn_adapter
=============================
Loads the trained Part 2 GNN artefacts once and exposes ``predict()`` with
the *same response shape* as ``markov_integration.markov_adapter.MarkovAdapter``,
so the frontend/service layer can treat both models interchangeably.
"""

import json
from typing import Any, Dict, List, Optional

import numpy as np

from gnn_integration.config import (
    GNN_WEIGHTS_NPZ,
    GNN_NODE_FEATURES_NPY,
    GNN_CAMERA_INDEX_JSON,
    GNN_TRAIN_GRAPH_JSON,
    GNN_METRICS_JSON,
    HIDDEN1,
    DECODER_HIDDEN,
    TOP_K,
)
from gnn_integration.graph_builder import build_smoothing_operator, FEATURE_NAMES
from gnn_integration.gnn_model import SimpleGNN
from markov_integration.markov_adapter import MarkovAdapter  # reuse camera metadata loader


class GnnNotTrainedError(RuntimeError):
    """Raised when artefacts don't exist yet (run `python -m gnn_integration.train_gnn`)."""


class GnnAdapter:
    def __init__(self) -> None:
        if not GNN_WEIGHTS_NPZ.exists():
            raise GnnNotTrainedError(
                f"GNN artefacts not found at {GNN_WEIGHTS_NPZ}. "
                "Run: python -m gnn_integration.train_gnn"
            )

        with open(GNN_CAMERA_INDEX_JSON, encoding="utf-8") as fh:
            camera_index: Dict[str, int] = json.load(fh)
        self._camera_list: List[str] = sorted(camera_index, key=lambda c: camera_index[c])

        with open(GNN_TRAIN_GRAPH_JSON, encoding="utf-8") as fh:
            self._train_graph: Dict[str, Dict[str, int]] = json.load(fh)

        X = np.load(GNN_NODE_FEATURES_NPY)
        S = build_smoothing_operator(self._camera_list, self._train_graph)

        self._model = SimpleGNN(
            self._camera_list, X, S,
            hidden1=HIDDEN1, decoder_hidden=DECODER_HIDDEN,
        )
        self._model.load_weights(GNN_WEIGHTS_NPZ)

        speed_idx = FEATURE_NAMES.index("norm_mean_speed")
        self._speed_lookup = {
            cam: float(X[i, speed_idx]) for i, cam in enumerate(self._camera_list)
        }

        # Reuse the Markov adapter's camera metadata loader for names/lat/lon/hub
        # (both models describe the same physical camera network).
        self._cam_meta = MarkovAdapter()._cam_meta

        self.metrics: Dict[str, Any] = {}
        if GNN_METRICS_JSON.exists():
            with open(GNN_METRICS_JSON, encoding="utf-8") as fh:
                self.metrics = json.load(fh)

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
        self, history: List[str], top_k: int = TOP_K, speed_mps: Optional[float] = None
    ) -> Dict[str, Any]:
        """Mirrors MarkovAdapter.predict()'s response shape."""
        if not history:
            return self._invalid_response(history, "Empty history provided; cannot predict.")

        valid_history = [c for c in history if c in self._model.camera_index]
        if not valid_history:
            return self._invalid_response(
                history,
                f"None of the provided cameras {history} are in the known "
                f"camera set ({len(self._camera_list)} cameras).",
            )

        curr = valid_history[-1]
        prev = valid_history[-2] if len(valid_history) >= 2 else None

        candidates = sorted(self._train_graph.get(curr, {}).keys())
        used_fallback = False
        if not candidates:
            candidates = list(self._camera_list)
            used_fallback = True

        speed_norm = self._speed_lookup.get(curr, 0.5)
        probs = self._model.score_candidates(curr, prev, candidates, speed_norm, dt_norm=0.5)

        ranked = sorted(zip(candidates, probs), key=lambda kv: kv[1], reverse=True)
        top_predictions = [self._enrich(cam, p) for cam, p in ranked[:top_k]]
        prob_dist = {cam: round(float(p), 6) for cam, p in ranked}

        if used_fallback:
            explanation = (
                f"Camera '{curr}' has no observed outgoing transitions in the "
                f"training graph; falling back to a graph-wide candidate set."
            )
        else:
            state_desc = f"{prev} -> {curr}" if prev else curr
            explanation = (
                f"GraphSAGE-style embedding of '{curr}' (2-hop neighbourhood over "
                f"geography, connectivity and traffic) scored against its "
                f"{len(candidates)} legal outgoing neighbours, conditioned on "
                f"history [{state_desc}]."
            )

        return {
            "input_valid": True,
            "history": history,
            "model_order_used": "gnn",
            "second_order_state_seen": f"{prev}_{curr}" if prev else None,
            "top_predictions": top_predictions,
            "probability_distribution": prob_dist,
            "explanation": explanation,
        }

    def roll_out_future_path(
        self, history: List[str], steps: int = 2, top_k: int = TOP_K,
        speed_mps: Optional[float] = None,
    ) -> List[str]:
        """Greedy multi-step rollout: repeatedly predict the top-1 next camera
        and append it to the working history."""
        working = list(history)
        path: List[str] = []
        for _ in range(steps):
            result = self.predict(working, top_k=top_k)
            if not result["input_valid"] or not result["top_predictions"]:
                break
            next_cam = result["top_predictions"][0]["camera_id"]
            path.append(next_cam)
            working.append(next_cam)
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
