"""
compare_service
================
Shared-data layer glue: fetches one plate's observed camera history, then
asks Part 1 (probabilistic Markov) and Part 2 (GNN) to predict independently
from the *same* observed prefix. Ground truth (later CSV observations, and
the SUMO road-edge route) is only attached for post-hoc scoring/display -
never passed into a predictor.
"""

from typing import Any, Dict, List, Optional

from markov_integration.config import MAX_TRAJECTORY_GAP_SECONDS, TOP_K
from markov_integration.db import get_plate_events
from markov_integration.markov_adapter import MarkovAdapter
from markov_integration.trajectory_service import (
    _split_into_segments,
    _collapse_consecutive_cameras,
)

try:
    from gnn_integration.torch_gnn_adapter import TorchGnnAdapter, TorchGnnNotAvailableError
except Exception:  # pragma: no cover - torch/PyG not installed
    TorchGnnAdapter = None
    TorchGnnNotAvailableError = RuntimeError

try:
    from gnn_integration.gnn_adapter import GnnAdapter, GnnNotTrainedError
except Exception:  # pragma: no cover - artefacts not trained yet
    GnnAdapter = None
    GnnNotTrainedError = RuntimeError

_markov_adapter: Optional[MarkovAdapter] = None
_gnn_adapter = None
_gnn_load_error: Optional[str] = None


def _get_markov_adapter() -> MarkovAdapter:
    global _markov_adapter
    if _markov_adapter is None:
        _markov_adapter = MarkovAdapter()
    return _markov_adapter


def get_gnn_adapter():
    """Returns the Part 2 GNN adapter, or None if nothing is available.

    Prefers the trained PyTorch Geometric GraphSAGE model
    (handoff_package_v2/trajectory_project/, used as-is/unmodified); falls
    back to the self-contained numpy GNN only if that isn't present.
    """
    global _gnn_adapter, _gnn_load_error
    if _gnn_adapter is not None or _gnn_load_error is not None:
        return _gnn_adapter

    if TorchGnnAdapter is not None:
        try:
            _gnn_adapter = TorchGnnAdapter()
            return _gnn_adapter
        except TorchGnnNotAvailableError:
            pass

    if GnnAdapter is not None:
        try:
            _gnn_adapter = GnnAdapter()
            return _gnn_adapter
        except GnnNotTrainedError as e:
            _gnn_load_error = str(e)

    if _gnn_adapter is None and _gnn_load_error is None:
        _gnn_load_error = "No GNN model available (neither trajectory_project nor gnn_integration artefacts found)."
    return _gnn_adapter


def gnn_load_error() -> Optional[str]:
    get_gnn_adapter()
    return _gnn_load_error


def get_full_camera_sequence(plate: str) -> Dict[str, Any]:
    """Latest contiguous observed segment for `plate`, collapsed to a
    duplicate-free camera sequence, plus per-camera raw event context."""
    raw_events = get_plate_events(plate)
    if not raw_events:
        return {"status": "plate_not_found", "sequence": [], "segment_events": []}

    segments = _split_into_segments(raw_events, MAX_TRAJECTORY_GAP_SECONDS)
    latest_segment = segments[-1]
    sequence = _collapse_consecutive_cameras(latest_segment)

    return {
        "status": "ok",
        "sequence": sequence,
        "segment_events": latest_segment,
        "raw_event_count": len(raw_events),
    }


def _prefix_last_speed(segment_events: List[Dict[str, Any]], prefix_length: int) -> Optional[float]:
    """Speed at the last raw event belonging to the `prefix_length`-th distinct
    (collapsed) camera in the segment -- i.e. the vehicle's real speed at the
    end of the observed prefix, not anywhere past it."""
    seen: List[str] = []
    last_event = None
    for evt in segment_events:
        cam = evt["camera_id"]
        if not seen or cam != seen[-1]:
            seen.append(cam)
        if len(seen) > prefix_length:
            break
        last_event = evt
    return last_event["speed_mps"] if last_event else None


def _path_match_rate(predicted: List[str], truth: List[str]) -> float:
    if not truth:
        return 0.0
    n = min(len(predicted), len(truth))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if predicted[i] == truth[i])
    return matches / len(truth)


def compare(plate: str, prefix_length: int = 2, rollout_steps: int = 2, top_k: int = TOP_K) -> Dict[str, Any]:
    """Runs both models on the same observed prefix and scores each against
    the held-back ground-truth future cameras. Mirrors the
    POST /api/predict/compare contract from the product spec."""
    info = get_full_camera_sequence(plate)
    if info["status"] != "ok":
        return {"plate": plate, "status": "plate_not_found"}

    full_sequence = info["sequence"]
    prefix_length = max(1, min(prefix_length, len(full_sequence)))
    observed_prefix = full_sequence[:prefix_length]
    ground_truth_future = full_sequence[prefix_length:prefix_length + rollout_steps]
    prefix_speed = _prefix_last_speed(info["segment_events"], prefix_length)

    markov_adapter = _get_markov_adapter()
    markov_next = markov_adapter.predict(observed_prefix, top_k=top_k)
    markov_future = markov_adapter.roll_out_future_path(
        observed_prefix, steps=rollout_steps, top_k=top_k
    )

    gnn_adapter = get_gnn_adapter()
    if gnn_adapter is not None:
        gnn_next = gnn_adapter.predict(observed_prefix, top_k=top_k, speed_mps=prefix_speed)
        gnn_future = gnn_adapter.roll_out_future_path(
            observed_prefix, steps=rollout_steps, top_k=top_k, speed_mps=prefix_speed
        )
    else:
        gnn_next = None
        gnn_future = []

    def _package(next_result, future_path):
        if next_result is None or not next_result["input_valid"]:
            return None
        top1 = next_result["top_predictions"][0] if next_result["top_predictions"] else None
        next_correct = (
            bool(ground_truth_future) and top1 is not None
            and top1["camera_id"] == ground_truth_future[0]
        )
        return {
            "next_camera": top1["camera_id"] if top1 else None,
            "confidence": top1["probability"] if top1 else 0.0,
            "top_k": [
                {"camera_id": p["camera_id"], "probability": p["probability"]}
                for p in next_result["top_predictions"]
            ],
            "future_path": future_path,
            "explanation": next_result["explanation"],
            "model_order_used": next_result.get("model_order_used"),
            "next_correct": next_correct,
            "path_match_rate": _path_match_rate(future_path, ground_truth_future),
        }

    return {
        "plate": plate,
        "status": "ok",
        "observed_prefix": observed_prefix,
        "full_observed_sequence": full_sequence,
        "ground_truth_future": ground_truth_future,
        "probabilistic": _package(markov_next, markov_future),
        "gnn": _package(gnn_next, gnn_future),
        "gnn_available": gnn_adapter is not None,
    }
