"""
markov_integration.trajectory_service
======================================
Orchestrates DB lookup → trajectory segmentation → Markov prediction.

Key behaviours
--------------
* sim_time is a raw SUMO simulation float (seconds).  It is **never**
  converted to calendar/wall-clock time.
* ALL raw events are returned in DB counts (including zero-speed hits),
  because repeated hits at the same camera carry congestion-analytics value.
* A gap of more than ``MAX_TRAJECTORY_GAP_SECONDS`` between consecutive
  events starts a new trajectory segment.
* Only the **latest** segment is used for prediction.
* Within that segment, **consecutive duplicate camera IDs** are collapsed
  before the Markov model sees the history (route prediction does not need
  repeated visits to the same camera).
"""

from typing import Dict, Any, List, Optional

from markov_integration.config import MAX_TRAJECTORY_GAP_SECONDS, TOP_K
from markov_integration.db import get_plate_events
from markov_integration.markov_adapter import MarkovAdapter

# Singleton adapter (loaded once per process)
_adapter: Optional[MarkovAdapter] = None


def _get_adapter() -> MarkovAdapter:
    global _adapter
    if _adapter is None:
        _adapter = MarkovAdapter()
    return _adapter


# ---------------------------------------------------------------------------
# Segment helpers
# ---------------------------------------------------------------------------

def _split_into_segments(
    events: List[Dict[str, Any]],
    max_gap: int = MAX_TRAJECTORY_GAP_SECONDS,
) -> List[List[Dict[str, Any]]]:
    """
    Split a chronologically ordered event list into contiguous segments.

    A new segment begins whenever the gap between consecutive events
    exceeds *max_gap* SUMO simulation seconds.
    """
    if not events:
        return []
    segments: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = [events[0]]
    for evt in events[1:]:
        prev_time = current[-1]["sim_time"]
        curr_time = evt["sim_time"]
        if (curr_time - prev_time) > max_gap:
            segments.append(current)
            current = [evt]
        else:
            current.append(evt)
    segments.append(current)
    return segments


def _collapse_consecutive_cameras(events: List[Dict[str, Any]]) -> List[str]:
    """
    Return a deduplicated list of camera IDs by collapsing consecutive repeats.

    Example: [CAM_01, CAM_01, CAM_11, CAM_02] → ['CAM_01', 'CAM_11', 'CAM_02']
    """
    history: List[str] = []
    for evt in events:
        cam = evt["camera_id"]
        if not history or cam != history[-1]:
            history.append(cam)
    return history


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class TrajectoryPredictionService:
    """
    High-level service: plate → trajectory segment → Markov prediction.

    Usage
    -----
    svc = TrajectoryPredictionService()
    result = svc.predict_for_plate("TN03QH4950")
    """

    def predict_for_plate(
        self,
        plate: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Predict the next camera for *plate*.

        Parameters
        ----------
        plate:
            Exact licence-plate string.
        top_k:
            Override the default top-k.  ``None`` → use config default.

        Returns
        -------
        dict with keys:
            plate, status, latest_sim_time, active_history,
            raw_event_count, active_segment_event_count,
            unique_camera_count, max_gap_seconds,
            markov_result (nested dict from MarkovAdapter.predict).

        Possible *status* values
        ------------------------
        "plate_not_found"       – plate has no events in DB
        "invalid_sim_time"      – one or more sim_time values are non-numeric
        "prediction_ready"      – markov_result is populated (may itself be
                                  invalid if history has no known cameras)
        """
        effective_top_k = top_k if top_k is not None else TOP_K

        # 1. Fetch raw events
        raw_events = get_plate_events(plate)

        if not raw_events:
            return {
                "plate": plate,
                "status": "plate_not_found",
                "latest_sim_time": None,
                "active_history": [],
                "raw_event_count": 0,
                "active_segment_event_count": 0,
                "unique_camera_count": 0,
                "max_gap_seconds": MAX_TRAJECTORY_GAP_SECONDS,
                "markov_result": None,
            }

        # 2. Validate sim_time values
        invalid_times = [
            e for e in raw_events
            if not isinstance(e["sim_time"], (int, float))
        ]
        if invalid_times:
            return {
                "plate": plate,
                "status": "invalid_sim_time",
                "latest_sim_time": None,
                "active_history": [],
                "raw_event_count": len(raw_events),
                "active_segment_event_count": 0,
                "unique_camera_count": 0,
                "max_gap_seconds": MAX_TRAJECTORY_GAP_SECONDS,
                "markov_result": None,
            }

        # 3. Split into segments and take the latest
        segments = _split_into_segments(raw_events, MAX_TRAJECTORY_GAP_SECONDS)
        latest_segment = segments[-1]

        # 4. Collapse consecutive duplicates for Markov input
        active_history = _collapse_consecutive_cameras(latest_segment)

        # 5. Derive summary stats
        latest_sim_time: float = latest_segment[-1]["sim_time"]
        unique_cameras = len(set(e["camera_id"] for e in latest_segment))

        # 6. Run prediction
        adapter = _get_adapter()
        markov_result = adapter.predict(active_history, top_k=effective_top_k)

        return {
            "plate": plate,
            "status": "prediction_ready",
            "latest_sim_time": latest_sim_time,
            "active_history": active_history,
            "raw_event_count": len(raw_events),
            "active_segment_event_count": len(latest_segment),
            "unique_camera_count": unique_cameras,
            "max_gap_seconds": MAX_TRAJECTORY_GAP_SECONDS,
            "markov_result": markov_result,
        }
