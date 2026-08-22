"""
markov_integration.markov_adapter
==================================
Loads the four Markov artefacts from ``handoff_package_v2/`` **once** at
construction time and provides a ``predict()`` method.

Artefact formats (confirmed by inspection)
------------------------------------------
transition_probs_1st.csv
    Square matrix CSV with a header row and row-label column.
    Row label = "from" camera, column header = "to" camera.
    Cell value = float probability.

transition_probs_2nd.json
    Dict keyed by "<from1>_<from2>" (underscore-joined bigram).
    Value = dict of {next_camera_id: probability}.

camera_names.json
    Dict keyed by camera_id.
    Value = {"lat": float, "lon": float, "name": str, "is_hub": bool}

camera_locations.json
    Dict keyed by camera_id.
    Value = {"lon": float, "lat": float}

Prediction logic
----------------
*  If the last two cameras in history form a known 2nd-order state →
   use 2nd-order distribution.
*  Otherwise fall back to 1st-order distribution for the latest camera.
*  Returns a structured dict; no prediction is fabricated for invalid input.

This predicts the **next observed camera** (not the final destination).

Baseline accuracy (from offline evaluation)
-------------------------------------------
  1st-order:  Top-1 24.8 %,  Top-3 46.8 %
  2nd-order:  Top-1 25.4 %,  Top-3 46.6 %
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from markov_integration.config import (
    TRANSITION_1ST_CSV,
    TRANSITION_2ND_JSON,
    CAMERA_NAMES_JSON,
    CAMERA_LOCATIONS_JSON,
    TOP_K,
)

_PROB_SUM_TOLERANCE = 0.01   # max drift corrected silently (CSV 3dp rounding is ~0.0005)


class MarkovAdapter:
    """
    Loads Markov artefacts once; exposes ``predict(history, top_k)``.

    Parameters
    ----------
    model_dir:
        Optional path override.  When *None* the paths from ``config`` are used.
    """

    def __init__(self) -> None:
        self._trans1: Dict[str, Dict[str, float]] = {}   # from → {to: prob}
        self._trans2: Dict[str, Dict[str, float]] = {}   # "a_b" → {to: prob}
        self._cam_meta: Dict[str, Dict[str, Any]] = {}   # id → full metadata
        self._allowed_cameras: set = set()

        self._load_transition_1st()
        self._load_transition_2nd()
        self._load_camera_metadata()

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_transition_1st(self) -> None:
        """
        Parse transition_probs_1st.csv.

        Format: first column = row label (from), first row = header (to).
        """
        path = TRANSITION_1ST_CSV
        matrix: Dict[str, Dict[str, float]] = {}
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)               # first row
            col_labels = header[1:]             # skip first empty cell
            for row in reader:
                if not row:
                    continue
                from_cam = row[0].strip()
                probs = {}
                for j, val in enumerate(row[1:]):
                    to_cam = col_labels[j].strip()
                    try:
                        probs[to_cam] = float(val)
                    except ValueError:
                        probs[to_cam] = 0.0
                matrix[from_cam] = self._normalise(from_cam, "1st", probs)

        self._trans1 = matrix
        self._allowed_cameras = set(matrix.keys())

    def _load_transition_2nd(self) -> None:
        """Parse transition_probs_2nd.json keyed by 'CAM_X_CAM_Y'."""
        with open(TRANSITION_2ND_JSON, encoding="utf-8") as fh:
            raw: Dict[str, Dict[str, float]] = json.load(fh)

        self._trans2 = {
            state: self._normalise(state, "2nd", dist)
            for state, dist in raw.items()
        }

    def _load_camera_metadata(self) -> None:
        """
        Merge camera_names.json (primary – has name + is_hub) and
        camera_locations.json (fallback for lat/lon).

        camera_names.json value: {"lat": …, "lon": …, "name": …, "is_hub": bool}
        camera_locations.json value: {"lon": …, "lat": …}
        """
        with open(CAMERA_NAMES_JSON, encoding="utf-8") as fh:
            names_data: Dict[str, Any] = json.load(fh)
        with open(CAMERA_LOCATIONS_JSON, encoding="utf-8") as fh:
            locs_data: Dict[str, Any] = json.load(fh)

        meta: Dict[str, Dict[str, Any]] = {}
        all_ids = set(names_data.keys()) | set(locs_data.keys())
        for cam_id in all_ids:
            n = names_data.get(cam_id, {})
            l = locs_data.get(cam_id, {})
            meta[cam_id] = {
                "camera_id": cam_id,
                "camera_name": n.get("name", cam_id),
                "latitude": n.get("lat", l.get("lat", None)),
                "longitude": n.get("lon", l.get("lon", None)),
                "is_hub": bool(n.get("is_hub", False)),
            }

        self._cam_meta = meta

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _normalise(
        self, label: str, order: str, dist: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Correct trivial floating-point rounding drift.

        If |sum − 1| ≤ _PROB_SUM_TOLERANCE → silently renormalise.
        If sum == 0 → leave as-is (will be caught at predict time).
        Raises ValueError for larger discrepancies.
        """
        total = sum(dist.values())
        if total == 0:
            return dist
        drift = abs(total - 1.0)
        if drift <= _PROB_SUM_TOLERANCE:
            # Silently renormalise trivial rounding drift (CSV 3dp → ~0.0005 max)
            return {k: v / total for k, v in dist.items()}
        raise ValueError(
            f"[MarkovAdapter] {order}-order '{label}': prob sum={total:.6f} "
            f"deviates by {drift:.6f}, which exceeds the correction threshold."
        )

    def _enrich_prediction(self, cam_id: str, prob: float) -> Dict[str, Any]:
        """Build a single top-prediction entry with camera metadata."""
        meta = self._cam_meta.get(cam_id, {})
        return {
            "camera_id": cam_id,
            "camera_name": meta.get("camera_name", cam_id),
            "latitude": meta.get("latitude"),
            "longitude": meta.get("longitude"),
            "is_hub": meta.get("is_hub", False),
            "probability": round(prob, 6),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        history: List[str],
        top_k: int = TOP_K,
    ) -> Dict[str, Any]:
        """
        Predict the **next observed camera** given a camera-visit history.

        Parameters
        ----------
        history:
            Ordered list of camera IDs (consecutive duplicates already
            collapsed by the caller).  Must contain at least one camera
            from the known camera set to produce a valid prediction.
        top_k:
            Number of top predictions to return.

        Returns
        -------
        dict with keys:
            input_valid, history, model_order_used, second_order_state_seen,
            top_predictions, probability_distribution, explanation.
        """
        # ---- validation --------------------------------------------------
        if not history:
            return self._invalid_response(
                history=history,
                reason="Empty history provided; cannot predict.",
            )

        # Strip cameras that are not in the known camera set
        valid_history = [c for c in history if c in self._allowed_cameras]
        if not valid_history:
            return self._invalid_response(
                history=history,
                reason=(
                    f"None of the provided cameras {history} are in the "
                    f"known camera set ({len(self._allowed_cameras)} cameras)."
                ),
            )

        # ---- choose model order ------------------------------------------
        use_order = 1
        state_key: Optional[str] = None
        dist: Optional[Dict[str, float]] = None

        if len(valid_history) >= 2:
            candidate_state = f"{valid_history[-2]}_{valid_history[-1]}"
            if candidate_state in self._trans2:
                use_order = 2
                state_key = candidate_state
                dist = self._trans2[candidate_state]

        if dist is None:
            last_cam = valid_history[-1]
            if last_cam not in self._trans1:
                return self._invalid_response(
                    history=history,
                    reason=f"Camera '{last_cam}' has no 1st-order transition row.",
                )
            dist = self._trans1[last_cam]

        # ---- build top-k predictions -------------------------------------
        sorted_preds = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        top_predictions = [
            self._enrich_prediction(cam_id, prob)
            for cam_id, prob in sorted_preds[:top_k]
        ]

        # Full probability distribution (all cameras, sorted)
        prob_dist = {
            cam_id: round(prob, 6)
            for cam_id, prob in sorted_preds
        }

        if use_order == 2:
            explanation = (
                f"Used 2nd-order Markov on state '{state_key}'. "
                f"Predicts next observed camera (not final destination)."
            )
        else:
            explanation = (
                f"Used 1st-order Markov from camera '{valid_history[-1]}'. "
                f"Predicts next observed camera (not final destination)."
            )

        return {
            "input_valid": True,
            "history": history,
            "model_order_used": use_order,
            "second_order_state_seen": state_key,
            "top_predictions": top_predictions,
            "probability_distribution": prob_dist,
            "explanation": explanation,
        }

    def _invalid_response(
        self, history: List[str], reason: str
    ) -> Dict[str, Any]:
        return {
            "input_valid": False,
            "history": history,
            "model_order_used": None,
            "second_order_state_seen": None,
            "top_predictions": [],
            "probability_distribution": {},
            "explanation": reason,
        }

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    @property
    def allowed_cameras(self) -> set:
        return set(self._allowed_cameras)

    @property
    def second_order_states(self) -> set:
        return set(self._trans2.keys())
