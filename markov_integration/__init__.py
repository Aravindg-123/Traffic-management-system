"""
markov_integration
==================
Read-only SQLite → Markov trajectory prediction integration.

Public API
----------
from markov_integration import TrajectoryPredictionService
svc = TrajectoryPredictionService()
result = svc.predict_for_plate("TN03QH4950")
"""

from markov_integration.trajectory_service import TrajectoryPredictionService

__all__ = ["TrajectoryPredictionService"]
__version__ = "1.0.0"
