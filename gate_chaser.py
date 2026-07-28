"""Deprecated shim — use ``kalman_planner.KalmanDualGatePlanner``.

Kept so older imports keep working; contains no IBVS / velocity chase logic.
"""

from kalman_planner import GateChaserPlanner, KalmanDualGatePlanner

__all__ = ['GateChaserPlanner', 'KalmanDualGatePlanner']
