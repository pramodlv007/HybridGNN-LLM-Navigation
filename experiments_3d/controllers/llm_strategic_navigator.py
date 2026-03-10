"""
Strategic Navigator (LLM Proxy) for 3D Navigation
===================================================
EXACT PORT of the 2D StrategicNavigator from hybrid_nav.py.
Updates bearing every 2 seconds, pointing directly toward goal.
"""

import numpy as np


class StrategicNavigator:
    """
    Simulates LLM strategic reasoning.
    Mirrors the 2D StrategicNavigator exactly.
    """

    def __init__(self):
        self.last_update_time = -10.0
        self.current_bearing = np.array([1.0, 0.0, 0.0])  # Default East

    def get_bearing(self, robot_pos, goal_pos, time_now):
        """
        Determines the high-level bearing toward goal.
        Updates every 2 seconds (simulates LLM call latency).
        Same logic as 2D hybrid_nav.py line 81-104.
        """
        if time_now - self.last_update_time > 2.0:
            # LLM Decision: Direct to Goal
            vec = goal_pos[:2] - robot_pos[:2]
            norm = np.linalg.norm(vec)
            if norm > 0.1:
                direction = np.array([vec[0]/norm, vec[1]/norm, 0.0])
            else:
                direction = np.array([0.0, 0.0, 0.0])

            self.current_bearing = direction
            self.last_update_time = time_now

        return self.current_bearing
