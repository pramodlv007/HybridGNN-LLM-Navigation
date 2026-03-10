"""
APEX Baseline Controller (3D)
================================
Simple one-step scoring controller matching 2D APEX logic.
"""

import numpy as np
from .risk import compute_obstacle_risk


class ApexController:
    """
    APEX baseline controller — greedy one-step scoring.
    """

    def __init__(self, acc_scale=4.0):
        self.acc_scale = acc_scale
        self.dt = 0.05  # Match environment dt

    def act(self, obs):
        """Select best acceleration candidate by one-step scoring."""
        pos = obs["pos"]
        vel = obs["vel"]
        goal = obs["goal"]
        obstacles = obs["obstacles"]

        candidates = self._generate(goal, pos)

        best_score = -1e9
        best = np.zeros(3)

        for a in candidates:
            pred_vel = vel + a * self.dt
            pred_pos = pos + pred_vel * self.dt

            progress = -np.linalg.norm(pred_pos - goal)
            risk = compute_obstacle_risk(pred_pos, pred_vel, obstacles)
            score = progress - 2.0 * risk

            if score > best_score:
                best_score = score
                best = a

        return best

    def _generate(self, goal=None, pos=None):
        """Generate 9 cardinal (xy only) + goal-biased candidates."""
        actions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                actions.append(
                    np.array([dx, dy, 0], dtype=np.float64) * self.acc_scale
                )

        if goal is not None and pos is not None:
            direction = goal - pos
            norm = np.linalg.norm(direction)
            if norm > 0.1:
                direction = direction / norm
                actions.append(direction * self.acc_scale)
                actions.append(direction * self.acc_scale * 0.5)

        return actions
