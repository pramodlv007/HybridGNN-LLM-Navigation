"""
Hybrid Controller (3D) — Safety Shield + Rollout Scoring
==========================================================
Two-stage controller:
  1. Safety Shield: Filter unsafe candidates (proximity + TTC for dynamic)
  2. Rollout Scoring: Multi-step forward simulation scoring toward goal

Tuned for the mixed_random_env_20-style 3D environment (dt=0.05).
"""

import numpy as np
from .risk import ttc_risk, compute_obstacle_risk, proximity_risk


class HybridController:
    """
    Hybrid controller with safety shield and rollout-based goal scoring.
    """

    def __init__(self, horizon=8, acc_scale=4.0, risk_threshold=0.7):
        self.horizon = horizon
        self.acc_scale = acc_scale
        self.risk_threshold = risk_threshold
        self.dt = 0.05  # Match environment dt

    def act(self, obs):
        """Select best safe acceleration from candidates."""
        pos = obs["pos"]
        vel = obs["vel"]
        goal = obs["goal"]
        obstacles = obs["obstacles"]

        candidates = self._generate(goal, pos)

        # Stage 1: Safety shield — filter unsafe candidates
        safe = []
        for a in candidates:
            if self._is_safe(pos, vel, a, obstacles):
                safe.append(a)

        # Fallback: try smaller acceleration
        if not safe:
            for a in candidates:
                a_small = a * 0.2
                if self._is_safe(pos, vel, a_small, obstacles):
                    safe.append(a_small)

        # Ultimate fallback: brake toward zero velocity
        if not safe:
            brake = -vel * 2.0
            brake = np.clip(brake, -self.acc_scale, self.acc_scale)
            return brake

        # Stage 2: Rollout scoring — pick best safe candidate
        best_score = -1e9
        best = safe[0]

        for a in safe:
            score = self._rollout_score(pos, vel, a, goal, obstacles)
            if score > best_score:
                best_score = score
                best = a

        return best

    def _is_safe(self, pos, vel, accel, obstacles):
        """
        Check if a candidate acceleration is safe.
        - Static: proximity check only
        - Dynamic: proximity + TTC risk
        """
        p = pos.copy()
        v = vel.copy()

        for step in range(3):  # 3 sub-steps ahead
            v = v + accel * self.dt
            p = p + v * self.dt

            for obs in obstacles:
                collision_dist = 0.12 + obs["radius"] + 0.03

                if obs["is_dynamic"]:
                    risk = ttc_risk(p, v, obs["pos"], obs["vel"],
                                    radius=collision_dist + 0.05)
                    if risk > self.risk_threshold:
                        return False

                dist = np.linalg.norm(p - obs["pos"])
                if dist < collision_dist:
                    return False

        return True

    def _rollout_score(self, pos, vel, accel, goal, obstacles):
        """Score a candidate via multi-step forward rollout."""
        v = vel.copy()
        p = pos.copy()

        for _ in range(self.horizon):
            v = v + accel * self.dt
            speed = np.linalg.norm(v)
            if speed > 3.0:
                v = v / speed * 3.0
            p = p + v * self.dt

        # Primary: get close to goal
        goal_dist = np.linalg.norm(p - goal)
        score = -goal_dist

        # Secondary: avoid obstacles
        prox_risk = proximity_risk(p, obstacles, influence_radius=0.5)
        score -= 0.3 * prox_risk

        return score

    def _generate(self, goal=None, pos=None):
        """Generate candidate accelerations: 9 cardinal (xy only) + goal-biased."""
        actions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                actions.append(
                    np.array([dx, dy, 0], dtype=np.float64) * self.acc_scale
                )

        # Goal-biased candidates
        if goal is not None and pos is not None:
            direction = goal - pos
            norm = np.linalg.norm(direction)
            if norm > 0.1:
                direction = direction / norm
                actions.append(direction * self.acc_scale)
                actions.append(direction * self.acc_scale * 0.75)
                actions.append(direction * self.acc_scale * 0.5)
                # Lateral variants for obstacle avoidance
                perp_xy = np.array([-direction[1], direction[0], 0])
                for s in [0.3, -0.3]:
                    actions.append((direction + perp_xy * s) * self.acc_scale * 0.7)

        return actions
