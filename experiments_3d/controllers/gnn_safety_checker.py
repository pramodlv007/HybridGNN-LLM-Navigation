"""
GNN Safety Checker for 3D Drone Navigation
============================================
Core: inverse-square repulsion (same as 2D).
Enhanced with:
  - TTC risk for dynamic obstacles
  - NFZ proximity risk (proactive avoidance of No-Fly Zones)
"""

import numpy as np
from .risk import ttc_risk


class GNNSafetyChecker:
    """
    Evaluates potential velocity choices for safety.
    Core: inverse-square repulsion (same as 2D).
    Enhancements: TTC + NFZ awareness.
    """

    def __init__(self):
        pass

    def rank_candidates(self, candidates, robot_pos, obstacles, dt=0.5,
                        nfz_blocks=None):
        """
        Ranks velocity candidates by predicted risk.
        Returns: Sorted list of (velocity, risk_score) from safest to riskiest.
        """
        candidates_with_risk = []

        for cand_vel in candidates:
            risk_score = self._predict_risk(
                robot_pos, cand_vel, obstacles, dt, nfz_blocks)
            candidates_with_risk.append((cand_vel, risk_score))

        # Sort safe -> risky
        candidates_with_risk.sort(key=lambda x: x[1])
        return candidates_with_risk

    def _predict_risk(self, curr_pos, vel, obstacles, dt, nfz_blocks=None):
        """
        Core: inverse-square repulsion (from 2D).
        Enhancements:
          - Hard collision penalty + TTC for dynamic obstacles
          - NFZ proximity penalty (treats NFZ edges as walls)
        """
        next_pos = curr_pos + vel * dt

        risk = 0.0

        # === OBSTACLE RISK ===
        for obs in obstacles:
            obs_pos = obs["pos"]
            dist = np.linalg.norm(next_pos - obs_pos)
            # Allow drone to slip through tight 3D spaces
            collision_dist = obs.get("radius", 0.08) + 0.22  # Wider for compound drone body

            # Hard collision penalty
            if dist < collision_dist:
                risk += 50.0

            # Core inverse-square (nearby threats only)
            if dist < 0.05:
                dist = 0.05
            if dist < 1.0:  # Wider avoidance zone
                risk += 0.5 / (dist * dist)

            # TTC risk for dynamic obstacles
            if obs.get("is_dynamic"):
                ttc = ttc_risk(next_pos, vel, obs_pos, obs["vel"],
                               radius=collision_dist + 0.1)
                if ttc > 0.5:
                    risk += ttc * 5.0

        # === NFZ RISK (proactive avoidance) ===
        if nfz_blocks:
            robot_radius = 0.12
            nfz_margin = robot_radius + 0.15  # Stay this far from NFZ edge

            for block in nfz_blocks:
                bmin = block["min"]
                bmax = block["max"]

                # Check if predicted position would enter the inflated NFZ
                in_x = (bmin[0] - nfz_margin) <= next_pos[0] <= (bmax[0] + nfz_margin)
                in_y = (bmin[1] - nfz_margin) <= next_pos[1] <= (bmax[1] + nfz_margin)

                if in_x and in_y:
                    # Inside inflated NFZ — very high penalty
                    # Check if actually inside the real NFZ (extreme penalty)
                    real_in_x = bmin[0] <= next_pos[0] <= bmax[0]
                    real_in_y = bmin[1] <= next_pos[1] <= bmax[1]

                    if real_in_x and real_in_y:
                        risk += 100.0  # Cannot go here
                    else:
                        risk += 30.0   # Too close to NFZ edge

        # === WALL BOUNDARY RISK (proactive wall avoidance) ===
        # Arena bounds: x in [-4, 4], y in [-4, 4]
        wall_margin = 0.4  # Start penalizing 0.4m from wall
        arena_min, arena_max = -4.0, 4.0

        for axis in range(2):
            dist_to_low_wall = next_pos[axis] - arena_min
            dist_to_high_wall = arena_max - next_pos[axis]

            if dist_to_low_wall < wall_margin:
                d = max(dist_to_low_wall, 0.05)
                risk += 2.0 / (d * d)
            if dist_to_high_wall < wall_margin:
                d = max(dist_to_high_wall, 0.05)
                risk += 2.0 / (d * d)

        return risk
