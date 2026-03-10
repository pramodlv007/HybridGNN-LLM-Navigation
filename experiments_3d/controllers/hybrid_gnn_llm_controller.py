"""
Hybrid GNN-LLM Controller (3D)
=================================
Core Framework (UNCHANGED):
  1. GNN ranks 16 velocity candidates by risk
  2. Filter unsafe candidates
  3. LLM bearing selects best aligned safe candidate

3D Adaptation Layers (ADDED AROUND CORE):
  - Velocity → Acceleration conversion (PyBullet needs accel, not velocity)
  - Post-decision Safety Shield (last-resort collision + NFZ override)
  - NFZ-aware GNN risk scoring (proactive NFZ avoidance)
"""

import numpy as np
from .gnn_safety_checker import GNNSafetyChecker
from .llm_strategic_navigator import StrategicNavigator
from .risk import ttc_risk


class HybridGNNLLMController:
    """
    Core: Same GNN+LLM framework as 2D.
    Wrapper: Safety shield + NFZ awareness for 3D physics adaptation.
    """

    def __init__(self):
        self.gnn = GNNSafetyChecker()
        self.llm = StrategicNavigator()
        self.acc_scale = 4.0
        self.dt = 0.05  # Environment timestep

    def act(self, obs):
        """
        Core GNN+LLM decision + post-decision safety shield.
        """
        pos = obs["pos"]
        vel = obs["vel"]
        goal = obs["goal"]
        obstacles = obs["obstacles"]
        nfz_blocks = obs.get("nfz_blocks", None)
        time_now = obs.get("steps", 0) * self.dt

        # ============================================
        # CORE GNN+LLM FRAMEWORK (unchanged from 2D)
        # ============================================

        # 1. Get Strategic Direction (LLM)
        bearing = self.llm.get_bearing(pos, goal, time_now)

        # 2. Generate Candidates — 16 directions
        candidates = self._generate_candidates(speed=1.8)

        # 3. Rank by GNN Safety (now NFZ-aware)
        ranked_moves = self.gnn.rank_candidates(
            candidates, pos, obstacles, dt=0.5,
            nfz_blocks=nfz_blocks)

        # 4. Filter Unsafe — threshold 8.0
        safe_moves = [move for move in ranked_moves if move[1] < 8.0]

        # 5. Select Best Aligned with LLM Bearing
        if not safe_moves:
            target_vel = ranked_moves[0][0]
        else:
            best_move = max(safe_moves, key=lambda m: np.dot(m[0], bearing))
            target_vel = best_move[0]

        # ============================================
        # 3D ADAPTATION LAYER (added around core)
        # ============================================

        # 6. Convert velocity to acceleration
        accel = (target_vel - vel) * 6.0
        accel[2] = 0.0
        accel = np.clip(accel, -self.acc_scale, self.acc_scale)

        # 7. POST-DECISION SAFETY SHIELD (collision + NFZ)
        accel = self._safety_shield(pos, vel, accel, obstacles, nfz_blocks)

        return accel

    # ===================================================
    # 3D ADAPTATION HELPERS (not part of core framework)
    # ===================================================

    def _point_in_nfz(self, pos, nfz_blocks, margin=0.0):
        """Check if a point (with margin) is inside any NFZ block."""
        if not nfz_blocks:
            return False
        for block in nfz_blocks:
            if (block["min"][0] - margin <= pos[0] <= block["max"][0] + margin and
                block["min"][1] - margin <= pos[1] <= block["max"][1] + margin):
                return True
        return False

    def _safety_shield(self, pos, vel, accel, obstacles, nfz_blocks=None):
        """
        Post-decision safety check. Simulates 3 physics steps ahead.
        Checks for BOTH obstacle collisions AND NFZ entry.
        If danger detected, override with braking/escape.
        """
        p = pos.copy()
        v = vel.copy()
        robot_radius = 0.12

        for step in range(3):
            v = v + accel * self.dt
            p = p + v * self.dt

            # --- Check obstacle collisions ---
            for obs in obstacles:
                collision_dist = robot_radius + obs["radius"] + 0.03
                dist = np.linalg.norm(p - obs["pos"])

                if dist < collision_dist:
                    escape_dir = pos - obs["pos"]
                    norm = np.linalg.norm(escape_dir)
                    if norm > 0.01:
                        escape_dir = escape_dir / norm
                    else:
                        escape_dir = np.array([1.0, 0.0, 0.0])
                    brake = escape_dir * self.acc_scale * 0.8
                    brake[2] = 0.0
                    return brake

                if obs.get("is_dynamic"):
                    risk = ttc_risk(p, v, obs["pos"], obs["vel"],
                                    radius=collision_dist + 0.05)
                    if risk > 0.7:
                        escape_dir = pos - obs["pos"]
                        norm = np.linalg.norm(escape_dir)
                        if norm > 0.01:
                            escape_dir = escape_dir / norm
                        else:
                            escape_dir = np.array([1.0, 0.0, 0.0])
                        brake = escape_dir * self.acc_scale * 0.5
                        brake[2] = 0.0
                        return brake

            # --- STRICT NFZ CHECK ---
            # If predicted position would enter NFZ (with robot radius margin),
            # override with braking away from NFZ
            if self._point_in_nfz(p, nfz_blocks, margin=robot_radius + 0.05):
                # Find which block we'd enter and escape away from it
                escape = self._nfz_escape(pos, nfz_blocks)
                return escape

        return accel  # No danger — pass through GNN+LLM decision

    def _nfz_escape(self, pos, nfz_blocks):
        """Compute escape acceleration away from the nearest NFZ block."""
        best_escape = -pos / (np.linalg.norm(pos) + 0.01) * self.acc_scale * 0.5

        if not nfz_blocks:
            return best_escape

        min_dist = float("inf")
        for block in nfz_blocks:
            cx = (block["min"][0] + block["max"][0]) / 2
            cy = (block["min"][1] + block["max"][1]) / 2
            center = np.array([cx, cy, pos[2]])
            dist = np.linalg.norm(pos[:2] - center[:2])
            if dist < min_dist:
                min_dist = dist
                escape_dir = pos - center
                norm = np.linalg.norm(escape_dir[:2])
                if norm > 0.01:
                    escape_dir = escape_dir / norm
                else:
                    escape_dir = np.array([1.0, 0.0, 0.0])
                best_escape = escape_dir * self.acc_scale * 0.6
                best_escape[2] = 0.0

        return best_escape

    def _generate_candidates(self, speed):
        """16 evenly spaced directions (same as 2D)."""
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        cands = [np.array([speed*np.cos(a), speed*np.sin(a), 0.0]) for a in angles]
        return cands
