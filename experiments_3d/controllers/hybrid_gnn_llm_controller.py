"""
Hybrid GNN-LLM Controller (3D) — Final Optimized
===================================================================
Key changes from baseline:
  - 5-step safety shield (0.25s lookahead at 2.0 m/s = 0.5m coverage)
  - TTC threshold 0.6 for dynamic obstacles
  - NFZ-aware waypoint routing via StrategicNavigator
  - NFZ corridor speed reduction (1.5 m/s)
  - NFZ-only stagnation escape with tangent bearing
  - 16-candidate generation (matching 2D framework)
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

    START_POS = np.array([-3.6, -3.6, 1.0])

    def __init__(self, use_trained_model=False):
        self.gnn = GNNSafetyChecker(use_trained_model=use_trained_model)
        self.llm = StrategicNavigator()
        self.acc_scale = 4.0
        self.dt = 0.05

        self._step_count = 0
        self._prev_dist_to_goal = None

        # Stagnation detector state (NFZ only)
        self._dist_history = []
        self._stagnation_mode = False
        self._stagnation_timer = 0
        self._stagnation_direction = 1

    def act(self, obs):
        pos       = obs["pos"]
        vel       = obs["vel"]
        goal      = obs["goal"]
        obstacles = obs["obstacles"]
        nfz_blocks = obs.get("nfz_blocks", None)

        dist_to_goal = float(np.linalg.norm(pos[:2] - goal[:2]))

        # Detect episode reset
        if self._prev_dist_to_goal is not None and dist_to_goal > self._prev_dist_to_goal + 5.0:
            self._step_count = 0
            self._dist_history.clear()
            self._stagnation_mode = False
            self.llm.reset_episode()
        self._prev_dist_to_goal = dist_to_goal
        self._step_count += 1

        # ============================================
        # CORE GNN+LLM FRAMEWORK
        # ============================================

        # 1. Get Strategic Direction
        time_now = self._step_count * self.dt
        bearing = self.llm.get_bearing(pos, goal, time_now=time_now,
                                        nfz_blocks=nfz_blocks)

        # 2. Compute obstacle proximity for speed adaptation
        near_nfz = self._near_nfz(pos, nfz_blocks, margin=0.8)
        min_obs_dist = self._min_obstacle_dist(pos, obstacles)

        # 3. Generate Candidates — 16 directions (same as 2D)
        #    Adaptive speed: slow near NFZ or very close obstacles
        if near_nfz:
            speed = 1.5
        elif min_obs_dist < 0.5:
            speed = 1.5   # Close to obstacle — reduce speed
        else:
            speed = 2.0
        candidates = self._generate_candidates(speed=speed)

        # 4. Rank by GNN Safety (NFZ-aware)
        ranked_moves = self.gnn.rank_candidates(
            candidates, pos, obstacles, dt=0.5,
            nfz_blocks=nfz_blocks)

        # 5. Dynamic threshold based on context
        if dist_to_goal < 1.5:
            threshold = 10.0    # Near goal — push through to avoid timeout
        elif near_nfz:
            threshold = 5.0     # Conservative near NFZ walls
        elif min_obs_dist < 0.4:
            threshold = 6.0     # Conservative when very close to obstacles
        else:
            threshold = 8.0     # Standard

        safe_moves = [m for m in ranked_moves if m[1] < threshold]

        # 6. Select Best Aligned with LLM Bearing
        if not safe_moves:
            target_vel = ranked_moves[0][0]
        else:
            # Stagnation escape (all environments)
            self._update_stagnation(dist_to_goal)
            if self._stagnation_mode and dist_to_goal > 1.5:
                if nfz_blocks and near_nfz:
                    bearing = self._tangent_bearing(pos, goal, nfz_blocks)
                else:
                    bearing = self._rotate_bearing(
                        bearing, self._stagnation_direction * 30)

            best_move = max(safe_moves, key=lambda m: np.dot(m[0], bearing))
            target_vel = best_move[0]

        # ============================================
        # 3D ADAPTATION LAYER
        # ============================================

        # 7. Convert velocity to acceleration
        accel = (target_vel - vel) * 6.0
        accel[2] = 0.0
        accel = np.clip(accel, -self.acc_scale, self.acc_scale)

        # 8. Post-decision Safety Shield (5-step lookahead)
        accel = self._safety_shield(pos, vel, accel, obstacles, nfz_blocks)

        return accel

    # ===================================================
    # HELPERS
    # ===================================================

    def _min_obstacle_dist(self, pos, obstacles):
        """Minimum distance to any obstacle."""
        min_d = float("inf")
        for obs in obstacles:
            d = np.linalg.norm(pos - obs["pos"]) - obs["radius"]
            if d < min_d:
                min_d = d
        return min_d

    def _near_nfz(self, pos, nfz_blocks, margin=0.5):
        if not nfz_blocks:
            return False
        for block in nfz_blocks:
            if (block["min"][0] - margin <= pos[0] <= block["max"][0] + margin and
                    block["min"][1] - margin <= pos[1] <= block["max"][1] + margin):
                return True
        return False

    def _point_in_nfz(self, pos, nfz_blocks, margin=0.0):
        if not nfz_blocks:
            return False
        for block in nfz_blocks:
            if (block["min"][0] - margin <= pos[0] <= block["max"][0] + margin and
                    block["min"][1] - margin <= pos[1] <= block["max"][1] + margin):
                return True
        return False

    def _safety_shield(self, pos, vel, accel, obstacles, nfz_blocks=None):
        """
        Post-decision safety check — 5-step lookahead (0.25s).
        Checks obstacle collisions, dynamic TTC, and NFZ entry.
        """
        robot_radius = 0.12
        p = pos.copy()
        v = vel.copy()

        for step in range(5):
            v = v + accel * self.dt
            p = p + v * self.dt

            for obs in obstacles:
                collision_dist = robot_radius + obs["radius"] + 0.05
                dist = np.linalg.norm(p - obs["pos"])

                # Hard collision detection
                if dist < collision_dist:
                    escape_dir = pos - obs["pos"]
                    norm = np.linalg.norm(escape_dir)
                    if norm > 0.01:
                        escape_dir = escape_dir / norm
                    else:
                        escape_dir = np.array([1.0, 0.0, 0.0])
                    brake = escape_dir * self.acc_scale * 0.85
                    brake[2] = 0.0
                    return brake

                # TTC for dynamic obstacles
                if obs.get("is_dynamic"):
                    risk = ttc_risk(p, v, obs["pos"], obs["vel"],
                                    radius=collision_dist + 0.06)
                    if risk > 0.6:
                        escape_dir = pos - obs["pos"]
                        norm = np.linalg.norm(escape_dir)
                        if norm > 0.01:
                            escape_dir = escape_dir / norm
                        else:
                            escape_dir = np.array([1.0, 0.0, 0.0])
                        brake = escape_dir * self.acc_scale * 0.5
                        brake[2] = 0.0
                        return brake

            # NFZ entry check
            if self._point_in_nfz(p, nfz_blocks, margin=robot_radius + 0.08):
                escape = self._nfz_escape(pos, nfz_blocks)
                return escape

        return accel

    def _nfz_escape(self, pos, nfz_blocks):
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

    def _tangent_bearing(self, pos, goal, nfz_blocks):
        """Compute bearing tangent to NFZ wall, biased toward goal."""
        goal_vec = goal[:2] - pos[:2]
        goal_norm = np.linalg.norm(goal_vec)
        if goal_norm < 0.01:
            return np.array([0.0, 0.0, 0.0])
        goal_dir = goal_vec / goal_norm

        perp1 = np.array([-goal_dir[1], goal_dir[0]])
        perp2 = np.array([goal_dir[1], -goal_dir[0]])

        if nfz_blocks:
            cx = (nfz_blocks[0]["min"][0] + nfz_blocks[0]["max"][0]) / 2
            cy = (nfz_blocks[0]["min"][1] + nfz_blocks[0]["max"][1]) / 2
            from_nfz = pos[:2] - np.array([cx, cy])
            if np.dot(perp1, from_nfz) > np.dot(perp2, from_nfz):
                blended = 0.6 * perp1 + 0.4 * goal_dir
            else:
                blended = 0.6 * perp2 + 0.4 * goal_dir
        else:
            blended = goal_dir

        norm = np.linalg.norm(blended)
        if norm > 0.01:
            blended = blended / norm
        return np.array([blended[0], blended[1], 0.0])

    def _generate_candidates(self, speed):
        """16 evenly spaced directions (same as 2D)."""
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        return [np.array([speed * np.cos(a), speed * np.sin(a), 0.0])
                for a in angles]

    # ===================================================
    # STAGNATION DETECTOR (NFZ only)
    # ===================================================

    def _update_stagnation(self, dist_to_goal):
        self._dist_history.append(dist_to_goal)
        if len(self._dist_history) > 60:
            self._dist_history.pop(0)

        if self._stagnation_mode:
            self._stagnation_timer -= 1
            if self._stagnation_timer <= 0:
                self._stagnation_mode = False

        if len(self._dist_history) >= 60:
            progress = self._dist_history[0] - self._dist_history[-1]
            if progress < 0.2 and not self._stagnation_mode:
                self._stagnation_mode = True
                self._stagnation_timer = 25
                self._stagnation_direction *= -1
                self._dist_history.clear()

    def _rotate_bearing(self, bearing, degrees):
        rad = np.radians(degrees)
        cos_r, sin_r = np.cos(rad), np.sin(rad)
        rotated = np.array([
            bearing[0] * cos_r - bearing[1] * sin_r,
            bearing[0] * sin_r + bearing[1] * cos_r,
            0.0
        ])
        norm = np.linalg.norm(rotated[:2])
        if norm > 0.01:
            rotated = rotated / norm
        return rotated
