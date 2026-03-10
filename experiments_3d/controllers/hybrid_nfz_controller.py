"""
NFZ-aware Hybrid Controller (3D)
===================================
Extends the hybrid controller to STRICTLY avoid the No-Fly Zone.
The safety shield rejects ANY candidate that would move the drone
into or near the NFZ. Uses a safety margin around the NFZ.
"""

import numpy as np
from .risk import ttc_risk, proximity_risk


class HybridNFZController:
    """
    Hybrid controller with STRICT NFZ avoidance.
    Safety shield: proximity + TTC + NFZ entry with safety margin.
    """

    def __init__(self, horizon=15, acc_scale=4.0, risk_threshold=0.5):
        self.horizon = horizon
        self.acc_scale = acc_scale
        self.risk_threshold = risk_threshold
        self.dt = 0.05
        self.nfz_blocks = []
        self.nfz_margin = 0.1  # Reactive margin — only avoid when very close

    def act(self, obs):
        """
        Select best safe acceleration.
        Moves straight to goal until encountering NFZ/obstacles.
        """
        pos = obs["pos"]
        vel = obs["vel"]
        goal = obs["goal"]
        obstacles = obs["obstacles"]
        self.nfz_blocks = obs.get("nfz_blocks", [])

        candidates = self._generate(goal, pos)

        # Stage 1: Safety shield
        safe = []
        for a in candidates:
            if self._is_safe(pos, vel, a, obstacles):
                safe.append(a)

        # If no safe moves (likely hit NFZ boundary), use escape/small moves
        if not safe:
            # Try goal-biased tiny moves as fallback
            direction = (goal - pos)
            direction[2] = 0
            if np.linalg.norm(direction) > 0.01:
                direction /= np.linalg.norm(direction)
            for s in [0.05, 0.02]:
                a_tiny = direction * s * self.acc_scale
                if self._is_safe(pos, vel, a_tiny, obstacles):
                    safe.append(a_tiny)

        if not safe:
            # Try braking
            brake = -vel * 2.0
            brake = np.clip(brake, -self.acc_scale, self.acc_scale)
            if self._is_safe(pos, vel, brake, obstacles):
                safe.append(brake)

        # Reactive escape if stuck against NFZ
        if not safe:
            return self._nfz_escape(pos, vel)

        # Stage 2: Rollout scoring
        best_score = -1e9
        best = safe[0]
        for a in safe:
            score = self._rollout_score(pos, vel, a, goal, obstacles)
            if score > best_score:
                best_score = score
                best = a

        return best

    def _nfz_escape(self, pos, vel):
        """Reactive escape: push away from nearest wall if safety shield is tripped."""
        x, y = pos[0], pos[1]
        escape_vec = np.zeros(3)
        min_d = float('inf')

        for block in self.nfz_blocks:
            bmin, bmax = block["min"], block["max"]
            dx = max(bmin[0] - x, 0, x - bmax[0])
            dy = max(bmin[1] - y, 0, y - bmax[1])
            d = np.sqrt(dx*dx + dy*dy)
            if d < min_d:
                min_d = d
                px = np.clip(x, bmin[0], bmax[0])
                py = np.clip(y, bmin[1], bmax[1])
                escape_vec = pos - np.array([px, py, pos[2]])

        norm = np.linalg.norm(escape_vec)
        if norm > 0.001:
            escape_vec = escape_vec / norm
        else:
            escape_vec = -pos / (np.linalg.norm(pos) + 1e-6)

        escape_vec[2] = 0
        brake = -vel * 1.5
        # Stronger push away to ensure we don't get stuck in a logic loop
        escape = escape_vec * self.acc_scale * 0.9 + brake * 0.1
        return np.clip(escape, -self.acc_scale, self.acc_scale)

    def _nfz_proximity_cost(self, xy):
        """Only penalize if basically ON the boundary."""
        x, y = xy[0], xy[1]
        min_dist = float('inf')
        for block in self.nfz_blocks:
            dx = max(block["min"][0] - x, 0, x - block["max"][0])
            dy = max(block["min"][1] - y, 0, y - block["max"][1])
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = min(min_dist, dist)

        if min_dist < 0.2: # Very reactive
            return (0.2 - min_dist) * 20.0
        return 0.0

    def _point_in_nfz_margin(self, xy, margin=None):
        """Check if point is inside NFZ + safety margin."""
        if margin is None:
            margin = self.nfz_margin
        x, y = xy[0], xy[1]
        for block in self.nfz_blocks:
            if (block["min"][0] - margin <= x <= block["max"][0] + margin and
                block["min"][1] - margin <= y <= block["max"][1] + margin):
                return True
        return False

    def _is_safe(self, pos, vel, accel, obstacles):
        p = pos.copy()
        v = vel.copy()
        for step in range(5):
            v = v + accel * self.dt
            p = p + v * self.dt
            # Safety check: ONLY reject if we would actually ENTER or get within 0.1m
            if self._point_in_nfz_margin(p[:2]):
                return False
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
        v = vel.copy()
        p = pos.copy()
        nfz_cost = 0.0
        for _ in range(self.horizon):
            v = v + accel * self.dt
            speed = np.linalg.norm(v)
            if speed > 3.0:
                v = v / speed * 3.0
            p = p + v * self.dt
            nfz_cost += self._nfz_proximity_cost(p[:2])

        goal_dist = np.linalg.norm(p - goal)
        score = -goal_dist
        score -= 0.8 * proximity_risk(p, obstacles, influence_radius=1.0)
        score -= nfz_cost
        return score

    def _generate(self, goal, pos):
        actions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                actions.append(np.array([dx, dy, 0], dtype=np.float64) * self.acc_scale)

        direction = goal - pos
        norm = np.linalg.norm(direction)
        if norm > 0.1:
            direction = direction / norm
            direction[2] = 0
            # Goal-biased
            for s in [1.0, 0.75, 0.5]:
                actions.append(direction * self.acc_scale * s)
            # Lateral variants
            perp_xy = np.array([-direction[1], direction[0], 0])
            for s in [0.5, -0.5, 1.0, -1.0, 1.5, -1.5]:
                candidate = direction + perp_xy * s
                cnorm = np.linalg.norm(candidate)
                if cnorm > 0.01:
                    actions.append(candidate / cnorm * self.acc_scale * 0.7)
            # Pure lateral
            actions.append(perp_xy * self.acc_scale)
            actions.append(-perp_xy * self.acc_scale)
            # Backwards candidates for tight spots
            actions.append(-direction * self.acc_scale * 0.5)

        return actions
