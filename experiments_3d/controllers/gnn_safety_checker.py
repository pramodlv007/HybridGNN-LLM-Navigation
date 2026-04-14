"""
GNN Safety Checker for 3D Drone Navigation
============================================
Core: inverse-square repulsion (same as 2D) [HEURISTIC MODE].
Enhanced with:
  - TTC risk for dynamic obstacles
  - NFZ proximity risk (proactive avoidance of No-Fly Zones)

Option: use_trained_model=True to use a physics-informed trained GNN
instead of the heuristic for obstacle risk scoring.
NFZ and wall rules are ALWAYS applied (environment constraints).
"""

import os
import numpy as np
import torch
from .risk import ttc_risk


class GNNSafetyChecker:
    """
    Evaluates potential velocity choices for safety.

    Two modes:
      - Heuristic (default): inverse-square + TTC + NFZ rules
      - Trained GNN: learned obstacle risk + NFZ/wall rules (environment constraints)
    """

    def __init__(self, use_trained_model=False, model_path=None):
        self.use_trained_model = use_trained_model
        self.model = None

        if use_trained_model:
            self._load_trained_model(model_path)

    def _load_trained_model(self, model_path=None):
        """Load the physics-informed trained GNN model."""
        from models.local_risk_gnn import LocalRiskGNN

        if model_path is None:
            # Default path relative to experiments_3d/
            base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
            model_path = os.path.join(base, "models", "trained_risk_gnn.pt")

        self.model = LocalRiskGNN(in_feats=7, hidden_dim=64, msg_dim=64,
                                  num_layers=2, dropout=0.1)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location="cpu",
                                                  weights_only=True))
            print(f"[GNNSafetyChecker] Loaded trained GNN from {model_path}")
        else:
            print(f"[GNNSafetyChecker] WARNING: No trained model at {model_path}, "
                  f"using random weights")

        self.model.eval()

    def rank_candidates(self, candidates, robot_pos, obstacles, dt=0.5,
                        nfz_blocks=None):
        """
        Ranks velocity candidates by predicted risk.
        Returns: Sorted list of (velocity, risk_score) from safest to riskiest.
        """
        candidates_with_risk = []

        for cand_vel in candidates:
            if self.use_trained_model and self.model is not None:
                risk_score = self._predict_risk_gnn(
                    robot_pos, cand_vel, obstacles, dt, nfz_blocks)
            else:
                risk_score = self._predict_risk_heuristic(
                    robot_pos, cand_vel, obstacles, dt, nfz_blocks)
            candidates_with_risk.append((cand_vel, risk_score))

        # Sort safe -> risky
        candidates_with_risk.sort(key=lambda x: x[1])
        return candidates_with_risk

    # ================================================================
    # MODE 1: TRAINED GNN (physics-informed learned risk)
    # ================================================================

    def _predict_risk_gnn(self, curr_pos, vel, obstacles, dt, nfz_blocks=None):
        """
        Use the trained GNN for obstacle risk scoring.
        NFZ and wall rules are ALWAYS applied (not learnable).
        """
        next_pos = curr_pos + vel * dt
        robot_vel = vel

        # Current frame features
        robot_speed = np.linalg.norm(robot_vel)
        robot_vel_dir = robot_vel / (robot_speed + 1e-6)

        features_t = [[1.0, curr_pos[0], curr_pos[1], curr_pos[2],
                       robot_vel_dir[0], robot_vel_dir[1], robot_vel_dir[2]]]
        features_dt = [[1.0, next_pos[0], next_pos[1], next_pos[2],
                        robot_vel_dir[0], robot_vel_dir[1], robot_vel_dir[2]]]

        for obs in obstacles:
            obs_pos = obs["pos"]
            obs_vel = obs["vel"]
            obs_speed = np.linalg.norm(obs_vel)
            obs_vel_dir = obs_vel / (obs_speed + 1e-6)
            obs_next = obs_pos + obs_vel * dt

            features_t.append([0.0, obs_pos[0], obs_pos[1], obs_pos[2],
                              obs_vel_dir[0], obs_vel_dir[1], obs_vel_dir[2]])
            features_dt.append([0.0, obs_next[0], obs_next[1], obs_next[2],
                               obs_vel_dir[0], obs_vel_dir[1], obs_vel_dir[2]])

        x_t = torch.tensor(features_t, dtype=torch.float32)
        x_t_dt = torch.tensor(features_dt, dtype=torch.float32)

        # Star topology: robot -> each obstacle
        N_obs = len(obstacles)
        edge_index = torch.tensor([[0]*N_obs, list(range(1, N_obs+1))],
                                  dtype=torch.long)

        # Forward pass
        with torch.no_grad():
            risk_logits = self.model(x_t, x_t_dt, edge_index, dt=dt)

        # Robot node risk (index 0) through sigmoid
        robot_risk = torch.sigmoid(risk_logits[0]).item()

        # Scale probability to match the heuristic's threshold (8.0).
        # A scaler of 10.0 means a probability of 0.8 becomes 8.0 (the rejection limit).
        gnn_risk = robot_risk * 10.0

        # === NFZ + WALL RULES (always applied, not learnable) ===
        env_risk = self._compute_env_risk(next_pos, nfz_blocks)

        return gnn_risk + env_risk

    def _compute_env_risk(self, next_pos, nfz_blocks):
        """
        Environment constraint risk (NFZ + walls).
        Always applied regardless of GNN/heuristic mode.
        """
        risk = 0.0

        # NFZ RISK
        if nfz_blocks:
            robot_radius = 0.12
            nfz_margin = robot_radius + 0.15

            for block in nfz_blocks:
                bmin = block["min"]
                bmax = block["max"]

                in_x = (bmin[0] - nfz_margin) <= next_pos[0] <= (bmax[0] + nfz_margin)
                in_y = (bmin[1] - nfz_margin) <= next_pos[1] <= (bmax[1] + nfz_margin)

                if in_x and in_y:
                    real_in_x = bmin[0] <= next_pos[0] <= bmax[0]
                    real_in_y = bmin[1] <= next_pos[1] <= bmax[1]

                    if real_in_x and real_in_y:
                        risk += 100.0
                    else:
                        risk += 30.0

        # WALL BOUNDARY RISK
        wall_margin = 0.4
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

    # ================================================================
    # MODE 2: HEURISTIC (original, unchanged)
    # ================================================================

    def _predict_risk_heuristic(self, curr_pos, vel, obstacles, dt, nfz_blocks=None):
        """
        Core heuristic risk prediction.
        Inverse-square repulsion + TTC for dynamic + NFZ + walls.
        """
        next_pos = curr_pos + vel * dt

        risk = 0.0

        # === OBSTACLE RISK ===
        for obs in obstacles:
            obs_pos = obs["pos"]
            dist = np.linalg.norm(next_pos - obs_pos)
            collision_dist = obs.get("radius", 0.08) + 0.22

            # Hard collision penalty
            if dist < collision_dist:
                risk += 50.0

            # Core inverse-square (nearby threats only)
            if dist < 0.05:
                dist = 0.05
            if dist < 1.0:
                risk += 0.5 / (dist * dist)

            # TTC risk for dynamic obstacles
            if obs.get("is_dynamic"):
                ttc = ttc_risk(next_pos, vel, obs_pos, obs["vel"],
                               radius=collision_dist + 0.1)
                if ttc > 0.5:
                    risk += ttc * 5.0

        # === NFZ + WALL RISK ===
        risk += self._compute_env_risk(next_pos, nfz_blocks)

        return risk
