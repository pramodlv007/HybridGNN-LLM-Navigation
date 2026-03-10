"""
Explicit LLM Controller (3D) — Clean Rewrite
================================================
Modeled directly on the proven HybridGNNLLMController architecture.

Core Loop:
  1. LLM picks an explicit JSON action (rotate, translate, swing, etc.)
  2. Action sets a strategic BEARING (direction vector), NOT raw velocity.
  3. GNN generates 16 velocity candidates and ranks them by safety.
  4. Best safe candidate aligned with the LLM's bearing is executed.
  5. Post-decision safety shield (identical to HybridGNNLLMController).

This gives the fluid obstacle avoidance of eval_gnn_llm while
maintaining explicit JSON action logging and HUD display.
"""

import math
import json
import random
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
try:
    from experiments.cat_expt.utils.cat_game_agent import LLM_Agent
except ImportError:
    LLM_Agent = None

from .gnn_safety_checker import GNNSafetyChecker
from .risk import ttc_risk


class ExplicitLLMController:
    """
    Clean rewrite modeled on HybridGNNLLMController.
    LLM provides macro bearing; GNN provides micro avoidance.
    """

    def __init__(self, fps=20):
        self.gnn = GNNSafetyChecker()
        self.agent = LLM_Agent(model='gpt-4o') if LLM_Agent else None

        self.fps = fps
        self.dt = 1.0 / fps
        self.acc_scale = 4.0        # Same as HybridGNNLLMController
        self.cruise_speed = 1.8     # Same as HybridGNNLLMController

        self.frames_left = 0
        self.current_action = None
        self.drone_yaw = 0.0
        self.bearing = np.array([1.0, 0.0, 0.0])  # Current strategic bearing
        self.max_risk = 0.0

        # Load available moves
        moves_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "../../experiments/cat_expt/env/drone_available_move.json"))
        try:
            with open(moves_path, 'r') as f:
                self.available_move = f.read()
        except Exception:
            self.available_move = '{"error": "Failed to load drone_available_move.json"}'

        self.initialized_yaw = False
        self.log_file = None
        self.step_count = 0

    def set_log_file(self, log_file):
        self.log_file = log_file

    def get_hud_stats(self):
        return {"action": self.current_action, "risk": self.max_risk}

    def _log(self, step, msg):
        if self.log_file:
            self.log_file.write(f"[{step * self.dt:05.2f}s] {msg}\n")
            self.log_file.flush()
        print(f"  [{step * self.dt:.1f}s] {msg}")

    # ==================================================================
    # MAIN ACT — modeled on HybridGNNLLMController.act()
    # ==================================================================
    def act(self, obs):
        self.step_count += 1
        step = self.step_count

        pos = obs["pos"]
        vel = obs["vel"]
        goal = obs["goal"]
        obstacles = obs["obstacles"]
        nfz_blocks = obs.get("nfz_blocks", None)

        if not self.initialized_yaw:
            yaw_vec = goal[:2] - pos[:2]
            self.drone_yaw = math.atan2(yaw_vec[1], yaw_vec[0])
            self.bearing = np.array([math.cos(self.drone_yaw),
                                     math.sin(self.drone_yaw), 0.0])
            self.initialized_yaw = True

        # ==============================================================
        # STEP 1: LLM Macro Decision (when current action expires)
        # ==============================================================
        if self.frames_left <= 0:
            self._ask_llm(step, pos, vel, goal, obstacles, nfz_blocks)

        # Decrement frames
        if self.frames_left > 0:
            self.frames_left -= 1

        # ==============================================================
        # STEP 2: Convert LLM action → strategic bearing
        #         (same concept as StrategicNavigator.get_bearing)
        # ==============================================================
        self.bearing = self._action_to_bearing(pos, vel, goal)

        # ==============================================================
        # STEP 3: Generate 16 velocity candidates (IDENTICAL to Hybrid)
        # ==============================================================
        candidates = self._generate_candidates(speed=self.cruise_speed)

        # ==============================================================
        # STEP 4: GNN ranks candidates by safety (IDENTICAL to Hybrid)
        # ==============================================================
        ranked = self.gnn.rank_candidates(
            candidates, pos, obstacles, dt=0.5, nfz_blocks=nfz_blocks)

        # Track max risk for HUD
        if ranked:
            self.max_risk = ranked[0][1]

        # ==============================================================
        # STEP 5: Filter unsafe, pick best aligned with bearing
        #         (IDENTICAL to HybridGNNLLMController)
        # ==============================================================
        safe_moves = [m for m in ranked if m[1] < 8.0]

        if not safe_moves:
            target_vel = ranked[0][0]  # Least risky
        else:
            best = max(safe_moves, key=lambda m: np.dot(m[0], self.bearing))
            target_vel = best[0]

        # ==============================================================
        # STEP 6: Convert velocity to acceleration (IDENTICAL to Hybrid)
        # ==============================================================
        accel = (target_vel - vel) * 6.0
        accel[2] = 0.0
        accel = np.clip(accel, -self.acc_scale, self.acc_scale)

        # ==============================================================
        # STEP 7: Post-decision safety shield (IDENTICAL to Hybrid)
        # ==============================================================
        accel = self._safety_shield(pos, vel, accel, obstacles, nfz_blocks)

        # Update yaw from actual movement direction
        if np.linalg.norm(target_vel[:2]) > 0.1:
            self.drone_yaw = math.atan2(target_vel[1], target_vel[0])

        # Return 4D: [ax, ay, az, yaw] for visual rotation
        return np.append(accel, self.drone_yaw)

    # ==================================================================
    # LLM Decision Logic
    # ==================================================================
    def _ask_llm(self, step, pos, vel, goal, obstacles, nfz_blocks):
        """Ask LLM for next explicit action."""
        # Build obstacle perception string
        obs_list = []
        for o in obstacles:
            d = np.linalg.norm(pos - o["pos"])
            if d < 4.0:
                obs_list.append((d, o["pos"]))
        obs_list.sort(key=lambda x: x[0])
        obs_str = ""
        for i, (d, opos) in enumerate(obs_list[:3]):
            rel = opos - pos
            angle = math.degrees(math.atan2(rel[1], rel[0]) - self.drone_yaw)
            angle = (angle + 180) % 360 - 180
            obs_str += f"  - Obs {i+1}: Dist {d:.2f}m, Angle {angle:.1f}deg\n"
        if not obs_str:
            obs_str = "  - No obstacles within 4m\n"

        warn = "Clear"
        if self.max_risk > 5.0:
            warn = f"DANGER: Risk {self.max_risk:.1f}"

        state_desc = (
            f"Time: {step*self.dt:.1f}s\n"
            f"Drone pos: [{pos[0]:.1f}, {pos[1]:.1f}]\n"
            f"Goal: [{goal[0]:.1f}, {goal[1]:.1f}]\n"
            f"Yaw: {math.degrees(self.drone_yaw):.0f}deg\n"
            f"Obstacles:\n{obs_str}"
            f"Status: {warn}\n"
            f"Choose EXACTLY ONE action from 'available_move'."
        )

        move = {"error": "No Agent"}
        if self.agent:
            try:
                move, _ = self.agent.decide_move(state_desc, self.available_move)
            except Exception:
                move = {"error": "API Error"}

        if isinstance(move, dict) and "action_type" in move:
            self.current_action = move
            desc = move.get('description', 'Unknown')
            act_type = move.get('action_type', 'hover')
            self._log(step, f"LLM ACTION SELECTED: {desc} (Type: {act_type})")

            dur = float(move.get("duration", 0.5))
            self.frames_left = max(int(dur * self.fps), 1)

            if act_type == "rotate":
                self.drone_yaw += math.radians(move.get("yaw_change", 0.0))
        else:
            self._log(step, "LLM API Error/Quota. Using MOCK actions.")
            self._mock_action(step, pos, goal)

    def _mock_action(self, step, pos, goal):
        """Smart mock fallback biased toward goal."""
        mock_actions = [
            {"description": "Move Forward 1m", "action_type": "translate",
             "velocity": [2.0, 0.0, 0.0], "duration": 1.0},
            {"description": "Turn Left 30 deg", "action_type": "rotate",
             "yaw_change": 30.0, "duration": 0.5},
            {"description": "Turn Right 30 deg", "action_type": "rotate",
             "yaw_change": -30.0, "duration": 0.5},
            {"description": "Swing Left", "action_type": "translate_local",
             "velocity": [0.0, 1.5, 0.0], "duration": 0.5},
        ]

        goal_angle = math.degrees(math.atan2(goal[1] - pos[1], goal[0] - pos[0]))
        curr_angle = math.degrees(self.drone_yaw)
        diff = (goal_angle - curr_angle + 180) % 360 - 180

        if abs(diff) > 20:
            self.current_action = mock_actions[1] if diff > 0 else mock_actions[2]
        else:
            self.current_action = random.choices(
                mock_actions, weights=[0.7, 0.1, 0.1, 0.1])[0]

        desc = self.current_action['description']
        act_type = self.current_action['action_type']
        self._log(step, f"[MOCK] ACTION SELECTED: {desc}")

        dur = float(self.current_action.get("duration", 0.5))
        self.frames_left = max(int(dur * self.fps), 1)

        if act_type == "rotate":
            self.drone_yaw += math.radians(
                self.current_action.get("yaw_change", 0.0))

    # ==================================================================
    # Convert explicit action → bearing vector
    # ==================================================================
    def _action_to_bearing(self, pos, vel, goal):
        """
        ALWAYS point bearing toward the goal — same as StrategicNavigator.
        The LLM's explicit action (rotate/translate) affects yaw for visual
        rotation and logging, but the GNN bearing ALWAYS aims at the goal.
        This is the key insight from the original HybridGNNLLMController.
        """
        # Primary bearing: straight toward goal (identical to StrategicNavigator)
        d = goal - pos
        n = np.linalg.norm(d[:2])
        if n > 0.1:
            goal_bearing = np.array([d[0]/n, d[1]/n, 0.0])
        else:
            goal_bearing = np.array([0.0, 0.0, 0.0])

        if not self.current_action:
            return goal_bearing

        act_type = self.current_action.get("action_type", "hover")

        if act_type == "translate_local":
            # Lateral swing: perpendicular to yaw for dodge maneuver
            lat_yaw = self.drone_yaw + math.pi / 2
            return np.array([math.cos(lat_yaw),
                             math.sin(lat_yaw), 0.0])

        # ALL other actions (translate, rotate, hover, etc.)
        # use goal bearing — the GNN picks the safest path toward goal
        return goal_bearing

    # ==================================================================
    # 16 velocity candidates (IDENTICAL to HybridGNNLLMController)
    # ==================================================================
    def _generate_candidates(self, speed):
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        return [np.array([speed * np.cos(a), speed * np.sin(a), 0.0])
                for a in angles]

    # ==================================================================
    # Post-decision safety shield (IDENTICAL to HybridGNNLLMController)
    # ==================================================================
    def _safety_shield(self, pos, vel, accel, obstacles, nfz_blocks=None):
        p = pos.copy()
        v = vel.copy()
        robot_radius = 0.18  # Compound drone body is wider than 0.12 sphere

        for s in range(5):  # 5-step lookahead for earlier detection
            v = v + accel * self.dt
            p = p + v * self.dt

            for obs in obstacles:
                collision_dist = robot_radius + obs["radius"] + 0.08  # Wider margin
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
                    if risk > 0.5:  # Lower threshold = earlier evasion
                        escape_dir = pos - obs["pos"]
                        norm = np.linalg.norm(escape_dir)
                        if norm > 0.01:
                            escape_dir = escape_dir / norm
                        else:
                            escape_dir = np.array([1.0, 0.0, 0.0])
                        brake = escape_dir * self.acc_scale * 0.5
                        brake[2] = 0.0
                        return brake

            # NFZ check
            if nfz_blocks:
                for block in nfz_blocks:
                    margin = robot_radius + 0.05
                    if (block["min"][0] - margin <= p[0] <= block["max"][0] + margin and
                        block["min"][1] - margin <= p[1] <= block["max"][1] + margin):
                        escape = self._nfz_escape(pos, nfz_blocks)
                        return escape

        return accel

    def _nfz_escape(self, pos, nfz_blocks):
        best = -pos / (np.linalg.norm(pos) + 0.01) * self.acc_scale * 0.5
        if not nfz_blocks:
            return best
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
                best = escape_dir * self.acc_scale * 0.6
                best[2] = 0.0
        return best
