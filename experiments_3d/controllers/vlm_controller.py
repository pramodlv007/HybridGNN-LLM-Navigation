"""
VLM Controller (3D) — Vision-Language Model Navigation
========================================================
Uses GPT-4o vision to navigate by analyzing rendered camera frames.

Pipeline:
  1. Render chase camera view of the environment
  2. Encode frame as base64, send to GPT-4o with vision prompt
  3. GPT-4o returns a bearing angle (degrees) for safe navigation
  4. Convert bearing to velocity candidate
  5. GNN safety shield filters unsafe moves (reuses GNNSafetyChecker)

This is distinct from LLM-Only which uses TEXT state descriptions.
VLM uses the actual VISUAL observation for decision making.
"""

import math
import os
import sys
import base64
import io
import json
import numpy as np
import cv2

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .gnn_safety_checker import GNNSafetyChecker
from .risk import ttc_risk


class VLMController:
    """
    Vision-Language Model controller for 3D drone navigation.
    Uses GPT-4o vision to interpret camera frames and decide actions.
    """

    def __init__(self, fps=20, api_key=None):
        self.gnn = GNNSafetyChecker()  # Safety shield (same as others)
        self.fps = fps
        self.dt = 1.0 / fps
        self.acc_scale = 4.0
        self.cruise_speed = 1.8

        # VLM client
        self.client = None
        if OpenAI:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if key:
                self.client = OpenAI(api_key=key)

        # State
        self.bearing = np.array([1.0, 1.0, 0.0])
        self.last_vlm_time = -10.0
        self.vlm_interval = 2.0  # Call VLM every 2 seconds (same as LLM)
        self.step_count = 0
        self.cached_frame = None  # Set externally by benchmark runner
        self.vlm_calls = 0
        self.vlm_failures = 0

    def set_frame(self, frame):
        """Called by benchmark runner to provide the current camera frame."""
        self.cached_frame = frame

    def act(self, obs):
        """
        Main decision loop:
        1. If VLM interval elapsed → call GPT-4o vision with camera frame
        2. Generate candidates along VLM bearing
        3. GNN ranks by safety
        4. Select best safe candidate
        5. Safety shield
        """
        self.step_count += 1
        pos = obs["pos"]
        vel = obs["vel"]
        goal = obs["goal"]
        obstacles = obs["obstacles"]
        nfz_blocks = obs.get("nfz_blocks", None)
        time_now = self.step_count * self.dt

        # ============================================
        # STEP 1: VLM Decision (every 2 seconds)
        # ============================================
        if time_now - self.last_vlm_time > self.vlm_interval:
            bearing = self._call_vlm(pos, goal, obstacles)
            if bearing is not None:
                self.bearing = bearing
            self.last_vlm_time = time_now

        # ============================================
        # STEP 2: Generate 16 velocity candidates
        # ============================================
        candidates = self._generate_candidates(speed=self.cruise_speed)

        # ============================================
        # STEP 3: GNN ranks candidates by safety
        # ============================================
        ranked = self.gnn.rank_candidates(
            candidates, pos, obstacles, dt=0.5, nfz_blocks=nfz_blocks)

        # ============================================
        # STEP 4: Filter unsafe, pick best aligned with VLM bearing
        # ============================================
        safe_moves = [m for m in ranked if m[1] < 8.0]

        if not safe_moves:
            target_vel = ranked[0][0]  # Least risky
        else:
            best = max(safe_moves, key=lambda m: np.dot(m[0], self.bearing))
            target_vel = best[0]

        # ============================================
        # STEP 5: Convert velocity to acceleration
        # ============================================
        accel = (target_vel - vel) * 6.0
        accel[2] = 0.0
        accel = np.clip(accel, -self.acc_scale, self.acc_scale)

        # ============================================
        # STEP 6: Post-decision safety shield
        # ============================================
        accel = self._safety_shield(pos, vel, accel, obstacles, nfz_blocks)

        # Compute yaw from travel direction
        if np.linalg.norm(target_vel[:2]) > 0.1:
            yaw = math.atan2(target_vel[1], target_vel[0])
        else:
            yaw = math.atan2(self.bearing[1], self.bearing[0])

        return np.append(accel, yaw)

    def _call_vlm(self, pos, goal, obstacles):
        """
        Send camera frame + minimal context to GPT-4o vision.
        Returns bearing as np.array([bx, by, 0]) or None on failure.
        """
        if not self.client or self.cached_frame is None:
            # Fallback: point toward goal
            return self._goal_bearing(pos, goal)

        self.vlm_calls += 1

        try:
            # Encode frame to base64 JPEG
            _, buffer = cv2.imencode('.jpg', self.cached_frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, 60])
            img_b64 = base64.b64encode(buffer).decode('utf-8')

            # Build prompt
            goal_dir = goal[:2] - pos[:2]
            goal_dist = np.linalg.norm(goal_dir)
            goal_angle = math.degrees(math.atan2(goal_dir[1], goal_dir[0]))

            prompt = (
                f"You are a drone navigation AI. This is your chase camera view.\n"
                f"Goal is {goal_dist:.1f}m away at bearing {goal_angle:.0f}°.\n"
                f"Red spheres are dynamic obstacles. Blue spheres are static.\n"
                f"Green sphere is the goal.\n\n"
                f"Analyze the image and choose a SAFE bearing angle (0-360°) "
                f"to fly toward while avoiding obstacles.\n"
                f"Reply with ONLY a JSON object: {{\"bearing\": <angle_degrees>}}"
            )

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                            "detail": "low"
                        }}
                    ]
                }],
                max_tokens=20,
                temperature=0.0
            )

            text = response.choices[0].message.content.strip()

            # Parse JSON response
            # Handle markdown-wrapped JSON
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)
            angle_deg = float(data.get("bearing", goal_angle))
            angle_rad = math.radians(angle_deg)

            return np.array([math.cos(angle_rad), math.sin(angle_rad), 0.0])

        except Exception as e:
            self.vlm_failures += 1
            print(f"[VLM ERROR] API call failed: {type(e).__name__}: {e}")
            if "insufficient_quota" in str(e) or "429" in str(e):
                print("[VLM SYSTEM] Insufficient quota detected. Disabling VLM client to save time.")
                self.client = None  # Disable client to fail fast on subsequent calls
            # Fallback: point toward goal
            return self._goal_bearing(pos, goal)

    def _goal_bearing(self, pos, goal):
        """Fallback bearing: straight to goal."""
        d = goal[:2] - pos[:2]
        n = np.linalg.norm(d)
        if n > 0.1:
            return np.array([d[0]/n, d[1]/n, 0.0])
        return np.array([0.0, 0.0, 0.0])

    def _generate_candidates(self, speed):
        """16 velocity candidates in a circle (same as Hybrid)."""
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        return [np.array([speed * np.cos(a), speed * np.sin(a), 0.0])
                for a in angles]

    def _safety_shield(self, pos, vel, accel, obstacles, nfz_blocks=None):
        """Post-decision safety shield (identical to Hybrid)."""
        p = pos.copy()
        v = vel.copy()
        robot_radius = 0.18

        for s in range(5):
            v = v + accel * self.dt
            p = p + v * self.dt

            for obs in obstacles:
                collision_dist = robot_radius + obs["radius"] + 0.08
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
                    if risk > 0.5:
                        escape_dir = pos - obs["pos"]
                        norm = np.linalg.norm(escape_dir)
                        if norm > 0.01:
                            escape_dir = escape_dir / norm
                        else:
                            escape_dir = np.array([1.0, 0.0, 0.0])
                        brake = escape_dir * self.acc_scale * 0.5
                        brake[2] = 0.0
                        return brake

            if nfz_blocks:
                for block in nfz_blocks:
                    margin = robot_radius + 0.05
                    if (block["min"][0] - margin <= p[0] <= block["max"][0] + margin and
                        block["min"][1] - margin <= p[1] <= block["max"][1] + margin):
                        escape = self._nfz_escape(pos, nfz_blocks)
                        return escape

        return accel

    def _nfz_escape(self, pos, nfz_blocks):
        """NFZ escape maneuver (same as Hybrid)."""
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
