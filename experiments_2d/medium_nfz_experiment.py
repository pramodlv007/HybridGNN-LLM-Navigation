"""
Medium NFZ Experiment (2 No-Fly Zones)
=======================================
Environment with TWO No-Fly Zones:
  NFZ 1 (center):    x=[-1.2, 1.2], y=[-1.2, 1.2]  (2.4 x 2.4)
  NFZ 2 (top-right): x=[ 2.2, 3.2], y=[ 2.2, 3.2]  (1.0 x 1.0)

Robot navigates from (-3.6, -3.6) to goal (3.6, 3.6) avoiding both NFZs.

Usage:
    python experiments/cat_expt/medium_nfz_experiment.py --method APEX
    python experiments/cat_expt/medium_nfz_experiment.py --method PULSE
    python experiments/cat_expt/medium_nfz_experiment.py --method HYBRID
"""

import time
import json
import random
from datetime import datetime
import mujoco
import numpy as np
import cv2
import torch
import os
import sys
import sys
import argparse
import math
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.cat_game_agent import LLM_Agent
from utils.APEX import APEX
from model.graphormer import DiffGraphormer
from model.local_risk_gnn import LocalRiskGNN
from utils.mujoco_simulator import get_body_state, get_all_body_states

# ============================================================
# CONSTANTS
# ============================================================
START_POS = np.array([-3.6, -3.6, 0.12])
TARGET_POS = np.array([3.6, 3.6, 0.12])
NUM_DYNAMIC = 10
NUM_STATIC = 15
NUM_TOTAL = NUM_DYNAMIC + NUM_STATIC

# --- NFZ ZONES (list of dicts) ---
NFZ_ZONES = [
    {
        "name": "Center NFZ",
        "x_min": -1.2, "x_max": 1.2,
        "y_min": -1.2, "y_max": 1.2,
        "center": np.array([0.0, 0.0]),
    },
    {
        "name": "Right-Side NFZ",
        "x_min": 2.5, "x_max": 3.5,
        "y_min": -2.5, "y_max": -1.5, # Moved bottom (was -0.5 to 0.5)
        "center": np.array([3.0, -2.0]),
    },
]

NFZ_MARGIN = 0.3

base_path = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_path, "env/available_move.json"), 'r') as f:
    available_move = json.load(f)


# ============================================================
# MULTI-ZONE NFZ UTILITY FUNCTIONS
# ============================================================
def is_inside_any_nfz(x, y, margin=0.0):
    """Check if (x, y) is inside ANY NFZ zone."""
    for z in NFZ_ZONES:
        if (z["x_min"] - margin <= x <= z["x_max"] + margin and
            z["y_min"] - margin <= y <= z["y_max"] + margin):
            return True
    return False


def nfz_distance_to_zone(x, y, zone):
    """Distance from (x,y) to a single NFZ zone boundary. Negative = inside."""
    dx = max(zone["x_min"] - x, 0, x - zone["x_max"])
    dy = max(zone["y_min"] - y, 0, y - zone["y_max"])
    if dx == 0 and dy == 0:
        return -min(x - zone["x_min"], zone["x_max"] - x,
                    y - zone["y_min"], zone["y_max"] - y)
    return np.sqrt(dx**2 + dy**2)


def min_nfz_distance(x, y):
    """Minimum distance to any NFZ (most negative = deepest inside)."""
    return min(nfz_distance_to_zone(x, y, z) for z in NFZ_ZONES)


def closest_nfz(x, y):
    """Return the NFZ zone closest to (x, y)."""
    best_zone = NFZ_ZONES[0]
    best_dist = nfz_distance_to_zone(x, y, best_zone)
    for z in NFZ_ZONES[1:]:
        d = nfz_distance_to_zone(x, y, z)
        if d < best_dist:
            best_dist = d
            best_zone = z
    return best_zone, best_dist


def nfz_repulsion(robot_pos, strength=5.0, influence_radius=2.0):
    """Repulsion from the closest NFZ zone."""
    x, y = robot_pos[0], robot_pos[1]
    zone, dist = closest_nfz(x, y)

    if dist > influence_radius:
        return np.array([0.0, 0.0])

    diff = robot_pos[:2] - zone["center"]
    norm = np.linalg.norm(diff)
    if norm < 0.01:
        diff = np.array([1.0, 1.0])
        norm = np.linalg.norm(diff)
    direction = diff / norm

    if dist <= 0:
        magnitude = strength * 3.0
    else:
        magnitude = strength * (1.0 / (dist + 0.1))

    return direction * magnitude


def combined_nfz_repulsion(robot_pos, strength=5.0, influence_radius=2.0):
    """Combined repulsion from ALL nearby NFZ zones."""
    x, y = robot_pos[0], robot_pos[1]
    total_rep = np.array([0.0, 0.0])

    for zone in NFZ_ZONES:
        dist = nfz_distance_to_zone(x, y, zone)
        if dist > influence_radius:
            continue
        diff = robot_pos[:2] - zone["center"]
        norm = np.linalg.norm(diff)
        if norm < 0.01:
            diff = np.array([1.0, 1.0])
            norm = np.linalg.norm(diff)
        direction = diff / norm

        if dist <= 0:
            magnitude = strength * 3.0
        else:
            magnitude = strength * (1.0 / (dist + 0.1))
        total_rep += direction * magnitude

    return total_rep


# ============================================================
# ADVANCED SAFETY (TTC + Geometry)
# ============================================================
TTC_THRESHOLD   = 1.2    # seconds
ETA_Risk_Weight = 2.0
ETA_Turn_Weight = 0.5

def calculate_ttc(rob_pos, rob_vel, obs_pos, obs_vel, radius_sum):
    """
    Calculate Time-To-Collision between two circles moving at constant velocity.
    Returns: ttc (float) -> time in seconds, or infinity if no collision.
    """
    rel_pos = obs_pos - rob_pos
    rel_vel = obs_vel - rob_vel
    
    a = np.dot(rel_vel, rel_vel)
    b = 2.0 * np.dot(rel_pos, rel_vel)
    c = np.dot(rel_pos, rel_pos) - radius_sum*radius_sum
    
    if abs(a) < 1e-6:
        if c <= 0: return 0.0
        return float('inf')
        
    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return float('inf')
        
    sqrt_d = math.sqrt(discriminant)
    t1 = (-b - sqrt_d) / (2*a)
    t2 = (-b + sqrt_d) / (2*a)
    
    if t1 < 0 and t2 < 0: return float('inf')
    if t1 < 0: return t2
    return t1

def would_enter_nfz_swept(current_pos, velocity, dt_check=1.0, steps=5, margin=None):
    """Check if swept path enters any NFZ."""
    if margin is None:
        margin = NFZ_MARGIN

    start = current_pos[:2]
    end   = start + np.array(velocity[:2]) * dt_check
    
    for i in range(steps + 1):
        t = i / steps
        pt = start + (end - start) * t
        if is_inside_any_nfz(pt[0], pt[1], margin=margin):
            return True
    return False

def would_enter_nfz(current_pos, velocity, dt=0.5):
    """Check if velocity would move robot into ANY NFZ (legacy single point)."""
    future_pos = current_pos[:2] + np.array(velocity[:2]) * dt
    return is_inside_any_nfz(future_pos[0], future_pos[1], margin=NFZ_MARGIN)


# ============================================================
# OBSTACLE HELPERS
# ============================================================
def get_random_velocity(speed=1.0):
    angle = random.uniform(0, 2 * np.pi)
    return speed * np.cos(angle), speed * np.sin(angle)


def randomize_all_obstacles(model, data):
    """Randomize positions avoiding ALL NFZs, start, and goal."""
    np.random.seed(int(time.time()))
    random.seed(int(time.time()))
    placed = []

    for i in range(1, NUM_TOTAL + 1):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
        if body_id == -1:
            continue
        dof_start = model.body_dofadr[body_id]

        for _ in range(200):
            rx, ry = np.random.uniform(-3.3, 3.3), np.random.uniform(-3.3, 3.3)
            pos = np.array([rx, ry])

            if (not is_inside_any_nfz(rx, ry, margin=0.5) and
                np.linalg.norm(pos - START_POS[:2]) > 1.5 and
                np.linalg.norm(pos - TARGET_POS[:2]) > 1.5 and
                all(np.linalg.norm(pos - p) > 0.5 for p in placed)):
                z = 0.1 if i <= NUM_DYNAMIC else 0.08
                data.qpos[dof_start:dof_start + 3] = [rx, ry, z]
                placed.append(pos)
                break

    print(f"  Randomized {len(placed)} obstacles (avoiding {len(NFZ_ZONES)} NFZs)")
    return placed


def update_dynamic_obstacles(model, data, speed=1.0):
    """Dynamic obstacles bounce off NFZs too."""
    for i in range(1, NUM_DYNAMIC + 1):
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
            if body_id == -1:
                continue
            dof_start = model.body_dofadr[body_id]
            pos = data.xpos[body_id]
            vx, vy = data.qvel[dof_start], data.qvel[dof_start + 1]

            if pos[0] < -3.5: vx = abs(vx)
            elif pos[0] > 3.5: vx = -abs(vx)
            if pos[1] < -3.5: vy = abs(vy)
            elif pos[1] > 3.5: vy = -abs(vy)

            if is_inside_any_nfz(pos[0], pos[1], margin=0.3):
                repulsion = combined_nfz_repulsion(pos)
                vx += repulsion[0] * 0.5
                vy += repulsion[1] * 0.5

            if random.random() < 0.3:
                vx, vy = get_random_velocity(speed)
            data.qvel[dof_start], data.qvel[dof_start + 1] = vx, vy
        except Exception:
            pass


# ============================================================
# METHOD 1: APEX ORIGINAL
# ============================================================
def run_apex_method(physical_model, data, renderer, fps, dt, max_steps):
    agent = LLM_Agent(model='gpt-4o')
    danger_model = DiffGraphormer(in_feats=7, edge_feat_dim=3, hidden_dim=32, num_heads=4, dropout=0.3)
    danger_model.load_state_dict(torch.load(
        os.path.join(base_path, 'model/diffgraphormer_physics.pt'), map_location='cpu'))
    danger_model.eval()

    apex = APEX(graphormer_model=danger_model, physics_simulator="mujoco",
                llm_agent=agent, dt=dt, available_move=available_move)

    collision = False
    nfz_violated = False
    is_success = False
    collision_threshold = 0.2
    init_frames = int(fps / 2)
    robot_body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    robot_dof = physical_model.body_dofadr[robot_body_id]
    current_action = None
    frames_left = 0
    action_index = -1
    action_sequence = []

    def add_action(_move):
        nonlocal current_action, frames_left, action_index
        action_sequence.append(_move)
        if frames_left <= 0 and action_index + 1 < len(action_sequence):
            action_index += 1
            current_action = action_sequence[action_index]
            if not isinstance(current_action, dict):
                return
            frames_left = int(current_action["duration"] * fps)
            vel = current_action["velocity"]
            robot_state = get_body_state(physical_model, data, "robot")
            robot_pos = np.array(robot_state["position"][:3])
            if would_enter_nfz(robot_pos, vel):
                print("    [NFZ] Blocked move -> applying repulsion")
                repulsion = combined_nfz_repulsion(robot_pos)
                vel = [repulsion[0] * 1.5, repulsion[1] * 1.5, 0.0]
            data.qvel[robot_dof:robot_dof + 3] = vel

    snapshot_t, snapshot_t_dt = None, None

    for step in range(max_steps):
        robot_state = get_body_state(physical_model, data, "robot")
        robot_pos = np.array(robot_state["position"][:2])

        if is_inside_any_nfz(robot_pos[0], robot_pos[1]):
            if not nfz_violated:
                _, closest_d = closest_nfz(robot_pos[0], robot_pos[1])
                print(f"  [X] NFZ VIOLATION at step {step}!")
            nfz_violated = True

        dist_to_target = np.linalg.norm(robot_pos - TARGET_POS[:2])
        if dist_to_target < 0.5:
            print(f"  [SUCCESS] Reached goal at step {step} ({step/fps:.1f}s)!")
            is_success = True
            break

        for i in range(1, NUM_TOTAL + 1):
            try:
                cat_state = get_body_state(physical_model, data, f"cat{i}")
                cat_dist = np.linalg.norm(
                    np.array(robot_state["position"][:3]) - np.array(cat_state["position"][:3]))
                if step > init_frames and cat_dist < collision_threshold:
                    if not collision:
                        print(f"  [COLLISION] with cat{i} at step {step}")
                    collision = True
            except:
                pass

        snapshot_t_dt = {"objects": get_all_body_states(physical_model, data)}

        if step > init_frames and frames_left <= 0:
            if snapshot_t:
                triggered, move, action_valid = apex.run(
                    snapshot_t, snapshot_t_dt, dt, physical_model, data, step)

                if not triggered or move == "stay":
                    nfz_d = min_nfz_distance(robot_pos[0], robot_pos[1])
                    nfz_warning = ""
                    if nfz_d < 2.0:
                        zones_str = "; ".join(
                            f"{z['name']}: x=[{z['x_min']},{z['x_max']}] y=[{z['y_min']},{z['y_max']}]"
                            for z in NFZ_ZONES)
                        nfz_warning = (
                            f"\nCRITICAL: NO-FLY ZONES nearby! "
                            f"Zones: {zones_str}. "
                            f"Closest NFZ distance: {nfz_d:.2f}m. "
                            f"Navigate AROUND all NFZs!"
                        )

                    state_description = (
                        f"Robot Position: {robot_state['position'][:2]}\n"
                        f"Target: {TARGET_POS[:2]}\n"
                        f"Distance to target: {dist_to_target:.2f}m\n"
                        f"MISSION: Navigate to goal AVOIDING all obstacles "
                        f"AND two No-Fly Zones (center and top-right)."
                        f"{nfz_warning}\n"
                        f"Choose best move."
                    )
                    move, action_valid = agent.decide_move(state_description, available_move)
                    triggered = True

                if triggered and isinstance(move, dict):
                    add_action(move)

        snapshot_t = snapshot_t_dt

        if frames_left > 0:
            frames_left -= 1
            if frames_left <= 0 and action_index + 1 < len(action_sequence):
                action_index += 1
                current_action = action_sequence[action_index]
                if isinstance(current_action, dict):
                    frames_left = int(current_action["duration"] * fps)
            
            if isinstance(current_action, dict):
                vel = current_action["velocity"]
                robot_pos_3d = np.array(robot_state["position"][:3])
                if would_enter_nfz(robot_pos_3d, vel):
                    repulsion = combined_nfz_repulsion(robot_pos_3d)
                    vel = [repulsion[0] * 1.5, repulsion[1] * 1.5, 0.0]
                data.qvel[robot_dof:robot_dof + 3] = vel
        else:
            data.qvel[robot_dof:robot_dof + 3] = [0.0, 0.0, 0.0]

        if step % fps == 0:
            update_dynamic_obstacles(physical_model, data)

        mujoco.mj_step(physical_model, data)
        renderer.update_scene(data, camera="top")
        frame = cv2.cvtColor(np.flipud(renderer.render()), cv2.COLOR_RGB2BGR)
        yield frame, step

    yield None, {"success": is_success, "collision": collision, "nfz_violated": nfz_violated,
                 "steps": step, "time": step/fps}


# ============================================================
# METHOD 2: PULSE / 3-TIER
# ============================================================
def run_pulse_method(physical_model, data, renderer, fps, dt, max_steps):
    agent = LLM_Agent(model='gpt-4o')

    fast_gnn = LocalRiskGNN(in_feats=7, hidden_dim=64)
    fast_gnn.eval()

    graphormer = DiffGraphormer(in_feats=7, edge_feat_dim=3, hidden_dim=32, num_heads=4, dropout=0.3)
    graphormer.load_state_dict(torch.load(
        os.path.join(base_path, 'model/diffgraphormer_physics.pt'), map_location='cpu'))
    graphormer.eval()

    apex = APEX(graphormer_model=graphormer, physics_simulator="mujoco",
                llm_agent=agent, dt=dt, available_move=available_move)

    robot_body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    robot_dof = physical_model.body_dofadr[robot_body_id]

    collision = False
    nfz_violated = False
    collision_count = 0
    collision_threshold = 0.2
    init_frames = int(fps / 2)

    high_threshold = 0.7
    med_threshold = 0.3
    tier_counts = {1: 0, 2: 0, 3: 0}

    snapshot_t = None
    current_velocity = np.array([0.0, 0.0, 0.0])
    decision_interval = 15

    stuck_counter = 0
    last_position = START_POS[:2].copy()
    check_interval = 2 * fps

    def fast_risk_with_nfz(snapshot_t, snapshot_t_dt, dt_val):
        objs = snapshot_t["objects"]
        objs_dt = snapshot_t_dt["objects"]
        robot_pos = None
        robot_vel = None
        robot_idx = None
        for idx, obj in enumerate(objs):
            if obj["name"] == "robot":
                robot_pos = np.array(obj["position"][:3])
                robot_pos_dt = np.array(objs_dt[idx]["position"][:3])
                robot_vel = (robot_pos_dt - robot_pos) / max(dt_val, 1e-6)
                robot_idx = idx
                break
        if robot_pos is None:
            return 0.0, 0.0, []

        max_risk = 0.0
        dangerous_positions = []

        for idx, obj in enumerate(objs):
            if idx == robot_idx or obj["name"] == "floor":
                continue
            obs_pos = np.array(obj["position"][:3])
            dist = np.linalg.norm(robot_pos[:2] - obs_pos[:2])
            if dist > 4.0:
                continue
            proximity = min(1.0, 1.0 / (dist * dist + 0.1))
            obs_pos_dt = np.array(objs_dt[idx]["position"][:3])
            obs_vel = (obs_pos_dt - obs_pos) / max(dt_val, 1e-6)
            rel_vec = robot_pos[:2] - obs_pos[:2]
            rel_vel = obs_vel[:2] - robot_vel[:2]
            rel_norm = np.linalg.norm(rel_vec)
            vel_norm = np.linalg.norm(rel_vel)
            if rel_norm > 1e-6 and vel_norm > 1e-6:
                approach_factor = max(0.0, np.dot(rel_vel, rel_vec) / (vel_norm * rel_norm))
            else:
                approach_factor = 0.0
            risk = proximity * (0.3 + 0.7 * approach_factor)
            if risk > max_risk:
                max_risk = risk
            if risk > med_threshold:
                dangerous_positions.append(obs_pos)

        # Multi-zone NFZ risk
        nfz_dist = min_nfz_distance(robot_pos[0], robot_pos[1])
        if nfz_dist < 2.0:
            nfz_risk = min(1.0, 1.0 / (nfz_dist * nfz_dist + 0.05))
            if nfz_risk > max_risk:
                max_risk = nfz_risk
            if nfz_risk > med_threshold:
                zone, _ = closest_nfz(robot_pos[0], robot_pos[1])
                dangerous_positions.append(np.array([zone["center"][0], zone["center"][1], 0.0]))

        return max_risk, max_risk, dangerous_positions

    def tier3_greedy(robot_pos, speed=2.5):
        direction = TARGET_POS[:2] - robot_pos[:2]
        dist = np.linalg.norm(direction)
        if dist < 0.1:
            return np.array([0.0, 0.0, 0.0])
        goal_dir = direction / dist

        repulsion = combined_nfz_repulsion(robot_pos)
        rep_norm = np.linalg.norm(repulsion)

        if rep_norm > 0.1:
            rep_dir = repulsion / rep_norm
            blend = min(0.8, rep_norm / 5.0)
            combined = (1.0 - blend) * goal_dir + blend * rep_dir
            norm = np.linalg.norm(combined)
            if norm > 0.1:
                combined = combined / norm
        else:
            combined = goal_dir

        return np.array([combined[0] * speed, combined[1] * speed, 0.0])

    def tier2_gnn(robot_pos, dangerous_positions, speed=2.5):
        goal_dir = TARGET_POS[:2] - robot_pos[:2]
        goal_dist = np.linalg.norm(goal_dir)
        if goal_dist > 0.1:
            goal_dir = goal_dir / goal_dist

        repulsion = np.array([0.0, 0.0])
        for obs_pos in dangerous_positions:
            diff = robot_pos[:2] - obs_pos[:2]
            dist = np.linalg.norm(diff)
            if dist > 0.1:
                repulsion += (diff / dist) * (1.0 / (dist ** 2))

        nfz_rep = combined_nfz_repulsion(robot_pos)
        repulsion += nfz_rep * 0.5

        if np.linalg.norm(repulsion) > 0.1:
            repulsion = repulsion / np.linalg.norm(repulsion)

        rep_strength = np.linalg.norm(repulsion)
        if rep_strength > 2.0:
            blend = 0.8
        elif rep_strength > 0.5:
            blend = 0.5
        else:
            blend = 0.2

        combined = (1.0 - blend) * goal_dir + blend * repulsion
        norm = np.linalg.norm(combined)
        if norm > 0.1:
            combined = combined / norm

        return np.array([combined[0] * speed, combined[1] * speed, 0.0])

    for step in range(max_steps):
        robot_state = get_body_state(physical_model, data, "robot")
        robot_pos = np.array(robot_state["position"][:2])

        if is_inside_any_nfz(robot_pos[0], robot_pos[1]):
            if not nfz_violated:
                print(f"  [X] NFZ VIOLATION at step {step}!")
            nfz_violated = True

        if step % fps == 0:
            dist = np.linalg.norm(robot_pos - TARGET_POS[:2])
            nfz_d = min_nfz_distance(robot_pos[0], robot_pos[1])
            print(f"  [{step//fps:02d}s] Pos: [{robot_pos[0]:.2f}, {robot_pos[1]:.2f}] | "
                  f"Goal: {dist:.2f}m | NFZ: {nfz_d:.2f}m")
            update_dynamic_obstacles(physical_model, data)

        if step % check_interval == 0 and step > 0:
            movement = np.linalg.norm(robot_pos - last_position)
            if movement < 0.3:
                stuck_counter += 1
                if stuck_counter >= 3:
                    print(f"  [EARLY STOP] Robot stuck")
                    break
            else:
                stuck_counter = 0
            last_position = robot_pos.copy()

        dist_to_target = np.linalg.norm(robot_pos - TARGET_POS[:2])
        if dist_to_target < 0.5:
            print(f"  [SUCCESS] Reached goal at step {step} ({step/fps:.1f}s)!")
            break

        for i in range(1, NUM_TOTAL + 1):
            try:
                cat_state = get_body_state(physical_model, data, f"cat{i}")
                cat_dist = np.linalg.norm(
                    np.array(robot_state["position"][:3]) - np.array(cat_state["position"][:3]))
                if step > init_frames and cat_dist < collision_threshold:
                    if not collision:
                        print(f"  [COLLISION] First collision with cat{i} at step {step}")
                    collision = True
                    collision_count += 1
            except:
                pass

        snapshot_t_dt = {"objects": get_all_body_states(physical_model, data)}

        if step > init_frames and step % decision_interval == 0 and snapshot_t is not None:
            robot_risk, max_risk, dangerous_positions = fast_risk_with_nfz(
                snapshot_t, snapshot_t_dt, dt)
            robot_pos_3d = np.array(robot_state["position"][:3])

            if max_risk > high_threshold or robot_risk > 0.8:
                tier = 1
                try:
                    triggered, move, action_valid = apex.run(
                        snapshot_t, snapshot_t_dt, dt, physical_model, data, step)
                    if triggered and isinstance(move, dict):
                        vel = move["velocity"]
                        velocity = np.array(vel if len(vel) >= 3 else vel + [0.0])
                        if would_enter_nfz(robot_pos_3d, velocity):
                            velocity = tier2_gnn(robot_pos_3d, dangerous_positions)
                        current_velocity = velocity
                    else:
                        current_velocity = tier2_gnn(robot_pos_3d, dangerous_positions)
                except Exception:
                    current_velocity = tier2_gnn(robot_pos_3d, dangerous_positions)

            elif max_risk > med_threshold:
                tier = 2
                current_velocity = tier2_gnn(robot_pos_3d, dangerous_positions)

            else:
                tier = 3
                current_velocity = tier3_greedy(robot_pos_3d)

            # Final NFZ safety
            if would_enter_nfz(robot_pos_3d, current_velocity):
                rep = combined_nfz_repulsion(robot_pos_3d)
                current_velocity = np.array([rep[0] * 2.0, rep[1] * 2.0, 0.0])

            tier_counts[tier] += 1

        data.qvel[robot_dof:robot_dof + 3] = current_velocity
        snapshot_t = snapshot_t_dt

        for i in range(NUM_DYNAMIC + 1, NUM_TOTAL + 1):
            try:
                body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
                if body_id != -1:
                    dof_start = physical_model.body_dofadr[body_id]
                    data.qvel[dof_start:dof_start + 3] = [0.0, 0.0, 0.0]
            except:
                pass

        mujoco.mj_step(physical_model, data)
        renderer.update_scene(data, camera="top")
        frame = cv2.cvtColor(np.flipud(renderer.render()), cv2.COLOR_RGB2BGR)
        yield frame, step

    print(f"  Tier stats: T1={tier_counts[1]}, T2={tier_counts[2]}, T3={tier_counts[3]}")
    yield None, {"collision": collision, "nfz_violated": nfz_violated,
                 "collision_count": collision_count, "steps": step, "time": step/fps}


# ============================================================
# METHOD 3: HYBRID GNN-LLM
# ============================================================
# ============================================================
# METHOD 3: HYBRID GNN-LLM (Enhanced with TTC/ETA)
# ============================================================
def run_hybrid_method(physical_model, data, renderer, fps, dt, max_steps):
    robot_body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    robot_dof = physical_model.body_dofadr[robot_body_id]
    
    collision = False
    nfz_violated = False
    success = False
    collision_threshold = 0.2
    init_frames = int(fps / 2)

    snapshot_t = None
    last_bearing_time = -10.0
    current_bearing = np.array([1.0, 1.0, 0.0])

    stuck_counter = 0
    last_position = START_POS[:2].copy()
    check_interval = 0.5 * fps # Faster stuck detection (0.5s instead of 2.0s)
    
    current_vel_vector = np.array([0.0, 0.0, 0.0])
    recovery_steps = 0

    def predict_safety_and_score(curr_pos, vel, obstacles_data, dt_val, bearing_vec, static_margin=0.25):
        """
        Returns: (is_safe, score)
        is_safe: boolean (TTC > threshold AND no NFZ entry)
        score: float (ETA progress - risk penalties)
        """
        # 1. NFZ Geometric Filter (Hard Constraint)
        if would_enter_nfz_swept(curr_pos, vel[:2], dt_check=0.8):
            return False, -float('inf')

        # Soft NFZ Risk
        predicted_pos = curr_pos[:2] + vel[:2] * dt_val
        nfz_d = min_nfz_distance(predicted_pos[0], predicted_pos[1])
        if nfz_d < 0.4:
             risk_penalty = 5.0 / (nfz_d + 0.1)
        else:
             risk_penalty = 0.0

        # Goal Attraction Boost (Drone "Target Lock" when close)
        dist_to_goal = np.linalg.norm(predicted_pos - TARGET_POS[:2])
        if dist_to_goal < 1.0:
            risk_penalty -= 2.0 / (dist_to_goal + 0.1)

        # 2. Moving Obstacle TTC Filter (Hard Shield)
        rob_vel = vel[:2]
        # risk_penalty is accumulated below
        min_ttc = float('inf')
        
        for i in range(1, NUM_TOTAL + 1):
            try:
                # Direct data access for speed
                body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
                if body_id == -1: continue
                
                obs_pos = data.xpos[body_id][:2]
                
                is_dynamic = (i <= NUM_DYNAMIC)
                
                if is_dynamic:
                    # TTC for moving obstacles
                    dof_adr = physical_model.body_dofadr[body_id]
                    obs_v = data.qvel[dof_adr:dof_adr+2]
                    radius_sum = 0.12 + 0.12 + 0.15
                    
                    # Relative velocity
                    rel_v = rob_vel - obs_v
                    
                    # Only check if moving towards each other
                    dv = np.dot(rel_v, rel_v)
                    if dv > 1e-4:
                        ttc = calculate_ttc(curr_pos, rob_vel, obs_pos, obs_v, radius_sum)
                        if ttc < min_ttc: min_ttc = ttc
                        
                        if ttc < 0.6: 
                            return False, -float('inf')
                        
                        if ttc < 1.5:
                            risk_penalty += 2.0 / (ttc + 0.1)
                else:
                    # STATIC OBSTACLE HANDLING
                    dist_to_static = np.linalg.norm(curr_pos - obs_pos)
                    # Hard veto at margin (0.25 normal, 0.21 creep)
                    if dist_to_static < static_margin:
                        return False, -float('inf')

                    # Swept proximity check for static obstacles (multiple samples)
                    for t_frac in [0.2, 0.5, 0.8, 1.0]:
                        sample_pos = curr_pos + vel[:2] * dt_val * t_frac
                        dist_s = np.linalg.norm(sample_pos - obs_pos)
                        if dist_s < static_margin:
                            return False, -float('inf')
                    
                    if dist_to_static < 0.40:
                         risk_penalty += 0.5 / (dist_to_static + 0.05)
            except:
                pass
                
        # 3. ETA Scoring
        predicted_pos = curr_pos[:2] + vel[:2] * dt_val
        dist_now = np.linalg.norm(curr_pos[:2] - TARGET_POS[:2])
        dist_future = np.linalg.norm(predicted_pos - TARGET_POS[:2])
        progress = dist_now - dist_future
        
        vel_norm = np.linalg.norm(vel[:2])
        if vel_norm > 1e-3:
            vel_dir = vel[:2] / vel_norm
            alignment = np.dot(vel_dir, bearing_vec[:2])
            turn_cost = (1.0 - alignment) * ETA_Turn_Weight
        else:
            turn_cost = 0.0
            
        # Stopping penalty to reduce pauses
        stop_penalty = 0.5 if vel_norm < 1e-3 else 0.0
        score = (progress / dt_val) - (0.5 * risk_penalty) - turn_cost - stop_penalty
        
        return True, score

    def get_bearing(robot_pos, goal_pos, time_now):
        nonlocal last_bearing_time, current_bearing
        if time_now - last_bearing_time > 1.0:
            vec = goal_pos[:2] - robot_pos[:2]
            norm = np.linalg.norm(vec)
            if norm > 0.1:
                direction = vec / norm
            else:
                direction = np.array([0.0, 0.0])

            # NFZ-aware bearing adjustment (simple repulsion blend)
            future = robot_pos[:2] + direction * 2.0
            if is_inside_any_nfz(future[0], future[1], margin=0.5):
                rep = combined_nfz_repulsion(robot_pos)
                rep_norm = np.linalg.norm(rep)
                if rep_norm > 0.01:
                    rep_dir = rep / rep_norm
                    direction = 0.4 * direction + 0.6 * rep_dir
                    norm = np.linalg.norm(direction)
                    if norm > 0.01:
                        direction = direction / norm

            current_bearing = np.array([direction[0], direction[1], 0.0])
            last_bearing_time = time_now

        return current_bearing

    def generate_candidates(speed=2.5, current_vel=None):
        candidates = []
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        for a in angles:
            candidates.append(np.array([speed * np.cos(a), speed * np.sin(a), 0.0]))
        for a in angles[::2]: # Half speed
            hm = 0.5 * speed
            candidates.append(np.array([hm * np.cos(a), hm * np.sin(a), 0.0]))
        candidates.append(np.array([0.0, 0.0, 0.0]))
        
        if current_vel is not None and np.linalg.norm(current_vel[:2]) > 0.5:
            vx, vy = current_vel[0], current_vel[1]
            p1 = np.array([-vy, vx, 0.0])
            p2 = np.array([vy, -vx, 0.0])
            p1 = p1 / np.linalg.norm(p1) * speed * 0.8
            p2 = p2 / np.linalg.norm(p2) * speed * 0.8
            candidates.append(p1)
            candidates.append(p2)
        return candidates

    for step in range(max_steps):
        robot_state = get_body_state(physical_model, data, "robot")
        robot_pos = np.array(robot_state["position"][:2])

        if recovery_steps > 0:
            recovery_steps -= 1

        if is_inside_any_nfz(robot_pos[0], robot_pos[1]):
            if not nfz_violated:
                print(f"  [X] NFZ VIOLATION at step {step}!")
            nfz_violated = True

        if step % fps == 0:
            dist = np.linalg.norm(robot_pos - TARGET_POS[:2])
            nfz_d = min_nfz_distance(robot_pos[0], robot_pos[1])
            print(f"  [{step//fps:02d}s] Pos: [{robot_pos[0]:.2f}, {robot_pos[1]:.2f}] | "
                  f"Goal: {dist:.2f}m | NFZ: {nfz_d:.2f}m")
            update_dynamic_obstacles(physical_model, data)

        if step % check_interval == 0 and step > 0:
            movement = np.linalg.norm(robot_pos - last_position)
            if movement < 0.2:
                stuck_counter += 1
                if stuck_counter >= 2:
                    print(f"  [STUCK RECOVERY] random escape")
                    found_esc = False
                    for _ in range(100): # Increased samples from 10 to 100
                        rand_angle = random.uniform(0, 2*np.pi)
                        rand_speed = random.choice([1.0, 2.0, 3.0]) # Try different speeds
                        escape_vel = np.array([rand_speed*np.cos(rand_angle), rand_speed*np.sin(rand_angle), 0.0])
                        escape_vel = np.array([rand_speed*np.cos(rand_angle), rand_speed*np.sin(rand_angle), 0.0])
                        # Use tighter margin (0.15) for emergency escape checks
                        if not would_enter_nfz_swept(robot_pos, escape_vel[:2], dt_check=0.5, margin=0.15):
                            # Check for collision with obstacles
                            # Check for collision with obstacles
                            safe_escape = True
                            predicted_pos_esc = robot_pos[:2] + escape_vel[:2] * 0.5
                            for k in range(1, NUM_TOTAL + 1):
                                try:
                                    bid = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, f"cat{k}")
                                    if bid == -1: continue
                                    opos = data.xpos[bid][:2]
                                    if np.linalg.norm(predicted_pos_esc - opos) < 0.35:
                                        safe_escape = False
                                        break
                                except: pass
                            
                            if safe_escape:
                                data.qvel[robot_dof:robot_dof + 3] = escape_vel
                                current_vel_vector = escape_vel
                                recovery_steps = 20 # Hold velocity
                                print(f"    -> Found escape velocity: {escape_vel[:2]}")
                                found_esc = True
                                break
                    
                    if not found_esc:
                        # Try CREEP ESCAPE (0.5 m/s, super relaxed)
                        for _ in range(50):
                            rand_angle = random.uniform(0, 2*np.pi)
                            escape_vel = np.array([0.5*np.cos(rand_angle), 0.5*np.sin(rand_angle), 0.0])
                            # Use even smaller radius check for creep (0.12 or even less if needed)
                            if not would_enter_nfz_swept(robot_pos, escape_vel[:2], dt_check=0.5, margin=0.15):
                                 # Minimal obstacle check (0.22)
                                safe_escape = True
                                predicted_pos_esc = robot_pos[:2] + escape_vel[:2] * 0.5
                                for k in range(1, NUM_TOTAL + 1):
                                    try:
                                        bid = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, f"cat{k}")
                                        if bid == -1: continue
                                        opos = data.xpos[bid][:2]
                                        if np.linalg.norm(predicted_pos_esc - opos) < 0.22:
                                            safe_escape = False
                                            break
                                    except: pass
                                
                                if safe_escape:
                                    data.qvel[robot_dof:robot_dof + 3] = escape_vel
                                    current_vel_vector = escape_vel
                                    found_esc = True
                                    recovery_steps = 20 
                                    print(f"    -> Found CREEP escape: {escape_vel[:2]}")
                                    break

                    if not found_esc:
                        print("    -> No valid escape found")

                if stuck_counter >= 8:
                    print(f"  [EARLY STOP] Robot stuck")
                    break
            else:
                stuck_counter = 0
            last_position = robot_pos.copy()

        dist_to_target = np.linalg.norm(robot_pos - TARGET_POS[:2])
        if dist_to_target < 0.5:
            print(f"  [SUCCESS] Reached goal at step {step} ({step/fps:.1f}s)!")
            success = True
            break

        for i in range(1, NUM_TOTAL + 1):
            try:
                cat_state = get_body_state(physical_model, data, f"cat{i}")
                cat_dist = np.linalg.norm(
                    np.array(robot_state["position"][:3]) - np.array(cat_state["position"][:3]))
                if step > init_frames and cat_dist < collision_threshold:
                    if not collision:
                        print(f"  [COLLISION] with cat{i} at step {step}!")
                    collision = True
            except:
                pass

        snapshot_t_dt = {"objects": get_all_body_states(physical_model, data)}

        # Decision Logic (20Hz)
        if step > init_frames and (step % 5 == 0) and snapshot_t is not None and recovery_steps == 0:
            robot_pos_3d = np.array(robot_state['position'])
            bearing = get_bearing(robot_pos_3d, TARGET_POS, step / fps)
            candidates = generate_candidates(speed=2.5, current_vel=current_vel_vector)

            best_score = -float('inf')
            best_vel = np.array([0.0, 0.0, 0.0])
            safe_found = False

            for cand in candidates:
                is_safe, score = predict_safety_and_score(
                    robot_pos_3d, cand, snapshot_t["objects"], 0.5, bearing)
                if is_safe:
                    safe_found = True
                    if score > best_score:
                        best_score = score
                        best_vel = cand

            # If no safe move found, try CREEP MODE (slow speed, relaxed margin)
            if not safe_found:
                # Try creep mode: generate slow candidates (0.5 speed)
                creep_candidates = generate_candidates(speed=0.5, current_vel=current_vel_vector)
                for cand in creep_candidates:
                     # Use tighter margin 0.21 (bare minimum clearance)
                     is_safe, score = predict_safety_and_score(
                        robot_pos_3d[:2], cand, snapshot_t["objects"], 0.5, bearing, static_margin=0.21)
                     if is_safe:
                         safe_found = True
                         # Penalty for creeping
                         score -= 5.0
                         if score > best_score:
                             best_score = score
                             best_vel = cand

            if not safe_found:
                # Try stop
                is_safe_stop, _ = predict_safety_and_score(
                    robot_pos_3d[:2], np.array([0.,0.,0.]), snapshot_t["objects"], 0.5, bearing)
                if is_safe_stop:
                    best_vel = np.array([0.0, 0.0, 0.0])
                else:
                    best_vel = np.array([0.0, 0.0, 0.0]) # Emergency brake

            data.qvel[robot_dof:robot_dof + 3] = best_vel
            current_vel_vector = best_vel
            
        snapshot_t = snapshot_t_dt

        for i in range(NUM_DYNAMIC + 1, NUM_TOTAL + 1):
            try:
                body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
                if body_id != -1:
                    dof_start = physical_model.body_dofadr[body_id]
                    data.qvel[dof_start:dof_start + 3] = [0.0, 0.0, 0.0]
            except:
                pass

        mujoco.mj_step(physical_model, data)
        renderer.update_scene(data, camera="top")
        frame = cv2.cvtColor(np.flipud(renderer.render()), cv2.COLOR_RGB2BGR)
        yield frame, step

    yield None, {"success": success, "collision": collision, "nfz_violated": nfz_violated,
                 "steps": step, "time": step/fps}


# ============================================================
# MAIN
# ============================================================
def run_medium_nfz_experiment(method='APEX', obstacle_speed=1.0):
    """Run Medium NFZ experiment (2 zones)."""
    env_path = os.path.join(base_path, "env/medium_nfz_env.xml")
    with open(env_path, 'r') as f:
        xml_string = f.read()

    physical_model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(physical_model)
    renderer = mujoco.Renderer(physical_model, height=480, width=640)

    randomize_all_obstacles(physical_model, data)
    mujoco.mj_step(physical_model, data)

    for i in range(1, NUM_DYNAMIC + 1):
        try:
            body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
            if body_id != -1:
                dof_start = physical_model.body_dofadr[body_id]
                vx, vy = get_random_velocity(obstacle_speed)
                data.qvel[dof_start], data.qvel[dof_start + 1] = vx, vy
        except:
            pass

    for i in range(NUM_DYNAMIC + 1, NUM_TOTAL + 1):
        try:
            body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
            if body_id != -1:
                dof_start = physical_model.body_dofadr[body_id]
                data.qvel[dof_start:dof_start + 3] = [0.0, 0.0, 0.0]
        except:
            pass

    fps = 100
    dt = 1.0 / fps
    width, height = 640, 480
    output_dir = os.path.abspath(os.path.join(base_path, "../../videos"))
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.join(output_dir, f"medium_nfz_{method}_{timestamp}.mp4")
    video_writer = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    max_steps = 30 * fps

    method_names = {
        "APEX": "Expt 1: APEX Original (DiffGraphormer + LLM)",
        "PULSE": "Expt 2: Pulse / 3-Tier Strategy",
        "HYBRID": "Expt 3: Hybrid GNN-LLM"
    }

    nfz_desc = " | ".join(f"{z['name']}: x=[{z['x_min']},{z['x_max']}] y=[{z['y_min']},{z['y_max']}]"
                          for z in NFZ_ZONES)

    print(f"\n{'='*60}")
    print(f"MEDIUM NFZ EXPERIMENT - {method_names.get(method, method)}")
    print(f"{'='*60}")
    print(f"  NFZ Zones: {len(NFZ_ZONES)}")
    print(f"    {nfz_desc}")
    print(f"  Obstacles: {NUM_DYNAMIC} dynamic + {NUM_STATIC} static = {NUM_TOTAL}")
    print(f"  Robot: {START_POS[:2]} -> Goal: {TARGET_POS[:2]}")
    print(f"  Max Duration: {max_steps/fps:.0f}s")
    print(f"{'='*60}")

    if method == 'APEX':
        generator = run_apex_method(physical_model, data, renderer, fps, dt, max_steps)
    elif method == 'PULSE':
        generator = run_pulse_method(physical_model, data, renderer, fps, dt, max_steps)
    elif method == 'HYBRID':
        generator = run_hybrid_method(physical_model, data, renderer, fps, dt, max_steps)
    else:
        print(f"Unknown method: {method}")
        return None

    results = None
    for output in generator:
        frame, info = output
        if frame is None:
            results = info
        else:
            video_writer.write(cv2.resize(frame, (width, height)))

    video_writer.release()

    print(f"\n{'='*60}")
    print(f"RESULTS - {method}")
    print(f"{'='*60}")
    if results:
        # Fix: Check distance to goal, not just steps
        # Use simple success check
        is_success = results.get("success", False)
        print(f"  Reached Goal: {is_success}")
        print(f"  Collision: {results['collision']}")
        print(f"  NFZ Violated: {results['nfz_violated']}")
        print(f"  Time: {results['time']:.1f}s")
        print(f"  Video: {file_name}")

        # # CONDITIONAL DELETION
        # if not is_success:
        #     print("  [INFO] Robot did not reach goal. Deleting video...")
        #     try:
        #         os.remove(file_name)
        #         print("  [INFO] Video deleted.")
        #     except Exception as e:
        #         print(f"  [ERROR] Could not delete video: {e}")
    
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("WARNING: OPENAI_API_KEY not found. APEX method requires it.")

    parser = argparse.ArgumentParser(description="Medium NFZ Experiment - 2 No-Fly Zones")
    parser.add_argument("--method", type=str, default="APEX",
                        choices=["APEX", "PULSE", "HYBRID"],
                        help="Experiment method: APEX, PULSE, or HYBRID")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Dynamic obstacle speed")
    args = parser.parse_args()

    run_medium_nfz_experiment(method=args.method, obstacle_speed=args.speed)
