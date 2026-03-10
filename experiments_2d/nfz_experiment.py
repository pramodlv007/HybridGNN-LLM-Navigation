"""
No-Fly Zone (NFZ) Experiment
==============================
Simulation environment with a No-Enter Zone (red rectangle) in the center.
The robot must navigate from start (-3.6, -3.6) to goal (3.6, 3.6)
avoiding BOTH obstacles AND the NFZ.

Supports 3 experiment methods:
  --method APEX    : Experiment 1 - APEX Original (DiffGraphormer + LLM)
  --method PULSE   : Experiment 2 - Pulse / 3-Tier Strategy (GNN tiers)
  --method HYBRID  : Experiment 3 - Hybrid GNN-LLM Controller

NFZ Rules:
  - NFZ is an axis-aligned rectangle: x in [-1.2, 1.2], y in [-1.2, 1.2]
  - Robot must NEVER enter this zone (Drone-like strictness)
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
import argparse
import math

# Adjust sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.cat_expt.utils.cat_game_agent import LLM_Agent
from experiments.cat_expt.utils.APEX import APEX
from experiments.cat_expt.model.graphormer import DiffGraphormer
from experiments.cat_expt.model.local_risk_gnn import LocalRiskGNN
from experiments.cat_expt.utils.mujoco_simulator import get_body_state, get_all_body_states

# ============================================================
# CONSTANTS
# ============================================================
START_POS = np.array([-3.6, -3.6, 0.12])
TARGET_POS = np.array([3.6, 3.6, 0.12])
NUM_DYNAMIC = 10   # cat1-cat10
NUM_STATIC = 15    # cat11-cat25
NUM_TOTAL = NUM_DYNAMIC + NUM_STATIC

# NFZ boundaries (center of arena)
NFZ_X_MIN, NFZ_X_MAX = -1.2, 1.2
NFZ_Y_MIN, NFZ_Y_MAX = -1.2, 1.2
NFZ_CENTER = np.array([0.0, 0.0])
NFZ_MARGIN = 0.3  # Safety margin around NFZ

base_path = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_path, "env/available_move.json"), 'r') as f:
    available_move = json.load(f)


# ============================================================
# NFZ UTILITY FUNCTIONS
# ============================================================
def is_inside_nfz(x, y, margin=0.0):
    """Check if point (x, y) is inside the NFZ (with optional margin)."""
    return (NFZ_X_MIN - margin <= x <= NFZ_X_MAX + margin and
            NFZ_Y_MIN - margin <= y <= NFZ_Y_MAX + margin)


def nfz_distance(x, y):
    """Distance from point (x,y) to nearest NFZ boundary. Negative = inside."""
    dx = max(NFZ_X_MIN - x, 0, x - NFZ_X_MAX)
    dy = max(NFZ_Y_MIN - y, 0, y - NFZ_Y_MAX)
    if dx == 0 and dy == 0:
        # Inside NFZ - return negative distance to nearest edge
        return -min(x - NFZ_X_MIN, NFZ_X_MAX - x, y - NFZ_Y_MIN, NFZ_Y_MAX - y)
    return np.sqrt(dx**2 + dy**2)


def nfz_repulsion(robot_pos, strength=5.0, influence_radius=2.0):
    """
    Compute repulsion vector pushing robot away from NFZ.
    Returns a 2D repulsion vector.
    """
    x, y = robot_pos[0], robot_pos[1]
    dist = nfz_distance(x, y)

    if dist > influence_radius:
        return np.array([0.0, 0.0])

    # Direction away from NFZ center
    diff = robot_pos[:2] - NFZ_CENTER
    norm = np.linalg.norm(diff)
    if norm < 0.01:
        # If exactly at center, push diagonally
        diff = np.array([1.0, 1.0])
        norm = np.linalg.norm(diff)

    direction = diff / norm

    # Stronger repulsion when closer (inverse distance)
    if dist <= 0:
        # Inside NFZ - MAXIMUM repulsion
        magnitude = strength * 3.0
    else:
        magnitude = strength * (1.0 / (dist + 0.1))

    return direction * magnitude


def would_enter_nfz_swept(current_pos, velocity, dt_check=1.0, steps=5, margin=None):
    """Check if swept path enters NFZ."""
    if margin is None:
        margin = NFZ_MARGIN
        
    start = current_pos[:2]
    end   = start + np.array(velocity[:2]) * dt_check
    
    for i in range(steps + 1):
        t = i / steps
        pt = start + (end - start) * t
        if is_inside_nfz(pt[0], pt[1], margin=margin):
            return True
    return False

def calculate_ttc(pos1, vel1, pos2, vel2, radius_sum):
    rel_pos = pos2 - pos1
    rel_vel = vel2 - vel1
    dv = np.dot(rel_vel, rel_vel)
    if dv < 1e-6: return float('inf')
    dp = np.dot(rel_pos, rel_vel)
    if dp >= 0: return float('inf') 
    dist_sq = np.dot(rel_pos, rel_pos)
    rad_sq = radius_sum * radius_sum
    discr = dp*dp - dv*(dist_sq - rad_sq)
    if discr < 0: return float('inf')
    t = -(dp + np.sqrt(discr)) / dv
    return t if t >= 0 else float('inf')

# ============================================================
# OBSTACLE HELPERS
# ============================================================
def get_random_velocity(speed=1.0):
    angle = random.uniform(0, 2 * np.pi)
    return speed * np.cos(angle), speed * np.sin(angle)


def randomize_all_obstacles(model, data):
    """Randomize positions of all obstacles, avoiding NFZ, start, and goal."""
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

            # Must be outside NFZ (with margin), away from start/goal, and away from others
            if (not is_inside_nfz(rx, ry, margin=0.5) and
                np.linalg.norm(pos - START_POS[:2]) > 1.5 and
                np.linalg.norm(pos - TARGET_POS[:2]) > 1.5 and
                all(np.linalg.norm(pos - p) > 0.5 for p in placed)):
                z = 0.1 if i <= NUM_DYNAMIC else 0.08
                data.qpos[dof_start:dof_start + 3] = [rx, ry, z]
                placed.append(pos)
                break

    print(f"  Randomized {len(placed)} obstacles (avoiding NFZ)")
    return placed


def update_dynamic_obstacles(model, data, speed=1.0):
    """Update dynamic obstacles with random movement, keeping them outside NFZ."""
    for i in range(1, NUM_DYNAMIC + 1):
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
            if body_id == -1:
                continue
            dof_start = model.body_dofadr[body_id]
            pos = data.xpos[body_id]
            vx, vy = data.qvel[dof_start], data.qvel[dof_start + 1]

            # Bounce off walls
            if pos[0] < -3.5: vx = abs(vx)
            elif pos[0] > 3.5: vx = -abs(vx)
            if pos[1] < -3.5: vy = abs(vy)
            elif pos[1] > 3.5: vy = -abs(vy)

            # Bounce off NFZ boundaries
            if is_inside_nfz(pos[0], pos[1], margin=0.3):
                repulsion = nfz_repulsion(pos)
                vx += repulsion[0] * 0.5
                vy += repulsion[1] * 0.5

            if random.random() < 0.3:
                vx, vy = get_random_velocity(speed)
            data.qvel[dof_start], data.qvel[dof_start + 1] = vx, vy
        except Exception:
            pass


# ============================================================
# METHOD 1: APEX ORIGINAL (Restored)
# ============================================================
def run_apex_method(physical_model, data, renderer, fps, dt, max_steps):
    """Experiment 1: APEX Original with NFZ-aware LLM prompts."""
    agent = LLM_Agent(model='gpt-4o-mini')

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

            # NFZ SAFETY CHECK: reject moves that would enter NFZ
            robot_state = get_body_state(physical_model, data, "robot")
            robot_pos = np.array(robot_state["position"][:3])
            # Use swept check if available, else simple
            if would_enter_nfz_swept(robot_pos, vel, dt_check=0.8):
                print("    [NFZ] Blocked move that would enter NFZ!")
                repulsion = nfz_repulsion(robot_pos)
                vel = [repulsion[0] * 1.5, repulsion[1] * 1.5, 0.0]
            data.qvel[robot_dof:robot_dof + 3] = vel

    snapshot_t, snapshot_t_dt = None, None

    for step in range(max_steps):
        robot_state = get_body_state(physical_model, data, "robot")
        robot_pos = np.array(robot_state["position"][:2])

        # NFZ check
        if is_inside_nfz(robot_pos[0], robot_pos[1]):
            if not nfz_violated:
                print(f"  [X] NFZ VIOLATION at step {step}!")
            nfz_violated = True

        # Goal check
        dist_to_target = np.linalg.norm(robot_pos - TARGET_POS[:2])
        if dist_to_target < 0.5:
            print(f"  [SUCCESS] Reached goal at step {step} ({step/fps:.1f}s)!")
            is_success = True
            break

        # Collision check
        for i in range(1, NUM_TOTAL + 1):
            try:
                cat_state = get_body_state(physical_model, data, f"cat{i}")
                cat_dist = np.linalg.norm(
                    np.array(robot_state["position"][:3]) - np.array(cat_state["position"][:3]))
                if step > init_frames and cat_dist < collision_threshold:
                    if not collision:
                        print(f"  [COLLISION] Collision with cat{i} at step {step}!")
                    collision = True
            except:
                pass

        snapshot_t_dt = {"objects": get_all_body_states(physical_model, data)}

        # APEX decision with NFZ-aware prompts
        if step > init_frames and frames_left <= 0:
            if snapshot_t:
                triggered, move, action_valid = apex.run(
                    snapshot_t, snapshot_t_dt, dt, physical_model, data, step)

                if not triggered or move == "stay":
                    nfz_dist = nfz_distance(robot_pos[0], robot_pos[1])
                    nfz_warning = ""
                    if nfz_dist < 2.0:
                        nfz_warning = (
                            f"\nCRITICAL: NO-FLY ZONE ahead! "
                            f"NFZ is at x=[-1.2, 1.2], y=[-1.2, 1.2]. "
                            f"Distance to NFZ: {nfz_dist:.2f}m. "
                            f"Navigate AROUND it."
                        )

                    state_description = (
                        f"Robot Position: {robot_state['position'][:2]}\n"
                        f"Target: {TARGET_POS[:2]}\n"
                        f"Distance to target: {dist_to_target:.2f}m\n"
                        f"MISSION: Navigate to goal AVOIDING all obstacles "
                        f"AND the No-Fly Zone at center."
                        f"{nfz_warning}\n"
                        f"Choose best move to reach target."
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
                if would_enter_nfz_swept(robot_pos_3d, vel, dt_check=0.8):
                    print("    [NFZ] Blocked move that would enter NFZ!")
                    repulsion = nfz_repulsion(robot_pos_3d)
                    vel = [repulsion[0] * 1.5, repulsion[1] * 1.5, 0.0]
                data.qvel[robot_dof:robot_dof + 3] = vel
        else:
            data.qvel[robot_dof:robot_dof + 3] = [0.0, 0.0, 0.0]

        # Update dynamic obstacles
        if step % fps == 0:
            update_dynamic_obstacles(physical_model, data)

        mujoco.mj_step(physical_model, data)

        renderer.update_scene(data, camera="top")
        frame = cv2.cvtColor(np.flipud(renderer.render()), cv2.COLOR_RGB2BGR)
        yield frame, {"success": is_success, "collision": collision, "nfz_violated": nfz_violated,
                      "steps": step, "time": step/fps}

    yield None, {"success": is_success, "collision": collision, "nfz_violated": nfz_violated,
                 "steps": step, "time": step/fps}


# ============================================================
# METHOD 3: HYBRID GNN-LLM (Upgraded)
# ============================================================
def run_hybrid_method(physical_model, data, renderer, fps, dt, max_steps):
    """Experiment 3: Hybrid GNN-LLM with Advanced Drone-Like Logic."""
    robot_body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    robot_dof = physical_model.body_dofadr[robot_body_id]

    collision = False
    nfz_violated = False
    success = False
    collision_threshold = 0.2
    
    snapshot_t = None
    last_bearing_time = -10.0
    current_bearing = np.array([1.0, 1.0, 0.0])  # Default: toward goal

    stuck_counter = 0
    last_position = START_POS[:2].copy()
    check_interval = int(0.5 * fps)  # Fast check
    
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
        nfz_d = nfz_distance(predicted_pos[0], predicted_pos[1])
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
        min_ttc = float('inf')
        
        for i in range(1, NUM_TOTAL + 1):
            try:
                body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
                if body_id == -1: continue
                
                obs_pos = data.xpos[body_id][:2]
                is_dynamic = (i <= NUM_DYNAMIC)
                
                if is_dynamic:
                    dof_adr = physical_model.body_dofadr[body_id]
                    obs_v = data.qvel[dof_adr:dof_adr+2]
                    radius_sum = 0.12 + 0.12 + 0.15 # Safe radius
                    
                    rel_v = rob_vel - obs_v
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
                    # Hard veto at margin
                    if dist_to_static < static_margin:
                        return False, -float('inf')

                    # Swept proximity check
                    for t_frac in [0.2, 0.5, 0.8, 1.0]:
                        sample_pos = curr_pos + vel[:2] * dt_val * t_frac
                        dist_s = np.linalg.norm(sample_pos - obs_pos)
                        if dist_s < static_margin:
                            return False, -float('inf')
                    
                    if dist_to_static < 0.40:
                         risk_penalty += 0.5 / (dist_to_static + 0.05)
            except:
                pass

        # 3. Score = Progress + Penalties
        advancement = np.dot(vel[:2], bearing_vec[:2])
        score = advancement - risk_penalty
        return True, score

    def get_bearing(robot_pos, goal_pos, time_now):
        """Strategic bearing toward goal, NFZ-aware."""
        nonlocal last_bearing_time, current_bearing

        if time_now - last_bearing_time > 0.5: # Faster update
            vec = goal_pos[:2] - robot_pos[:2]
            norm = np.linalg.norm(vec)
            if norm > 0.1:
                direction = vec / norm
            else:
                direction = np.array([0.0, 0.0])

            # Simple predictive bearing adjustment around NFZ
            future = robot_pos[:2] + direction * 2.0
            if is_inside_nfz(future[0], future[1], margin=0.5):
                rep = nfz_repulsion(robot_pos)
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
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False) # 24 angles
        cands = [np.array([speed * np.cos(a), speed * np.sin(a), 0.0]) for a in angles]
        
        # Momentum candidate
        if current_vel is not None and np.linalg.norm(current_vel) > 0.1:
            cands.append(current_vel)
        return cands

    for step in range(max_steps):
        robot_state = get_body_state(physical_model, data, "robot")
        robot_pos = np.array(robot_state["position"][:2])

        if recovery_steps > 0:
            recovery_steps -= 1

        if is_inside_nfz(robot_pos[0], robot_pos[1]):
            if not nfz_violated:
                print(f"  [X] NFZ VIOLATION at step {step}!")
            nfz_violated = True

        if step % fps == 0:
            dist = np.linalg.norm(robot_pos - TARGET_POS[:2])
            nfz_d = nfz_distance(robot_pos[0], robot_pos[1])
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
                    for _ in range(100):
                        rand_angle = random.uniform(0, 2*np.pi)
                        rand_speed = random.choice([1.0, 2.0, 3.0])
                        escape_vel = np.array([rand_speed*np.cos(rand_angle), rand_speed*np.sin(rand_angle), 0.0])
                        
                        # Use tighter margin (0.15) for emergency escapes
                        if not would_enter_nfz_swept(robot_pos, escape_vel[:2], dt_check=0.5, margin=0.15):
                            safe_escape = True
                            # Check obstacle proximity
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
                                recovery_steps = 20
                                print(f"    -> Found escape velocity: {escape_vel[:2]}")
                                found_esc = True
                                break
                    
                    if not found_esc:
                        # Try CREEP ESCAPE
                        for _ in range(50):
                            rand_angle = random.uniform(0, 2*np.pi)
                            escape_vel = np.array([0.5*np.cos(rand_angle), 0.5*np.sin(rand_angle), 0.0])
                            if not would_enter_nfz_swept(robot_pos, escape_vel[:2], dt_check=0.5, margin=0.15):
                                 # Minimal obstacle check
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
                        print(f"  [COLLISION] Collision with cat{i} at step {step}!")
                    collision = True
            except:
                pass

        snapshot_t_dt = {"objects": get_all_body_states(physical_model, data)}

        # HYBRID DECISION
        if recovery_steps == 0 and snapshot_t is not None:
            bearing = get_bearing(robot_pos, TARGET_POS, step / fps)
            candidates = generate_candidates(speed=2.5, current_vel=current_vel_vector)

            best_score = -float('inf')
            best_vel = np.array([0.0, 0.0, 0.0])
            safe_found = False
            
            for cand in candidates:
                is_safe, score = predict_safety_and_score(
                    robot_pos[:2], cand, snapshot_t["objects"], 0.5, bearing)
                
                if is_safe:
                    safe_found = True
                    if score > best_score:
                        best_score = score
                        best_vel = cand
            
            if not safe_found:
                 # Try Creep Mode
                creep_candidates = generate_candidates(speed=0.5, current_vel=current_vel_vector)
                for cand in creep_candidates:
                     is_safe, score = predict_safety_and_score(
                        robot_pos[:2], cand, snapshot_t["objects"], 0.5, bearing, static_margin=0.21)
                     if is_safe:
                         safe_found = True
                         score -= 5.0 # Penalty
                         if score > best_score:
                             best_score = score
                             best_vel = cand

            if not safe_found:
                # Brake
                is_safe_stop, _ = predict_safety_and_score(
                    robot_pos[:2], np.array([0.,0.,0.]), snapshot_t["objects"], 0.5, bearing)
                if is_safe_stop:
                    best_vel = np.array([0.0, 0.0, 0.0])
            
            data.qvel[robot_dof:robot_dof + 3] = best_vel
            current_vel_vector = best_vel

        snapshot_t = snapshot_t_dt

        # Freeze static obstacles
        for i in range(NUM_DYNAMIC + 1, NUM_TOTAL + 1):
            pass # Keep static static

        mujoco.mj_step(physical_model, data)
        renderer.update_scene(data, camera="top")
        frame = cv2.cvtColor(np.flipud(renderer.render()), cv2.COLOR_RGB2BGR)
        yield frame, {"success": success, "collision": collision, "nfz_violated": nfz_violated,
                      "steps": step, "time": step/fps}

    yield None, {"success": success, "collision": collision, "nfz_violated": nfz_violated,
                 "steps": step, "time": step/fps}


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def run_nfz_experiment(method='APEX', obstacle_speed=1.0):
    """Run NFZ experiment with the specified method."""

    env_path = os.path.join(base_path, "env/nfz_env.xml")
    with open(env_path, 'r') as f:
        xml_string = f.read()

    physical_model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(physical_model)
    renderer = mujoco.Renderer(physical_model, height=480, width=640)

    # Randomize obstacles (avoiding NFZ)
    randomize_all_obstacles(physical_model, data)
    mujoco.mj_step(physical_model, data)

    # Init dynamic obstacle velocities
    for i in range(1, NUM_DYNAMIC + 1):
        try:
            body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
            if body_id != -1:
                dof_start = physical_model.body_dofadr[body_id]
                vx, vy = get_random_velocity(obstacle_speed)
                data.qvel[dof_start], data.qvel[dof_start + 1] = vx, vy
        except:
            pass

    # Freeze static obstacles
    for i in range(NUM_DYNAMIC + 1, NUM_TOTAL + 1):
        try:
            body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
            if body_id != -1:
                dof_start = physical_model.body_dofadr[body_id]
                data.qvel[dof_start:dof_start + 3] = [0.0, 0.0, 0.0]
        except:
            pass

    # Video
    fps = 100
    dt = 1.0 / fps
    width, height = 640, 480
    output_dir = os.path.abspath(os.path.join(base_path, "../../videos"))
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.join(output_dir, f"nfz_{method}_{timestamp}.mp4")
    video_writer = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    max_steps = 30 * fps  # 30 seconds

    print(f"\n{'='*60}")
    print(f"NFZ EXPERIMENT - {method}")
    print(f"{'='*60}")
    
    # Select method
    if method == 'APEX':
        generator = run_apex_method(physical_model, data, renderer, fps, dt, max_steps)
    elif method == 'HYBRID':
        generator = run_hybrid_method(physical_model, data, renderer, fps, dt, max_steps)
    else:
        # Placeholder for other methods (not updated for this task)
        generator = run_hybrid_method(physical_model, data, renderer, fps, dt, max_steps)

    results = None
    for output in generator:
        frame, info = output
        if frame is None:
            results = info
        else:
            video_writer.write(cv2.resize(frame, (width, height)))
            results = info

    video_writer.release()

    print(f"\n{'='*60}")
    print(f"RESULTS - {method}")
    print(f"{'='*60}")
    if results:
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

    parser = argparse.ArgumentParser(description="NFZ Experiment - Navigate around No-Fly Zone")
    parser.add_argument("--method", type=str, default="APEX",
                        choices=["APEX", "PULSE", "HYBRID"],
                        help="Experiment method: APEX, PULSE, or HYBRID")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Dynamic obstacle speed")
    args = parser.parse_args()

    run_nfz_experiment(method=args.method, obstacle_speed=args.speed)
