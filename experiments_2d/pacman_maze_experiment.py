"""
Pac-Man Maze Experiment
=======================
MuJoCo-based Pac-Man maze with 3 methods: APEX, PULSE, HYBRID.
Runs all 3 and prints comparison stats.
"""

import time
import json
import random
import math
from datetime import datetime
import mujoco
import numpy as np
import cv2
import torch
import os
import sys
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.cat_game_agent import LLM_Agent
from utils.APEX import APEX
from model.graphormer import DiffGraphormer
from utils.mujoco_simulator import get_body_state, get_all_body_states

# ============================================================
# CONSTANTS
# ============================================================
START_POS  = np.array([-3.6, -3.6, 0.12])
TARGET_POS = np.array([ 3.6,  3.6, 0.12])
NUM_DYNAMIC = 6
NUM_STATIC  = 9
NUM_TOTAL   = NUM_DYNAMIC + NUM_STATIC

base_path = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_path, "env/available_move.json"), 'r') as f:
    available_move = json.load(f)


# ============================================================
# WALL / OBSTACLE SAFETY CONSTANTS
# ============================================================
WALL_SAFE_MARGIN = 0.25        # min clearance between robot centre and wall surface
ROBOT_RADIUS     = 0.12
WALL_REPULSION_STRENGTH = 3.0  # multiplier for wall push-back


# ============================================================
# WALL HELPERS — parse walls from MuJoCo model
# ============================================================
def get_wall_segments(model):
    """Return list of (center_xy, half_size_xy) for every box geom
    whose name starts with 'wall_' or 'maze_'."""
    walls = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name and (name.startswith("wall_") or name.startswith("maze_")):
            pos = model.geom_pos[gid][:2].copy()
            hs  = model.geom_size[gid][:2].copy()   # half-sizes for box
            walls.append((pos, hs))
    return walls


def wall_distance(robot_xy, wall_center, wall_half):
    """Signed distance from robot centre to the nearest surface of an
    axis-aligned box.  Negative → inside the box."""
    dx = max(abs(robot_xy[0] - wall_center[0]) - wall_half[0], 0.0)
    dy = max(abs(robot_xy[1] - wall_center[1]) - wall_half[1], 0.0)
    return math.sqrt(dx*dx + dy*dy)


def wall_repulsion_vec(robot_xy, walls, margin=WALL_SAFE_MARGIN):
    """Return a 2-D repulsion vector pushing the robot away from
    any wall surface closer than *margin*."""
    rep = np.array([0.0, 0.0])
    for center, hs in walls:
        # closest point on box surface to robot
        cx = np.clip(robot_xy[0], center[0] - hs[0], center[0] + hs[0])
        cy = np.clip(robot_xy[1], center[1] - hs[1], center[1] + hs[1])
        diff = robot_xy - np.array([cx, cy])
        dist = np.linalg.norm(diff)
        if dist < margin and dist > 1e-4:
            rep += (diff / dist) * (margin - dist) / margin * WALL_REPULSION_STRENGTH
        elif dist < 1e-4:
            # Robot centre is inside the wall — push toward goal
            rep += (TARGET_POS[:2] - robot_xy) * WALL_REPULSION_STRENGTH
    return rep


def wall_risk_for_candidate(robot_xy, velocity_xy, walls, dt_val=0.3,
                            margin=WALL_SAFE_MARGIN):
    """Return a penalty score for *velocity* based on how close the
    predicted next position will be to any wall."""
    next_pos = robot_xy + velocity_xy * dt_val
    penalty = 0.0
    for center, hs in walls:
        d = wall_distance(next_pos, center, hs)
        if d < margin:
            # strong penalty as we approach the wall
            penalty += (margin - d) / margin * 25.0
    return penalty


def clamp_pos_away_from_walls(robot_xy, walls, margin=WALL_SAFE_MARGIN):
    """If the robot is already within *margin* of a wall, return a
    corrective velocity (2-D) pushing it back into free space."""
    correction = np.array([0.0, 0.0])
    for center, hs in walls:
        cx = np.clip(robot_xy[0], center[0] - hs[0], center[0] + hs[0])
        cy = np.clip(robot_xy[1], center[1] - hs[1], center[1] + hs[1])
        diff = robot_xy - np.array([cx, cy])
        dist = np.linalg.norm(diff)
        if dist < margin:
            if dist > 1e-4:
                push = (diff / dist) * (margin - dist) * 5.0
            else:
                push = (TARGET_POS[:2] - robot_xy)
                n = np.linalg.norm(push)
                if n > 0.1: push = push / n * 3.0
            correction += push
    return correction


# ============================================================
# SAFETY & SCORING CONSTANTS
# ============================================================
TTC_THRESHOLD   = 1.2    # seconds to look ahead for collision (hard shield)
RISK_DIST_HORIZON = 1.6  # seconds for soft risk cost
ETA_Risk_Weight = 2.0
ETA_Turn_Weight = 0.5


# ============================================================
# ADVANCED SAFETY HELPERS (TTC + Geometry)
# ============================================================
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
        # Relative velocity is zero
        if c <= 0: return 0.0  # Already colliding
        return float('inf')
        
    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return float('inf') # No real roots -> no collision
        
    sqrt_d = math.sqrt(discriminant)
    t1 = (-b - sqrt_d) / (2*a)
    t2 = (-b + sqrt_d) / (2*a)
    
    # We want the smallest positive time
    if t1 < 0 and t2 < 0: return float('inf')
    if t1 < 0: return t2
    return t1


def check_wall_swept_collision(robot_xy, velocity_xy, dt_check, walls, radius):
    """
    Check if the robot's swept segment (from current to current+vel*dt) 
    intersects any wall box (expanded by radius).
    Strict geometric filter.
    """
    start = robot_xy
    end   = robot_xy + velocity_xy * dt_check
    
    # Bounding box of the swept path for quick rejection
    min_x = min(start[0], end[0]) - radius
    max_x = max(start[0], end[0]) + radius
    min_y = min(start[1], end[1]) - radius
    max_y = max(start[1], end[1]) + radius
    
    for center, hs in walls:
        # Wall bounds
        w_min_x = center[0] - hs[0]
        w_max_x = center[0] + hs[0]
        w_min_y = center[1] - hs[1]
        w_max_y = center[1] + hs[1]
        
        # Broad phase
        if (max_x < w_min_x or min_x > w_max_x or
            max_y < w_min_y or min_y > w_max_y):
            continue
            
        # Detailed sweep check: 
        # Expand wall by robot radius and check segment intersection
        # Simplified: check start, end, and midpoint against expanded wall
        # (Exact swept-circle vs AABB is simpler here by just sampling)
        steps = max(2, int(np.linalg.norm(velocity_xy * dt_check) / (radius * 0.5)))
        for i in range(steps + 1):
            t = i / steps
            pt = start + (end - start) * t
            if (pt[0] > w_min_x - radius and pt[0] < w_max_x + radius and
                pt[1] > w_min_y - radius and pt[1] < w_max_y + radius):
                return True # Collision detected
                
    return False


def get_random_velocity(speed=1.0):
    angle = random.uniform(0, 2 * math.pi)
    return speed * math.cos(angle), speed * math.sin(angle)


def update_dynamic_obstacles(model, data, speed=1.0):
    for i in range(1, NUM_DYNAMIC + 1):
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
            if body_id == -1: continue
            dof_start = model.body_dofadr[body_id]
            pos = data.xpos[body_id]
            vx, vy = data.qvel[dof_start], data.qvel[dof_start + 1]

            if pos[0] < -3.5: vx = abs(vx)
            elif pos[0] > 3.5: vx = -abs(vx)
            if pos[1] < -3.5: vy = abs(vy)
            elif pos[1] > 3.5: vy = -abs(vy)

            if random.random() < 0.15:
                vx, vy = get_random_velocity(speed)
            data.qvel[dof_start], data.qvel[dof_start + 1] = vx, vy
        except Exception:
            pass


# ============================================================
# METHOD 1: APEX (GNN Pathfinding)
# ============================================================
def run_apex_method(physical_model, data, renderer, fps, dt, max_steps):
    agent = LLM_Agent(model='gpt-4o')
    danger_model = DiffGraphormer(in_feats=7, edge_feat_dim=3, hidden_dim=32, num_heads=4, dropout=0.3)
    danger_model.load_state_dict(torch.load(
        os.path.join(base_path, 'model/diffgraphormer_physics.pt'), map_location='cpu'))
    danger_model.eval()

    apex = APEX(graphormer_model=danger_model, physics_simulator="mujoco",
                llm_agent=agent, dt=dt, available_move=available_move)

    walls = get_wall_segments(physical_model)
    robot_body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    robot_dof = physical_model.body_dofadr[robot_body_id]

    collision = False
    collision_threshold = 0.22
    init_frames = int(fps / 2)
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
            # --- wall-safe velocity ---
            vel = np.array(current_action["velocity"], dtype=float)
            robot_state_now = get_body_state(physical_model, data, "robot")
            rp = np.array(robot_state_now["position"][:2])
            w_penalty = wall_risk_for_candidate(rp, vel[:2], walls)
            if w_penalty > 5.0:
                # Reject this move — apply wall repulsion instead
                rep = wall_repulsion_vec(rp, walls)
                goal_dir = TARGET_POS[:2] - rp
                gn = np.linalg.norm(goal_dir)
                if gn > 0.1: goal_dir /= gn
                safe_dir = 0.4 * goal_dir + 0.6 * (rep / max(np.linalg.norm(rep), 0.1))
                sn = np.linalg.norm(safe_dir)
                if sn > 0.1: safe_dir /= sn
                vel[:2] = safe_dir * 2.0
            data.qvel[robot_dof:robot_dof + 3] = vel

    snapshot_t, snapshot_t_dt = None, None

    for step in range(max_steps):
        robot_state = get_body_state(physical_model, data, "robot")
        robot_pos = np.array(robot_state["position"][:2])

        if np.linalg.norm(robot_pos - TARGET_POS[:2]) < 0.5:
            print(f"  [SUCCESS] Reached goal at step {step} ({step/fps:.1f}s)!")
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
                    state_description = (
                        f"Robot Position: {robot_state['position'][:2]}\n"
                        f"Target: {TARGET_POS[:2]}\n"
                        f"Distance to target: {np.linalg.norm(robot_pos - TARGET_POS[:2]):.2f}m\n"
                        f"This is a MAZE environment with walls. Navigate corridors to reach goal.\n"
                        f"IMPORTANT: Do NOT move into walls. Maintain safe distance from walls.\n"
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
                vel = np.array(current_action["velocity"], dtype=float)
                w_penalty = wall_risk_for_candidate(robot_pos, vel[:2], walls)
                if w_penalty > 5.0:
                    rep = wall_repulsion_vec(robot_pos, walls)
                    goal_dir = TARGET_POS[:2] - robot_pos
                    gn = np.linalg.norm(goal_dir)
                    if gn > 0.1: goal_dir /= gn
                    safe_dir = 0.4 * goal_dir + 0.6 * (rep / max(np.linalg.norm(rep), 0.1))
                    sn = np.linalg.norm(safe_dir)
                    if sn > 0.1: safe_dir /= sn
                    vel[:2] = safe_dir * 2.0
                data.qvel[robot_dof:robot_dof + 3] = vel
        else:
            # Even when idle, apply wall correction if too close
            correction = clamp_pos_away_from_walls(robot_pos, walls)
            if np.linalg.norm(correction) > 0.1:
                data.qvel[robot_dof:robot_dof + 3] = [correction[0], correction[1], 0.0]
            else:
                data.qvel[robot_dof:robot_dof + 3] = [0.0, 0.0, 0.0]

        if step % fps == 0:
            update_dynamic_obstacles(physical_model, data)
            print(f"  [{step//fps:02d}s] Pos: [{robot_pos[0]:.2f}, {robot_pos[1]:.2f}] | "
                  f"Goal: {np.linalg.norm(robot_pos - TARGET_POS[:2]):.2f}m")

        mujoco.mj_step(physical_model, data)
        renderer.update_scene(data, camera="top")
        frame = cv2.cvtColor(np.flipud(renderer.render()), cv2.COLOR_RGB2BGR)
        yield frame, step

    dist_to_target = np.linalg.norm(robot_pos - TARGET_POS[:2])
    yield None, {"success": dist_to_target < 0.5, "collision": collision, "steps": step, "time": step/fps}


# ============================================================
# METHOD 2: PULSE (3-Tier)
# ============================================================
def run_pulse_method(physical_model, data, renderer, fps, dt, max_steps):
    agent = LLM_Agent(model='gpt-4o')
    graphormer = DiffGraphormer(in_feats=7, edge_feat_dim=3, hidden_dim=32, num_heads=4, dropout=0.3)
    graphormer.load_state_dict(torch.load(
        os.path.join(base_path, 'model/diffgraphormer_physics.pt'), map_location='cpu'))
    graphormer.eval()

    apex = APEX(graphormer_model=graphormer, physics_simulator="mujoco",
                llm_agent=agent, dt=dt, available_move=available_move)

    walls = get_wall_segments(physical_model)

    robot_body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    robot_dof = physical_model.body_dofadr[robot_body_id]

    collision = False
    collision_threshold = 0.22
    init_frames = int(fps / 2)
    snapshot_t = None
    current_velocity = np.array([0.0, 0.0, 0.0])
    decision_interval = 15
    high_threshold = 0.7
    med_threshold  = 0.3

    stuck_counter = 0
    last_position = START_POS[:2].copy()
    check_interval = 2 * fps

    def fast_risk(snap_t, snap_t_dt, dt_val):
        objs = snap_t["objects"]
        objs_dt = snap_t_dt["objects"]
        robot_pos = None; robot_idx = None
        for idx, obj in enumerate(objs):
            if obj["name"] == "robot":
                robot_pos = np.array(obj["position"][:3])
                robot_pos_dt = np.array(objs_dt[idx]["position"][:3])
                robot_idx = idx
                break
        if robot_pos is None: return 0.0, []

        max_risk = 0.0
        dangerous = []
        for idx, obj in enumerate(objs):
            if idx == robot_idx or obj["name"] == "floor": continue
            obs_pos = np.array(obj["position"][:3])
            dist = np.linalg.norm(robot_pos[:2] - obs_pos[:2])
            if dist > 4.0: continue
            proximity = min(1.0, 1.0 / (dist*dist + 0.1))
            if proximity > max_risk: max_risk = proximity
            if proximity > med_threshold: dangerous.append(obs_pos)

        return max_risk, dangerous

    def tier3_greedy(robot_pos, speed=2.5):
        direction = TARGET_POS[:2] - robot_pos[:2]
        dist = np.linalg.norm(direction)
        if dist < 0.1: return np.array([0.0, 0.0, 0.0])
        goal_dir = direction / dist
        vel = np.array([goal_dir[0]*speed, goal_dir[1]*speed, 0.0])
        # --- wall-safe blend ---
        w_rep = wall_repulsion_vec(robot_pos[:2], walls)
        if np.linalg.norm(w_rep) > 0.1:
            combined = goal_dir + w_rep
            cn = np.linalg.norm(combined)
            if cn > 0.1: combined /= cn
            vel = np.array([combined[0]*speed, combined[1]*speed, 0.0])
        return vel

    def tier2_avoid(robot_pos, dangerous, speed=2.5):
        goal_dir = TARGET_POS[:2] - robot_pos[:2]
        gd = np.linalg.norm(goal_dir)
        if gd > 0.1: goal_dir /= gd

        repulsion = np.array([0.0, 0.0])
        for obs_pos in dangerous:
            diff = robot_pos[:2] - obs_pos[:2]
            d = np.linalg.norm(diff)
            if d > 0.1: repulsion += (diff / d) * (1.0 / (d**2))

        # --- add wall repulsion ---
        w_rep = wall_repulsion_vec(robot_pos[:2], walls)
        repulsion += w_rep

        if np.linalg.norm(repulsion) > 0.1:
            repulsion = repulsion / np.linalg.norm(repulsion)

        combined = 0.5 * goal_dir + 0.5 * repulsion
        n = np.linalg.norm(combined)
        if n > 0.1: combined /= n
        return np.array([combined[0]*speed, combined[1]*speed, 0.0])

    for step in range(max_steps):
        robot_state = get_body_state(physical_model, data, "robot")
        robot_pos = np.array(robot_state["position"][:2])

        if step % fps == 0:
            d = np.linalg.norm(robot_pos - TARGET_POS[:2])
            print(f"  [{step//fps:02d}s] Pos: [{robot_pos[0]:.2f}, {robot_pos[1]:.2f}] | Goal: {d:.2f}m")
            update_dynamic_obstacles(physical_model, data)

        if step % check_interval == 0 and step > 0:
            movement = np.linalg.norm(robot_pos - last_position)
            if movement < 0.3:
                stuck_counter += 1
                if stuck_counter >= 4:
                    print(f"  [EARLY STOP] Robot stuck")
                    break
            else:
                stuck_counter = 0
            last_position = robot_pos.copy()

        if np.linalg.norm(robot_pos - TARGET_POS[:2]) < 0.5:
            print(f"  [SUCCESS] Reached goal at step {step} ({step/fps:.1f}s)!")
            break

        for i in range(1, NUM_TOTAL + 1):
            try:
                cat_state = get_body_state(physical_model, data, f"cat{i}")
                cat_dist = np.linalg.norm(
                    np.array(robot_state["position"][:3]) - np.array(cat_state["position"][:3]))
                if step > init_frames and cat_dist < collision_threshold:
                    if not collision: print(f"  [COLLISION] with cat{i} at step {step}")
                    collision = True
            except:
                pass

        snapshot_t_dt = {"objects": get_all_body_states(physical_model, data)}

        if step > init_frames and step % decision_interval == 0 and snapshot_t is not None:
            risk, dangerous = fast_risk(snapshot_t, snapshot_t_dt, dt)
            robot_pos_3d = np.array(robot_state["position"][:3])

            if risk > high_threshold:
                try:
                    triggered, move, _ = apex.run(snapshot_t, snapshot_t_dt, dt, physical_model, data, step)
                    if triggered and isinstance(move, dict):
                        v = move["velocity"]
                        current_velocity = np.array(v if len(v) >= 3 else v + [0.0])
                    else:
                        current_velocity = tier2_avoid(robot_pos_3d, dangerous)
                except:
                    current_velocity = tier2_avoid(robot_pos_3d, dangerous)
            elif risk > med_threshold:
                current_velocity = tier2_avoid(robot_pos_3d, dangerous)
            else:
                current_velocity = tier3_greedy(robot_pos_3d)

        # --- per-step wall safety clamp ---
        wall_corr = clamp_pos_away_from_walls(robot_pos, walls)
        if np.linalg.norm(wall_corr) > 0.1:
            current_velocity[:2] += wall_corr
            # cap speed
            spd = np.linalg.norm(current_velocity[:2])
            if spd > 3.0:
                current_velocity[:2] = current_velocity[:2] / spd * 3.0

        data.qvel[robot_dof:robot_dof + 3] = current_velocity
        snapshot_t = snapshot_t_dt

        # Keep static obstacles still
        for i in range(NUM_DYNAMIC + 1, NUM_TOTAL + 1):
            try:
                body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
                if body_id != -1:
                    ds = physical_model.body_dofadr[body_id]
                    data.qvel[ds:ds + 3] = [0.0, 0.0, 0.0]
            except:
                pass

        mujoco.mj_step(physical_model, data)
        renderer.update_scene(data, camera="top")
        frame = cv2.cvtColor(np.flipud(renderer.render()), cv2.COLOR_RGB2BGR)
        yield frame, step

    yield None, {"collision": collision, "steps": step, "time": step/fps}


# ============================================================
# METHOD 3: HYBRID (GNN + LLM Bearing)
# ============================================================
# ============================================================
# METHOD 3: HYBRID (GNN + LLM Bearing + TTC Shield)
# ============================================================
def run_hybrid_method(physical_model, data, renderer, fps, dt, max_steps):
    robot_body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    robot_dof = physical_model.body_dofadr[robot_body_id]

    walls = get_wall_segments(physical_model)

    collision = False
    success = False
    collision_threshold = 0.22
    init_frames = int(fps / 2)
    snapshot_t = None
    last_bearing_time = -10.0
    current_bearing = np.array([1.0, 0.0, 0.0]) # Start facing right

    stuck_counter = 0
    last_position = START_POS[:2].copy()
    check_interval = 0.5 * fps # Faster stuck detection (0.5s instead of 2.0s)
    
    # Track obstacle velocities for TTC
    prev_obs_positions = {} # {id: pos}

    def predict_safety_and_score(curr_pos, vel, obstacles_data, dt_val, bearing_vec, static_margin=0.25):
        """
        Returns: (is_safe, score)
        is_safe: boolean (TTC > threshold AND no wall collision)
        score: float (ETA progress - risk penalties)
        """
        # 1. Wall Geometric Filter (Hard Constraint)
        # Relaxed radius 0.14 (was 0.20)
        if check_wall_swept_collision(curr_pos, vel[:2], 0.8, walls, 0.14): 
            # 0.14 radius includes safety margin (0.12 robot + 0.02)
            return False, -float('inf')

        risk_penalty = 0.0

        # Goal Attraction Boost (Drone "Target Lock" when close)
        predicted_pos = curr_pos[:2] + vel[:2] * dt_val
        dist_to_goal = np.linalg.norm(predicted_pos - TARGET_POS[:2])
        if dist_to_goal < 1.5: # Increased range from 1.0
            risk_penalty -= 3.0 / (dist_to_goal + 0.1) # Increased strength from 2.0

        # Soft Wall Risk
        risk_penalty += wall_risk_for_candidate(curr_pos, vel[:2], walls, dt_val=dt_val, margin=0.25)


        # 2. Moving Obstacle TTC Filter (Hard Shield)
        # ONLY for dynamic obstacles — static ones are handled by wall/geometric filters
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
                    radius_sum = 0.12 + 0.12 + 0.10 # Reduced to 0.34
                    
                    # Relative velocity
                    rel_v = rob_vel - obs_v
                    rel_p = curr_pos - obs_pos
                    
                    # Only check if moving towards each other
                    dv = np.dot(rel_v, rel_v)
                    if dv > 1e-4:
                        ttc = calculate_ttc(curr_pos, rob_vel, obs_pos, obs_v, radius_sum)
                        if ttc < min_ttc: min_ttc = ttc
                        
                        if ttc < 0.6: # 0.6s horizon
                            return False, -float('inf')
                        
                        if ttc < 1.2:
                            risk_penalty += 2.0 / (ttc + 0.1)
                else:
                    # STATIC OBSTACLE HANDLING (Geometric + Swept)
                    # We treat them like small walls.
                    # Calculate distance
                    dist_to_static = np.linalg.norm(curr_pos - obs_pos)
                    # Hard veto at margin (0.25 normal, 0.21 creep)
                    if dist_to_static < static_margin:
                        return False, -float('inf')
                        
                    # Swept check for static obstacles to prevent tunneling
                    # Use slightly larger margin for swept check
                    for t_frac in [0.2, 0.5, 0.8, 1.0]:
                        sample_pos = curr_pos + vel[:2] * dt_val * t_frac
                        dist_s = np.linalg.norm(sample_pos - obs_pos)
                        if dist_s < static_margin:
                             return False, -float('inf')
                    
                    if dist_to_static < 0.40:
                        risk_penalty += 0.5 / (dist_to_static + 0.05)  # Soft penalty
                    
            except:
                pass
                
        # 3. ETA Scoring
        # Progress towards goal
        predicted_pos = curr_pos + vel[:2] * dt_val
        dist_now = np.linalg.norm(curr_pos - TARGET_POS[:2])
        dist_future = np.linalg.norm(predicted_pos - TARGET_POS[:2])
        progress = dist_now - dist_future
        
        # Turn penalty (alignment with bearing)
        vel_norm = np.linalg.norm(vel[:2])
        if vel_norm > 1e-3:
            vel_dir = vel[:2] / vel_norm
            alignment = np.dot(vel_dir, bearing_vec[:2])
            turn_cost = (1.0 - alignment) * ETA_Turn_Weight
        else:
            turn_cost = 0.0
            
        # Total Score
        # prioritize progress, penalize risk and turning
        # Small penalty for stopping to encourage movement
        stop_penalty = 0.5 if vel_norm < 1e-3 else 0.0
        score = (progress / dt_val) - (0.5 * risk_penalty) - turn_cost - stop_penalty
        
        return True, score

    def get_bearing(robot_pos, goal_pos, time_now):
        nonlocal last_bearing_time, current_bearing
        # Update bearing less frequently to avoid jitter, but check LOS
        if time_now - last_bearing_time > 1.0:
            vec = goal_pos[:2] - robot_pos[:2]
            n = np.linalg.norm(vec)
            if n > 0.1: direction = vec / n
            else: direction = np.array([1.0, 0.0])
            
            # Simple LOS check: does this bearing point straight into a wall?
            # If so, finding a "corner waypoint" would be better, but for now 
            # we rely on the candidate generator to find side-steps.
            
            current_bearing = np.array([direction[0], direction[1], 0.0])
            last_bearing_time = time_now
        return current_bearing

    def generate_candidates(speed=2.5, current_vel=None):
        """
        Generate diverse candidates: 
        - 16 angles at max speed
        - 8 angles at half speed
        - Stop (zero velocity)
        - If moving, add side-steps perpendicular to motion
        """
        candidates = []
        
        # Max speed
        angles = np.linspace(0, 2 * math.pi, 16, endpoint=False)
        for a in angles:
            candidates.append(np.array([speed * math.cos(a), speed * math.sin(a), 0.0]))
            
        # Half speed (for tight maneuvers)
        for a in angles[::2]: # 8 angles
            hm = 0.5 * speed
            candidates.append(np.array([hm * math.cos(a), hm * math.sin(a), 0.0]))
            
        # Stop
        candidates.append(np.array([0.0, 0.0, 0.0]))
        
        # Side-steps relative to current velocity (if significant)
        if current_vel is not None and np.linalg.norm(current_vel[:2]) > 0.5:
            vx, vy = current_vel[0], current_vel[1]
            # Perpendicular vectors
            p1 = np.array([-vy, vx, 0.0])
            p2 = np.array([vy, -vx, 0.0])
            # Normalize and scale
            p1 = p1 / np.linalg.norm(p1) * speed * 0.8
            p2 = p2 / np.linalg.norm(p2) * speed * 0.8
            candidates.append(p1)
            candidates.append(p2)
            
        return candidates

    current_vel_vector = np.array([0.0, 0.0, 0.0])
    recovery_steps = 0

    for step in range(max_steps):
        robot_state = get_body_state(physical_model, data, "robot")
        robot_pos = np.array(robot_state["position"][:2])

        if recovery_steps > 0:
            recovery_steps -= 1

        if step % fps == 0:
            d = np.linalg.norm(robot_pos - TARGET_POS[:2])
            print(f"  [{step//fps:02d}s] Pos: [{robot_pos[0]:.2f}, {robot_pos[1]:.2f}] | Goal: {d:.2f}m")
            update_dynamic_obstacles(physical_model, data)

        if step % check_interval == 0 and step > 0:
            dist_to_target = np.linalg.norm(robot_pos - TARGET_POS[:2])
            movement = np.linalg.norm(robot_pos - last_position)
            
            # Dynamic Stuck Thresholds (Panic when close)
            # If close to goal, be MORE sensitive to being stuck (require MORE movement)
            stuck_threshold = 0.2
            trigger_count = 3
            if dist_to_target < 1.0:
                stuck_threshold = 0.4 # Require significant movement to not be "stuck"
                trigger_count = 2     # Trigger faster

            if movement < stuck_threshold: 
                stuck_counter += 1
                if stuck_counter >= trigger_count: 
                    print(f"  [STUCK RECOVERY] Engaging random escape maneuver (Close={dist_to_target<1.0})")
                    
                    found_escape = False
                    
                    # DESPERATION MODE: If very close, try lunging result directly at goal
                    if dist_to_target < 0.8:
                         print("    [DESPERATION] Trying direct lunge at goal...")
                         goal_vec = TARGET_POS[:2] - robot_pos[:2]
                         goal_dist = np.linalg.norm(goal_vec)
                         if goal_dist > 0.01:
                             lunge_vel = (goal_vec / goal_dist) * 2.0 # 2.0 m/s lunge
                             # Minimal check: just hard walls
                             if not check_wall_swept_collision(robot_pos, lunge_vel, 0.4, walls, 0.1):
                                 data.qvel[robot_dof:robot_dof + 3] = [lunge_vel[0], lunge_vel[1], 0.0]
                                 current_vel_vector = [lunge_vel[0], lunge_vel[1], 0.0]
                                 recovery_steps = 10 # Short burst
                                 found_escape = True
                                 print(f"    -> LUNGING at goal: {lunge_vel}")

                    if not found_escape:
                        # Force random high-speed move ignoring score (but respecting hard walls)
                        for _ in range(100): # Increased samples from 20 to 100
                            rand_angle = random.uniform(0, 2*math.pi)
                            rand_speed = random.choice([1.0, 2.0, 3.0]) # Variable speeds
                            escape_vel = np.array([rand_speed*math.cos(rand_angle), rand_speed*math.sin(rand_angle), 0.0])
                            
                            # Use smaller radius (0.13) so we can actually move if slightly pinched
                            # And allow start point to be slightly invalid if end point is better (simplified check)
                            if not check_wall_swept_collision(robot_pos, escape_vel[:2], 0.5, walls, 0.13):
                                # Check for collision with obstacles during escape
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
                                    found_escape = True
                                    recovery_steps = 20 # Hold this velocity for 20 steps (0.2s)
                                    print(f"    -> Found escape velocity: {escape_vel[:2]}")
                                    break
                    
                    if not found_escape:
                        # Try CREEP ESCAPE (0.5 m/s, super relaxed)
                        for _ in range(50):
                            rand_angle = random.uniform(0, 2*math.pi)
                            escape_vel = np.array([0.5*math.cos(rand_angle), 0.5*math.sin(rand_angle), 0.0])
                            # Use even smaller radius check for creep (0.12 = robot radius exactly)
                            if not check_wall_swept_collision(robot_pos, escape_vel[:2], 0.5, walls, 0.12):
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
                                    found_escape = True
                                    recovery_steps = 20 
                                    print(f"    -> Found CREEP escape: {escape_vel[:2]}")
                                    break

                    if not found_escape:
                         print("    -> No valid escape found (all colliding)")
                if stuck_counter >= 15: # Give it more time to wiggle out
                    print(f"  [EARLY STOP] Robot stuck")
                    break
            else:
                stuck_counter = 0
            last_position = robot_pos.copy()

        if np.linalg.norm(robot_pos - TARGET_POS[:2]) < 0.5:
            print(f"  [SUCCESS] Reached goal at step {step} ({step/fps:.1f}s)!")
            success = True
            break

        for i in range(1, NUM_TOTAL + 1):
            try:
                cat_state = get_body_state(physical_model, data, f"cat{i}")
                cat_dist = np.linalg.norm(
                    np.array(robot_state["position"][:3]) - np.array(cat_state["position"][:3]))
                if step > init_frames and cat_dist < collision_threshold:
                    if not collision: print(f"  [COLLISION] with cat{i} at step {step}")
                    collision = True
            except:
                pass

        snapshot_t_dt = {"objects": get_all_body_states(physical_model, data)}

        # DECISION LOGIC (Every few frames for smooth control, e.g. 5)
        # Decision interval 5 = 20Hz control
        if step > init_frames and (step % 5 == 0) and snapshot_t is not None and recovery_steps == 0:
            robot_pos_3d = np.array(robot_state['position'])
            bearing = get_bearing(robot_pos_3d, TARGET_POS, step / fps)
            
            # Generate diverse candidates
            candidates = generate_candidates(speed=2.5, current_vel=current_vel_vector)
            
            best_score = -float('inf')
            best_vel = np.array([0.0, 0.0, 0.0])
            safe_found = False
            
            for cand in candidates:
                # dt_val for scoring lookahead = 0.5s
                is_safe, score = predict_safety_and_score(
                    robot_pos_3d[:2], cand, snapshot_t["objects"], 0.5, bearing)
                
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
                         # Penalty for creeping (prefer normal movement if possible)
                         score -= 5.0
                         if score > best_score:
                             best_score = score
                             best_vel = cand
            
            if not safe_found:
                 # Check if staying still is safe
                is_safe_stop, score_stop = predict_safety_and_score(
                    robot_pos_3d[:2], np.array([0.,0.,0.]), snapshot_t["objects"], 0.5, bearing)
                if is_safe_stop:
                    best_vel = np.array([0.0, 0.0, 0.0])
                else:
                    # Emergency: collision inevitable? 
                    # Pick move with max time-to-collision (simplest fallback)
                    # For now just brake
                    best_vel = np.array([0.0, 0.0, 0.0])
            
            # Smooth velocity update (optional LPF)
            # data.qvel[robot_dof:robot_dof + 3] = 0.5*current_vel_vector + 0.5*best_vel
            data.qvel[robot_dof:robot_dof + 3] = best_vel
            current_vel_vector = best_vel

        snapshot_t = snapshot_t_dt

        # Freeze static obstacles (ensure they don't drift)
        for i in range(NUM_DYNAMIC + 1, NUM_TOTAL + 1):
            try:
                body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
                if body_id != -1:
                    ds = physical_model.body_dofadr[body_id]
                    data.qvel[ds:ds + 3] = [0.0, 0.0, 0.0]
            except:
                pass

        mujoco.mj_step(physical_model, data)
        renderer.update_scene(data, camera="top")
        frame = cv2.cvtColor(np.flipud(renderer.render()), cv2.COLOR_RGB2BGR)
        yield frame, step

    yield None, {"success": success, "collision": collision, "steps": step, "time": step/fps}


# ============================================================
# MAIN RUNNER
# ============================================================
def run_pacman_maze_experiment(method='APEX', obstacle_speed=1.0):
    env_path = os.path.join(base_path, "env/pacman_maze_env.xml")
    with open(env_path, 'r') as f:
        xml_string = f.read()

    physical_model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(physical_model)
    renderer = mujoco.Renderer(physical_model, height=480, width=640)

    # Initialize obstacle velocities
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
    output_dir = os.path.abspath(os.path.join(base_path, "../../videos/pacman_maze"))
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.join(output_dir, f"pacman_maze_{method}_{timestamp}.mp4")
    video_writer = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    max_steps = 30 * fps

    print(f"\n{'='*60}")
    print(f"PAC-MAN MAZE EXPERIMENT - {method}")
    print(f"{'='*60}")
    print(f"  Obstacles: {NUM_DYNAMIC} dynamic + {NUM_STATIC} static")
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
        # Use explicit success flag from loop
        is_success = results.get("success", False)
        print(f"  Reached Goal: {is_success}")
        print(f"  Collision: {results['collision']}")
        print(f"  Steps: {results['steps']}")
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
    import argparse
    parser = argparse.ArgumentParser(description="Pac-Man Maze Experiment")
    parser.add_argument("--method", type=str, default="PULSE",
                        choices=["APEX", "PULSE", "HYBRID"])
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    run_pacman_maze_experiment(method=args.method, obstacle_speed=args.speed)
