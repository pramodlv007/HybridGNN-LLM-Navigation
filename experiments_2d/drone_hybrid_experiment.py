"""
Drone Hybrid Experiment (Explicit LLM + HUD + Logging)
======================================================
The LLM selects an explicit action (e.g., turn_left_30, move_forward) from
drone_available_move.json. The simulation executes this for the specified duration.
A HUD overlay shows the current action, and an action_log.txt is saved alongside
the video. The 16-feature GNN acts as a proactive safety shield.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.cat_game_agent import LLM_Agent
from model.local_risk_gnn_16feat import LocalRiskGNN16Feat
from utils.mujoco_simulator import get_body_state, get_all_body_states
from medium_nfz_experiment import (
    START_POS, TARGET_POS, NUM_DYNAMIC, NUM_STATIC, NUM_TOTAL, NFZ_ZONES,
    is_inside_any_nfz, min_nfz_distance, combined_nfz_repulsion,
    would_enter_nfz_swept, randomize_all_obstacles, update_dynamic_obstacles,
    get_random_velocity
)

base_path = os.path.dirname(os.path.abspath(__file__))

def extract_gnn_features_16(snapshot_t, snapshot_t_dt, dt_val):
    objs = snapshot_t["objects"]
    objs_dt = snapshot_t_dt["objects"]

    valid_objs = []
    valid_objs_dt = []
    robot_idx = -1

    for i, obj in enumerate(objs):
        if obj["name"] in ("floor", "wall"): continue
        valid_objs.append(obj)
        valid_objs_dt.append(objs_dt[i])
        if obj["name"] == "robot": robot_idx = len(valid_objs) - 1

    num_nodes = len(valid_objs)
    x_t = torch.zeros(num_nodes, 16)
    x_t_dt = torch.zeros(num_nodes, 16)

    for i in range(num_nodes):
        obj = valid_objs[i]
        obj_dt = valid_objs_dt[i]

        is_master = 1.0 if i == robot_idx else 0.0
        pos = np.array(obj["position"][:3])
        pos_dt = np.array(obj_dt["position"][:3])
        vel = (pos_dt - pos) / max(dt_val, 1e-6)
        speed = float(np.linalg.norm(vel))
        vel_dir = vel / speed if speed > 1e-3 else np.zeros(3)

        dist_to_goal = float(np.linalg.norm(pos[:2] - TARGET_POS[:2]))
        bearing = (TARGET_POS[:3] - pos) / max(dist_to_goal, 0.01)
        nfz_d = float(min_nfz_distance(pos[0], pos[1]))
        is_dynamic = 1.0 if i < (1 + NUM_DYNAMIC) and i != robot_idx else 0.0

        x_t[i, 0] = is_master
        x_t[i, 1:4] = torch.tensor(pos, dtype=torch.float32)
        x_t[i, 4:7] = torch.tensor(vel_dir, dtype=torch.float32)
        x_t[i, 7] = speed
        x_t[i, 8] = dist_to_goal
        x_t[i, 9:12] = torch.tensor(bearing[:3], dtype=torch.float32)
        x_t[i, 12] = nfz_d
        x_t[i, 13] = is_dynamic

        x_t_dt[i, 0] = is_master
        x_t_dt[i, 1:4] = torch.tensor(pos_dt, dtype=torch.float32)

    if robot_idx != -1 and num_nodes > 1:
        src = [robot_idx] * (num_nodes - 1)
        dst = [j for j in range(num_nodes) if j != robot_idx]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return x_t, x_t_dt, edge_index, robot_idx

def add_hud(frame, step, fps, current_action, max_risk, nfz_d, dist):
    """Draw a dark banner HUD at the top of the video."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
    
    act_str = current_action.get('description', 'Hovering / Waiting') if current_action else 'Starting'
    
    cv2.putText(frame, f"Action: {act_str}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    stats = f"Time: {step/fps:.1f}s | Dist: {dist:.1f}m | Risk: {max_risk:.2f} | NFZ: {nfz_d:.1f}m"
    cv2.putText(frame, stats, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def run_drone_hybrid(physical_model, data, renderer, fps, dt, max_steps, log_file):
    robot_body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    robot_dof = physical_model.body_dofadr[robot_body_id]

    gnn_model = LocalRiskGNN16Feat(in_feats=16, hidden_dim=64)
    gnn_model.eval()

    agent = LLM_Agent(model='gpt-4o')
    with open(os.path.join(base_path, "env/drone_available_move.json"), 'r') as f:
        available_move = json.load(f)

    collision = False
    nfz_violated = False
    success = False
    init_frames = int(fps / 2)

    snapshot_t = None
    drone_yaw = math.atan2(TARGET_POS[1] - START_POS[1], TARGET_POS[0] - START_POS[0])
    
    current_action = None
    frames_left = 0
    max_risk = 0.0

    for step in range(max_steps):
        robot_state = get_body_state(physical_model, data, "robot")
        robot_pos = np.array(robot_state["position"][:3])

        nfz_d = min_nfz_distance(robot_pos[0], robot_pos[1])
        if is_inside_any_nfz(robot_pos[0], robot_pos[1]):
            nfz_violated = True

        dist_to_target = np.linalg.norm(robot_pos[:2] - TARGET_POS[:2])
        if dist_to_target < 0.5:
            success = True
            break

        # Check collisions
        for i in range(1, NUM_TOTAL + 1):
            try:
                cat_state = get_body_state(physical_model, data, f"cat{i}")
                cat_dist = np.linalg.norm(robot_pos - np.array(cat_state["position"][:3]))
                if step > init_frames and cat_dist < 0.2:
                    collision = True
            except: pass

        snapshot_t_dt = {"objects": get_all_body_states(physical_model, data)}

        # GNN Background Safety Check
        if snapshot_t is not None:
            x_t, x_t_dt, edge_index, robot_idx = extract_gnn_features_16(snapshot_t, snapshot_t_dt, dt)
            with torch.no_grad():
                risk_logits = gnn_model(x_t, x_t_dt, edge_index, dt)
                risk_probs = torch.sigmoid(risk_logits)
            
            if robot_idx != -1 and len(risk_probs) > 1:
                max_risk = risk_probs[1:].max().item()

            # Safety Override
            vel_now = data.qvel[robot_dof:robot_dof + 3].copy()
            should_override = False
            
            if would_enter_nfz_swept(robot_pos, vel_now[:2], dt_check=0.4):
                should_override = True
            elif max_risk > 0.80 and np.linalg.norm(vel_now[:2]) > 0.1:
                should_override = True

            if should_override:
                frames_left = 0 # Force LLM to pick a new move
                rep = combined_nfz_repulsion(robot_pos)
                if np.linalg.norm(rep) > 0.0:
                    data.qvel[robot_dof:robot_dof + 3] = np.array([rep[0]*1.5, rep[1]*1.5, 0])
                else:
                    data.qvel[robot_dof:robot_dof + 3] = np.array([-vel_now[0]*0.5, -vel_now[1]*0.5, 0])

        # LLM Decision Logic
        if frames_left <= 0 and step > init_frames:
            warn = f"CRITICAL DANGER! Dist: {nfz_d:.2f}m" if nfz_d < 1.0 or max_risk > 0.8 else "Clear"
            
            state_desc = (
                f"Time: {step/fps:.1f}s\n"
                f"Drone pos: {robot_pos[:3]}\n"
                f"Drone Goal: {TARGET_POS[:3]}\n"
                f"Yaw: {math.degrees(drone_yaw):.1f} deg\n"
                f"Obstacle Risk GNN: {max_risk:.2f}/1.00\n"
                f"Status: {warn}\n"
                f"Choose EXACTLY ONE dictionary object from 'available_move'."
            )

            # Block simulation to wait for API
            move, _ = agent.decide_move(state_desc, available_move)
            
            if isinstance(move, dict):
                current_action = move
                desc = current_action.get('description', 'Unknown')
                act_type = current_action.get('action_type', 'hover')
                
                # Write to log
                log_file.write(f"[{step/fps:05.2f}s] ACTION SELECTED: {desc} (Type: {act_type})\n")
                log_file.flush()
                print(f"  [{step/fps:.1f}s] LLM: {desc}")
                
                dur = float(current_action.get("duration", 0.5))
                frames_left = max(int(dur * fps), 1)

                if act_type == "rotate":
                    drone_yaw += math.radians(current_action.get("yaw_change", 0.0))
            else:
                print(f"  [{step/fps:.1f}s] LLM API Quota Exceeded/Error: {move}")
                print(f"  [{step/fps:.1f}s] Using MOCK LLM Actions for demonstration.")
                # Mock a random explicit drone action from the defined JSON list
                mock_actions = [
                    {"move": "move_forward", "description": "Move Forward 2m", "action_type": "translate", "velocity": [2.0, 0.0, 0.0], "duration": 1.0},
                    {"move": "turn_left_30", "description": "Turn Left 30 deg", "action_type": "rotate", "yaw_change": 30.0, "duration": 0.5},
                    {"move": "turn_right_30", "description": "Turn Right 30 deg", "action_type": "rotate", "yaw_change": -30.0, "duration": 0.5},
                    {"move": "swing_left_30", "description": "Swing Left", "action_type": "translate_local", "velocity": [0.0, 1.5, 0.0], "duration": 1.0},
                    {"move": "ascend", "description": "Ascend 1m", "action_type": "translate_z", "velocity": [0.0, 0.0, 1.0], "duration": 1.0}
                ]
                
                # Biased towards moving forward and turning towards goal
                goal_angle = math.degrees(math.atan2(TARGET_POS[1] - robot_pos[1], TARGET_POS[0] - robot_pos[0]))
                curr_angle = math.degrees(drone_yaw)
                diff = (goal_angle - curr_angle + 180) % 360 - 180
                
                if abs(diff) > 20:
                    current_action = mock_actions[1] if diff > 0 else mock_actions[2]
                else:
                    current_action = random.choices(mock_actions, weights=[0.6, 0.1, 0.1, 0.1, 0.1])[0]

                desc = current_action.get('description', 'Mock Hover')
                act_type = current_action.get('action_type', 'hover')
                
                log_file.write(f"[{step/fps:05.2f}s] [MOCK] ACTION SELECTED: {desc} (Type: {act_type})\n")
                log_file.flush()
                
                dur = float(current_action.get("duration", 0.5))
                frames_left = max(int(dur * fps), 1)

                if act_type == "rotate":
                    drone_yaw += math.radians(current_action.get("yaw_change", 0.0))
        # Apply Drone Physics from Current Action
        if current_action and frames_left > 0:
            frames_left -= 1
            act_type = current_action.get("action_type", "hover")
            vel_mag = current_action.get("velocity", [0.0, 0.0, 0.0])
            
            vx, vy, vz = 0.0, 0.0, 0.0
            
            if act_type == "translate":
                speed = np.linalg.norm(vel_mag)
                vx = speed * math.cos(drone_yaw)
                vy = speed * math.sin(drone_yaw)
            elif act_type == "translate_local":
                # swinging laterally uses y velocity from json
                speed = vel_mag[1] 
                vx = speed * math.cos(drone_yaw + math.pi/2)
                vy = speed * math.sin(drone_yaw + math.pi/2)
            elif act_type == "translate_z":
                vz = vel_mag[2]
                
            data.qvel[robot_dof:robot_dof + 3] = [vx, vy, vz]
        else:
            # Gravity sink guard
            if robot_pos[2] > 0.12: data.qvel[robot_dof:robot_dof + 3] = [0, 0, -0.5]


        snapshot_t = snapshot_t_dt

        # Static obstacles stay still
        for i in range(NUM_DYNAMIC + 1, NUM_TOTAL + 1):
            try:
                body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
                if body_id != -1:
                    dof_start = physical_model.body_dofadr[body_id]
                    data.qvel[dof_start:dof_start + 3] = [0.0, 0.0, 0.0]
            except: pass

        if step % fps == 0:
            update_dynamic_obstacles(physical_model, data)

        mujoco.mj_step(physical_model, data)
        renderer.update_scene(data, camera="top")
        frame = cv2.cvtColor(np.flipud(renderer.render()), cv2.COLOR_RGB2BGR)
        
        # Add HUD overlay
        frame = add_hud(frame, step, fps, current_action, max_risk, nfz_d, dist_to_target)
        
        yield frame, step

    yield None, {"success": success, "collision": collision, "nfz_violated": nfz_violated,
                 "steps": step, "time": step / fps}


def run_drone_hybrid_experiment(obstacle_speed=1.0):
    env_path = os.path.join(base_path, "env/drone_nfz_env.xml")

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
        except: pass
    for i in range(NUM_DYNAMIC + 1, NUM_TOTAL + 1):
        try:
            body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
            if body_id != -1:
                dof_start = physical_model.body_dofadr[body_id]
                data.qvel[dof_start:dof_start + 3] = [0.0, 0.0, 0.0]
        except: pass

    fps = 100
    dt = 1.0 / fps
    width, height = 640, 480
    max_steps = 30 * fps

    # Setup out dirs
    output_dir = os.path.abspath(os.path.join(base_path, "../../videos/LLM_Actions"))
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.join(output_dir, f"drone_hybrid_{timestamp}.mp4")
    log_name = os.path.join(output_dir, f"drone_hybrid_{timestamp}_action_log.txt")
    
    video_writer = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    log_file = open(log_name, "w")

    print(f"\n{'=' * 60}")
    print(f"DRONE HYBRID SIMULATION (EXPLICIT ACTIONS + HUD)")
    print(f"{'=' * 60}")

    generator = run_drone_hybrid(physical_model, data, renderer, fps, dt, max_steps, log_file)

    results = None
    for output in generator:
        frame, info = output
        if frame is None:
            results = info
        else:
            video_writer.write(frame)

    video_writer.release()
    log_file.close()

    if results:
        print(f"\nRESULTS: Goal={results.get('success')}, Col={results['collision']}, NFZ={results['nfz_violated']}")
        print(f"Video saved to: {file_name}")
        print(f"Log saved to: {log_name}\n")


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("WARNING: OPENAI_API_KEY not found.")
    run_drone_hybrid_experiment(obstacle_speed=1.0)
