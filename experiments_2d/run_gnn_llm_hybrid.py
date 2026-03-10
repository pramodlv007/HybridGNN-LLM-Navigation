"""
GNN Path + Mixed Dynamic Obstacles Experiment (Standard APEX)
=============================================================
- 20 total obstacles: 10 dynamic (red, moving randomly), 10 static (blue, fixed after placement)
- ALL obstacle positions randomized at startup for different layouts each run
- Uses STANDARD APEX framework (DiffGraphormer danger detection + LLM decision)
  Priority 1: Avoid ALL contact with obstacles
  Priority 2: Navigate toward goal
- Video saved to videos/ folder
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

# Adjust sys.path to include the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
# Import local APEX_Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# from experiments.cat_expt.utils.cat_game_agent import LLM_Agent
# from experiments.cat_expt.utils.APEX import APEX

from experiments.cat_expt.model.local_risk_gnn import LocalRiskGNN
from experiments.cat_expt.utils.mujoco_simulator import get_body_state, get_all_body_states
from hybrid_nav import HybridController, GNNSafetyChecker, StrategicNavigator

# Constants
START_POS = np.array([-3.6, -3.6, 0.12])
TARGET_POS = np.array([3.6, 3.6, 0.12])
NUM_DYNAMIC = 10   # cat1 to cat10 - moving
NUM_STATIC = 15    # cat11 to cat25 - fixed after placement
NUM_TOTAL = NUM_DYNAMIC + NUM_STATIC

# Initialization for Stuck Recovery
stuck_mode = False
last_stuck_pos = START_POS[:2]

# Paths
base_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(base_path, "../"))

with open(os.path.join(root_path, "env/available_move.json"), 'r') as f:
    available_move = json.load(f)


def get_random_velocity(speed=1.0):
    """Generate random velocity in a random direction."""
    angle = random.uniform(0, 2 * np.pi)
    vx = speed * np.cos(angle)
    vy = speed * np.sin(angle)
    return vx, vy


def randomize_all_obstacles(model, data):
    """Randomize positions of all 20 obstacles, keeping clearance from start/goal."""
    np.random.seed(int(time.time()))
    random.seed(int(time.time()))

    placed_positions = []

    for i in range(1, NUM_TOTAL + 1):
        cat_name = f"cat{i}"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cat_name)
        if body_id == -1:
            print(f"Warning: {cat_name} not found in model")
            continue

        dof_start = model.body_dofadr[body_id]

        # Try to find a valid position
        for _ in range(200):
            rx = np.random.uniform(-3.3, 3.3)
            ry = np.random.uniform(-3.3, 3.3)
            pos = np.array([rx, ry])

            # Clearance from start and goal
            dist_start = np.linalg.norm(pos - START_POS[:2])
            dist_goal = np.linalg.norm(pos - TARGET_POS[:2])

            # Clearance from other placed obstacles
            too_close = False
            for placed in placed_positions:
                if np.linalg.norm(pos - placed) < 0.5:
                    too_close = True
                    break

            if dist_start > 1.5 and dist_goal > 1.5 and not too_close:
                z = 0.1 if i <= NUM_DYNAMIC else 0.08
                data.qpos[dof_start:dof_start + 3] = [rx, ry, z]
                placed_positions.append(pos)
                break

    print(f"Randomized {len(placed_positions)} obstacle positions")
    return placed_positions


def update_dynamic_obstacles(model, data, speed=1.0):
    """Update velocities for dynamic obstacles (cat1-cat10) with random movement."""
    for i in range(1, NUM_DYNAMIC + 1):
        cat_name = f"cat{i}"
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cat_name)
            if body_id == -1:
                continue

            dof_start = model.body_dofadr[body_id]
            pos = data.xpos[body_id]

            current_vx = data.qvel[dof_start]
            current_vy = data.qvel[dof_start + 1]

            # Bounce off walls (arena is -4 to 4)
            if pos[0] < -3.5:
                current_vx = abs(current_vx) if current_vx < 0 else current_vx
            elif pos[0] > 3.5:
                current_vx = -abs(current_vx) if current_vx > 0 else current_vx

            if pos[1] < -3.5:
                current_vy = abs(current_vy) if current_vy < 0 else current_vy
            elif pos[1] > 3.5:
                current_vy = -abs(current_vy) if current_vy > 0 else current_vy

            # 30% chance to change direction randomly
            if random.random() < 0.3:
                vx, vy = get_random_velocity(speed)
                data.qvel[dof_start] = vx
                data.qvel[dof_start + 1] = vy
            else:
                data.qvel[dof_start] = current_vx
                data.qvel[dof_start + 1] = current_vy

        except Exception:
            pass




def run_gnn_path_mixed_exp(method='APEX', model_name='gpt-4o-mini', obstacle_speed=1.0):
    """Run standard APEX experiment with mixed dynamic+static obstacles."""

    # Load mixed    # Load MuJoCo model
    # Use the new 25-obstacle environment
    env_path = os.path.join(root_path, "env/mixed_random_env_25.xml") # Updated to 25 obstacles
    with open(env_path, 'r') as f:
        xml_string = f.read()

    physical_model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(physical_model)
    renderer = mujoco.Renderer(physical_model, height=480, width=640)

    # Get robot DOF address (obstacles have freejoints before robot)
    robot_body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    robot_dof = physical_model.body_dofadr[robot_body_id]
    print(f"Robot body_id={robot_body_id}, dof_start={robot_dof}")

    # Randomize all obstacle positions
    randomize_all_obstacles(physical_model, data)

    # Run one mj_step to apply positions
    mujoco.mj_step(physical_model, data)

    # Initialize agent
    # agent = LLM_Agent(model=model_name)
    collision = False
    collision_threshold = 0.2

    # Action management (standard APEX pattern)
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
                print(f"Skipping invalid action: {current_action}")
                return
            frames_left = int(current_action["duration"] * fps)
            vel = current_action["velocity"]
            data.qvel[robot_dof:robot_dof + 3] = vel
            print(f"Executing action: velocity={vel}, duration={current_action['duration']}s")

    # Video settings
    fps = 100
    width, height = 640, 480
    output_dir = os.path.abspath(os.path.join(root_path, "../../videos"))
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.join(output_dir, f"gnn_path_mixed_{method}_{timestamp}.mp4")
    video_writer = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Simulation variables
    dt = 1.0 / fps
    init_frames = int(fps / 2)
    
    # Initialize Hybrid Navigator (GNN Safety + LLM Strategy)
    safety_model_path = os.path.join(root_path, 'model/diffgraphormer_physics.pt')
    
    # 1. Safety Checker (GNN)
    safety_checker = GNNSafetyChecker(
        model_path=safety_model_path,
        use_local_gnn=False, # Use heuristic/Graphormer logic
        device="cpu"
    )
    
    # 2. Key Strategist (LLM)
    strategist = StrategicNavigator()
    
    # 3. Hybrid Controller
    # navigator handles danger logic via HybridController.get_velocity()
    navigator = HybridController(safety_checker, strategist)

    # Initialize random velocities for dynamic obstacles (cat1-cat10)
    for i in range(1, NUM_DYNAMIC + 1):
        cat_name = f"cat{i}"
        try:
            body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, cat_name)
            if body_id != -1:
                dof_start = physical_model.body_dofadr[body_id]
                vx, vy = get_random_velocity(obstacle_speed)
                data.qvel[dof_start] = vx
                data.qvel[dof_start + 1] = vy
        except Exception:
            pass

    # Ensure static obstacles have zero velocity
    for i in range(NUM_DYNAMIC + 1, NUM_TOTAL + 1):
        cat_name = f"cat{i}"
        try:
            body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, cat_name)
            if body_id != -1:
                dof_start = physical_model.body_dofadr[body_id]
                data.qvel[dof_start] = 0.0
                data.qvel[dof_start + 1] = 0.0
                data.qvel[dof_start + 2] = 0.0
        except Exception:
            pass

    snapshot_t, snapshot_t_dt = None, None
    max_steps = 30 * fps  # 30 seconds max (for reasonable testing time)
    
    # Decision throttling variables
    last_decision_step = 0
    decision_interval = 10  # Make decision every 10 frames (~0.10s) - more responsive
    
    # Early stopping: detect if robot is stuck
    stuck_counter = 0
    last_position = START_POS[:2].copy()
    position_check_interval = 2 * fps  # Check every 2 seconds

    print(f"=" * 60)
    print(f"GNN + Standard APEX Mixed Obstacles Experiment (ORIGINAL)")
    print(f"=" * 60)
    print(f"Method: {method} (Standard APEX - Safety First)")
    print(f"Dynamic obstacles (moving): {NUM_DYNAMIC}")
    print(f"Static obstacles (fixed):   {NUM_STATIC}")
    print(f"Robot Start: {START_POS}")
    print(f"Goal Target: {TARGET_POS}")
    print(f"Obstacle Speed: {obstacle_speed}")
    print(f"Max Duration: {max_steps/fps:.0f}s")
    print(f"=" * 60)

    for step in range(max_steps):
        # Current State
        robot_state = get_body_state(physical_model, data, "robot")
        robot_pos = np.array(robot_state["position"][:2])

        # Log progress every 1 second
        if step % fps == 0:
            dist = np.linalg.norm(robot_pos - TARGET_POS[:2])
            print(f"[{step//fps:02d}s] Step {step}/{max_steps} | Robot: [{robot_pos[0]:.2f}, {robot_pos[1]:.2f}] | Dist: {dist:.2f}m")
            # Update dynamic obstacle velocities every second
            update_dynamic_obstacles(physical_model, data, speed=obstacle_speed)
        
        # Check if robot is stuck (hasn't moved much)
        if step % position_check_interval == 0 and step > 0:
            movement = np.linalg.norm(robot_pos - last_position)
            if movement < 0.3:  # Moved less than 0.3m in 2 seconds
                stuck_counter += 1
                print(f"[WARNING] Robot may be stuck (moved {movement:.3f}m in {position_check_interval/fps:.1f}s)")
                if stuck_counter >= 3:  # Stuck for 6 seconds
                    print(f"[EARLY STOP] Robot stuck for {stuck_counter * position_check_interval/fps:.0f}s - terminating")
                    break
            else:
                stuck_counter = 0
            last_position = robot_pos.copy()

        # Goal Check
        dist_to_target = np.linalg.norm(robot_pos - TARGET_POS[:2])
        if dist_to_target < 0.5:
            print(f"[SUCCESS] Reached target at step {step} ({step/fps:.1f}s)!")
            break

        # Collision Check against all obstacles
        for i in range(1, NUM_TOTAL + 1):
            try:
                cat_state = get_body_state(physical_model, data, f"cat{i}")
                cat_distance = np.linalg.norm(
                    np.array(robot_state["position"][:3]) - np.array(cat_state["position"][:3])
                )
                if step > init_frames and cat_distance < collision_threshold:
                    if not collision:
                        print(f"[WARNING] Collision detected with cat{i} at step {step}!")
                    collision = True
            except Exception:
                pass

        snapshot_t_dt = {"objects": get_all_body_states(physical_model, data)}

        # HYBRID NAVIGATION LOGIC
        if snapshot_t is not None:
             # navigator.get_velocity(robot_state, obstacles, goal_pos, time, dt, fps)
             velocity = navigator.get_velocity(
                 robot_state, 
                 snapshot_t["objects"], 
                 TARGET_POS, 
                 step/fps, 
                 dt, 
                 fps
             )
             data.qvel[robot_dof:robot_dof + 3] = velocity

        
        snapshot_t = snapshot_t_dt


        # Keep static obstacles frozen
        for i in range(NUM_DYNAMIC + 1, NUM_TOTAL + 1):
            try:
                body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
                if body_id != -1:
                    dof_start = physical_model.body_dofadr[body_id]
                    data.qvel[dof_start] = 0.0
                    data.qvel[dof_start + 1] = 0.0
            except Exception:
                pass

        mujoco.mj_step(physical_model, data)

        # Render
        renderer.update_scene(data, camera="top")
        frame = cv2.cvtColor(np.flipud(renderer.render()), cv2.COLOR_RGB2BGR)
        video_writer.write(cv2.resize(frame, (width, height)))

    video_writer.release()
    print(f"\n{'=' * 60}")
    print(f"Finished! Video saved: {file_name}")
    print(f"Collision occurred: {collision}")
    print(f"{'=' * 60}")
    return collision


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("WARNING: OPENAI_API_KEY not found in environment variables.")

    run_gnn_path_mixed_exp()
