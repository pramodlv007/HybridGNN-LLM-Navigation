import copy
import random
import time
import mujoco
import numpy as np


def randomize_obstacles(model, num_obstacles=None, bounds=(-4.0, 4.0), 
                        start_pos=None, goal_pos=None, min_clearance=1.5,
                        obstacle_prefix="cat"):
    """
    Randomize obstacle positions in the MuJoCo model.
    
    This function can be called before creating MjData to generate random 
    obstacle layouts for any experiment.
    
    Args:
        model: MuJoCo model to modify
        num_obstacles: Number of obstacles to randomize (auto-detected if None)
        bounds: Tuple of (min, max) for x and y coordinates
        start_pos: Robot start position to avoid (default: None)
        goal_pos: Goal position to avoid (default: None)  
        min_clearance: Minimum distance from start/goal positions
        obstacle_prefix: Prefix for obstacle body names (default: "cat")
    
    Returns:
        int: Number of obstacles randomized
    """
    # Seed with current time for different layout each run
    random.seed(time.time())
    
    # Auto-detect number of obstacles if not specified
    if num_obstacles is None:
        num_obstacles = 0
        for i in range(1, 100):  # Check up to 100 obstacles
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{obstacle_prefix}{i}")
            if body_id == -1:
                break
            num_obstacles = i
    
    if num_obstacles == 0:
        return 0
    
    print(f"Randomizing {num_obstacles} obstacles...")
    
    for i in range(1, num_obstacles + 1):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{obstacle_prefix}{i}")
        if body_id == -1:
            continue
            
        # Try to find valid random position
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.uniform(bounds[0], bounds[1])
            y = random.uniform(bounds[0], bounds[1])
            
            # Check clearance from start position
            if start_pos is not None:
                dist_start = np.linalg.norm(np.array([x, y]) - np.array(start_pos[:2]))
                if dist_start < min_clearance:
                    continue
            
            # Check clearance from goal position
            if goal_pos is not None:
                dist_goal = np.linalg.norm(np.array([x, y]) - np.array(goal_pos[:2]))
                if dist_goal < min_clearance:
                    continue
            
            # Valid position found
            model.body_pos[body_id][0] = x
            model.body_pos[body_id][1] = y
            break
    
    return num_obstacles


def get_body_state(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    pos = data.xpos[body_id].copy()
    vel = data.cvel[body_id, :6].copy()
    return {
        "name": body_name,
        "position": pos.tolist(),
        "velocity": vel.tolist()
    }


def get_all_body_states(model, data, filter_out=['world']):
    states = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name not in filter_out:
            pos = data.xpos[i].copy()
            vel = data.cvel[i, :6].copy()
            states.append({
                "name": name,
                "position": pos.tolist(),
                "velocity": vel.tolist()
            })
    return states


class simulator:
    def __init__(self, method):
        self.method = method

    def mujoco_sim(self, model, env_data, available_moves):
        sim_results = {}
        max_duration = 1.0
        timestep = model.opt.timestep
        max_steps = int(max_duration / timestep)

        for action_name, action_desc in available_moves.items():
            sim_data = copy.deepcopy(env_data)

            robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
            dof_start = model.body_dofadr[robot_body_id]

            vel = [0.0, 0.0, 0.0]  # Can be replaced with the real vel of the robot
            if action_name == "move_left":
                vel[0] = -3.0
            elif action_name == "move_right":
                vel[0] = 3.0
            elif action_name == "move_up":
                vel[1] = -3.0
            elif action_name == "move_down":
                vel[1] = 3.0
            elif action_name == "jump":
                vel[2] = 3.0

            for i in range(3):
                sim_data.qvel[dof_start + i] = vel[i]

            safe_steps = 0
            last_pos = np.array(sim_data.xpos[robot_body_id])

            for step in range(max_steps):
                mujoco.mj_step(model, sim_data)
                current_pos = np.array(sim_data.xpos[robot_body_id])
                movement = np.linalg.norm(current_pos[:3] - last_pos[:3])
                if movement < 1e-3 and action_name not in ['jump', 'stay']:
                    break

                last_pos = current_pos
                safe_steps += 1

            safe_duration = safe_steps * timestep
            # stay away from wall
            if safe_duration < 1.0 and action_name not in ['jump', 'stay']:
                safe_duration -= 0.1
                # if save_duration < 0, then the action will be evaluated into 'invalid' in the summary
            else:
                safe_duration = 1.0

            sim_results[action_name] = {
                "final_pos": get_all_body_states(model, sim_data),
                "description": {
                    "velocity": vel,
                    "duration": safe_duration,
                    "description": f"{action_name} with velocity={vel} for {safe_duration:.2f}s"
                }
            }

        return sim_results

    def sim(self, model, env_data, action):
        if self.method == 'mujoco':
            return self.mujoco_sim(model, env_data, action)
        return None
