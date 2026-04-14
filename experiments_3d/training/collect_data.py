"""
Physics-Informed GNN Data Collection
======================================
Runs random trajectories in the PyBullet 3D environment and collects
collision data for training the LocalRiskGNN.

For each timestep, records:
  - Graph features: robot + obstacle positions/velocities
  - Collision label: did robot collide within next K steps?
  - Physics features: TTC values and closing speeds (for physics-informed loss)

Usage:
  python training/collect_data.py --episodes 200 --output training/collision_data.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
import random
import math

# Add parent to path so we can import envs/controllers
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from envs.drone_nav_3d import DroneNav3DEnv


def compute_ttc_value(pos1, vel1, pos2, vel2, radius=0.3):
    """Compute raw TTC between two objects. Returns seconds (inf if no collision)."""
    rel_pos = np.array(pos2) - np.array(pos1)
    rel_vel = np.array(vel2) - np.array(vel1)
    dist = np.linalg.norm(rel_pos)

    if dist < radius:
        return 0.0  # Already colliding

    if dist < 1e-6:
        return 0.0

    closing_speed = -np.dot(rel_pos, rel_vel) / dist
    if closing_speed <= 0:
        return float('inf')  # Diverging

    ttc = (dist - radius) / closing_speed
    return max(ttc, 0.0)


def compute_closing_speed(pos1, vel1, pos2, vel2):
    """Compute closing speed between two objects. Positive = approaching."""
    rel_pos = np.array(pos2) - np.array(pos1)
    rel_vel = np.array(vel2) - np.array(vel1)
    dist = np.linalg.norm(rel_pos)
    if dist < 1e-6:
        return 0.0
    return -np.dot(rel_pos, rel_vel) / dist


def build_graph_features(obs):
    """
    Build GNN graph from observation dict.

    Returns:
        x_t: [N, 7] current features [is_master, pos_x, pos_y, pos_z, vel_dir_x, vel_dir_y, vel_dir_z]
        edge_index: [2, E] edges from robot to each obstacle
        raw_data: dict with positions/velocities for physics loss
    """
    robot_pos = obs["pos"]
    robot_vel = obs["vel"]
    obstacles = obs["obstacles"]

    N = 1 + len(obstacles)  # robot + obstacles

    # Normalize velocities to direction vectors
    robot_speed = np.linalg.norm(robot_vel)
    robot_vel_dir = robot_vel / (robot_speed + 1e-6)

    features = []

    # Robot node (index 0)
    features.append([1.0,  # is_master
                     robot_pos[0], robot_pos[1], robot_pos[2],
                     robot_vel_dir[0], robot_vel_dir[1], robot_vel_dir[2]])

    # Obstacle nodes
    for obs_data in obstacles:
        obs_pos = obs_data["pos"]
        obs_vel = obs_data["vel"]
        obs_speed = np.linalg.norm(obs_vel)
        obs_vel_dir = obs_vel / (obs_speed + 1e-6)

        features.append([0.0,  # not master
                         obs_pos[0], obs_pos[1], obs_pos[2],
                         obs_vel_dir[0], obs_vel_dir[1], obs_vel_dir[2]])

    x_t = torch.tensor(features, dtype=torch.float32)

    # Star topology: robot (0) -> each obstacle (1..N-1)
    src = [0] * len(obstacles)
    tgt = list(range(1, N))
    edge_index = torch.tensor([src, tgt], dtype=torch.long)

    # Raw data for physics loss computation
    raw_data = {
        "robot_pos": robot_pos.copy(),
        "robot_vel": robot_vel.copy(),
        "obstacle_positions": [o["pos"].copy() for o in obstacles],
        "obstacle_velocities": [o["vel"].copy() for o in obstacles],
        "obstacle_radii": [o["radius"] for o in obstacles],
    }

    return x_t, edge_index, raw_data


def collect_episode(env, max_steps=300, collision_horizon=40):
    """
    Run one episode with random actions and collect training samples.

    Args:
        env: DroneNav3DEnv instance
        max_steps: max steps per episode
        collision_horizon: how many future steps to check for collision label

    Returns:
        list of sample dicts
    """
    obs = env.reset()
    trajectory = []  # Store all observations
    actions = []
    done_steps = []

    # Run episode
    done = False
    step = 0
    episode_done_info = {}

    while not done and step < max_steps:
        # Random action with occasional goal-bias for variety
        if random.random() < 0.3:
            # Goal-biased
            goal_dir = obs["goal"][:2] - obs["pos"][:2]
            norm = np.linalg.norm(goal_dir)
            if norm > 0.1:
                goal_dir = goal_dir / norm
            accel = np.array([goal_dir[0] * 3.0, goal_dir[1] * 3.0, 0.0])
        else:
            # Random
            angle = random.uniform(0, 2 * math.pi)
            magnitude = random.uniform(0.5, 4.0)
            accel = np.array([magnitude * math.cos(angle),
                              magnitude * math.sin(angle), 0.0])

        trajectory.append(obs)
        actions.append(accel)

        obs, reward, done, info = env.step(accel)
        step += 1

        if done:
            episode_done_info = info

    # Determine collision labels for each timestep
    # collision_label[t] = 1 if collision happened within [t, t+K]
    total_steps = len(trajectory)
    collision_happened_at = total_steps if not episode_done_info.get("collision") else total_steps

    # If episode ended in collision, the collision is at the last step
    if episode_done_info.get("collision"):
        collision_happened_at = total_steps - 1

    samples = []

    for t in range(total_steps - 1):  # Need t and t+1
        # Collision label: was there a collision within next K steps?
        steps_to_collision = collision_happened_at - t
        if steps_to_collision <= collision_horizon and steps_to_collision >= 0:
            collision_label = 1.0
        else:
            collision_label = 0.0

        # Build graph for time t
        x_t, edge_index, raw_data_t = build_graph_features(trajectory[t])

        # Build graph for time t+1 (for temporal encoding)
        x_t_dt, _, raw_data_dt = build_graph_features(trajectory[t + 1])

        # Compute physics features for each obstacle
        robot_pos = raw_data_t["robot_pos"]
        robot_vel = raw_data_t["robot_vel"]
        ttc_values = []
        closing_speeds = []

        for i in range(len(raw_data_t["obstacle_positions"])):
            obs_pos = raw_data_t["obstacle_positions"][i]
            obs_vel = raw_data_t["obstacle_velocities"][i]
            combined_r = 0.12 + raw_data_t["obstacle_radii"][i]

            ttc = compute_ttc_value(robot_pos, robot_vel, obs_pos, obs_vel, radius=combined_r)
            cs = compute_closing_speed(robot_pos, robot_vel, obs_pos, obs_vel)

            ttc_values.append(min(ttc, 10.0))  # Cap at 10s
            closing_speeds.append(cs)

        samples.append({
            "x_t": x_t,
            "x_t_dt": x_t_dt,
            "edge_index": edge_index,
            "collision_label": torch.tensor(collision_label, dtype=torch.float32),
            "ttc_values": torch.tensor(ttc_values, dtype=torch.float32),
            "closing_speeds": torch.tensor(closing_speeds, dtype=torch.float32),
        })

    return samples, episode_done_info


def main():
    parser = argparse.ArgumentParser(description="Collect collision data for GNN training")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--horizon", type=int, default=40, help="Collision look-ahead (steps)")
    parser.add_argument("--output", type=str, default="training/collision_data.pt", help="Output path")
    args = parser.parse_args()

    # Resolve output path relative to experiments_3d/
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    output_path = os.path.join(base_dir, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    env = DroneNav3DEnv(gui=False)

    all_samples = []
    total_collisions = 0
    total_successes = 0

    print("=" * 60)
    print("Physics-Informed GNN Data Collection")
    print(f"Episodes: {args.episodes} | Max steps: {args.max_steps}")
    print(f"Collision horizon: {args.horizon} steps ({args.horizon * 0.05:.1f}s)")
    print(f"Output: {output_path}")
    print("=" * 60)

    for ep in range(1, args.episodes + 1):
        samples, info = collect_episode(env, args.max_steps, args.horizon)
        all_samples.extend(samples)

        if info.get("collision"):
            total_collisions += 1
        if info.get("success"):
            total_successes += 1

        if ep % 20 == 0:
            pos_count = sum(1 for s in all_samples if s["collision_label"].item() > 0.5)
            neg_count = len(all_samples) - pos_count
            print(f"  Ep {ep:3d}/{args.episodes} | Samples: {len(all_samples)} "
                  f"| Positive: {pos_count} | Negative: {neg_count} "
                  f"| Collisions: {total_collisions} | Successes: {total_successes}")

    # Save dataset
    # Separate into tensors for efficient loading
    dataset = {
        "x_t": torch.stack([s["x_t"] for s in all_samples]),
        "x_t_dt": torch.stack([s["x_t_dt"] for s in all_samples]),
        "edge_index": all_samples[0]["edge_index"],  # Same topology for all
        "collision_labels": torch.stack([s["collision_label"] for s in all_samples]),
        "ttc_values": torch.stack([s["ttc_values"] for s in all_samples]),
        "closing_speeds": torch.stack([s["closing_speeds"] for s in all_samples]),
        "num_samples": len(all_samples),
    }

    torch.save(dataset, output_path)

    pos_count = (dataset["collision_labels"] > 0.5).sum().item()
    neg_count = len(all_samples) - pos_count

    print(f"\n{'='*60}")
    print(f"DATA COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total samples:  {len(all_samples)}")
    print(f"  Positive (collision):  {pos_count} ({pos_count/len(all_samples)*100:.1f}%)")
    print(f"  Negative (safe):       {neg_count} ({neg_count/len(all_samples)*100:.1f}%)")
    print(f"  Total episodes:        {args.episodes}")
    print(f"  Collision episodes:    {total_collisions}")
    print(f"  Success episodes:      {total_successes}")
    print(f"  Saved to: {output_path}")
    print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    main()
