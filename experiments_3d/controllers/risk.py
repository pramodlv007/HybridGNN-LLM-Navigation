"""
Time-to-Collision (TTC) Risk Module
=====================================
Computes TTC-based collision risk between the drone and an obstacle.
Mirrors the TTC risk patterns used in the 2D APEX experiments.
"""

import numpy as np


def ttc_risk(pos, vel, obs_pos, obs_vel, radius=0.5):
    """
    Compute TTC-based collision risk score in [0, 1].

    Args:
        pos: Drone position (3D)
        vel: Drone velocity (3D)
        obs_pos: Obstacle position (3D)
        obs_vel: Obstacle velocity (3D)
        radius: Combined collision radius (drone + obstacle)

    Returns:
        Risk score in [0, 1]. Higher = more dangerous.
    """
    rel_pos = np.array(obs_pos) - np.array(pos)
    rel_vel = np.array(obs_vel) - np.array(vel)

    dist = np.linalg.norm(rel_pos)

    # Already colliding
    if dist < radius:
        return 1.0

    # Closing speed: positive means objects are approaching each other
    # dot(rel_pos_unit, rel_vel) < 0 means closing
    if dist < 1e-6:
        return 1.0

    closing_speed = -np.dot(rel_pos, rel_vel) / dist

    # Not approaching — diverging or parallel
    if closing_speed <= 0:
        return 0.0

    # Time to closest approach
    ttc = (dist - radius) / closing_speed

    # Risk based on TTC
    if ttc < 0.3:
        return 1.0
    elif ttc < 2.0:
        return max(0.0, 1.0 - (ttc - 0.3) / 1.7)  # Linear decay
    return 0.0


def compute_obstacle_risk(pos, vel, obstacles):
    """
    Compute aggregate risk from all obstacles.

    Args:
        pos: Drone position (3D)
        vel: Drone velocity (3D)
        obstacles: List of obstacle dicts with 'pos', 'vel', 'radius'

    Returns:
        Total risk score (sum of individual TTC risks)
    """
    total_risk = 0.0
    for obs in obstacles:
        combined_radius = 0.12 + obs["radius"]  # drone radius + obstacle radius
        risk = ttc_risk(pos, vel, obs["pos"], obs["vel"], radius=combined_radius)
        total_risk += risk
    return total_risk


def proximity_risk(pos, obstacles, influence_radius=2.0):
    """
    Compute proximity-based risk (inverse square repulsion).

    Args:
        pos: Drone position (3D)
        obstacles: List of obstacle dicts
        influence_radius: Max distance for risk influence

    Returns:
        Total proximity risk score
    """
    risk = 0.0
    for obs in obstacles:
        obs_pos = np.array(obs["pos"])
        dist = np.linalg.norm(pos - obs_pos)
        if dist < 0.1:
            dist = 0.1
        if dist < influence_radius:
            risk += 1.0 / (dist * dist)
    return risk
