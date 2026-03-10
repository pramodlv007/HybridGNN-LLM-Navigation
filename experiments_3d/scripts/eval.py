"""
Evaluation Runner — 3D Drone Navigation (Hybrid vs APEX)
==========================================================
Runs episodes with the specified controller, records MP4 videos,
and reports metrics.

Usage:
    python -m scripts.eval --controller hybrid --episodes 20
    python -m scripts.eval --controller apex --episodes 20
"""

import argparse
import sys
import os
import numpy as np
import cv2
from datetime import datetime
from tqdm import tqdm

# Ensure the package root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.drone_nav_3d import DroneNav3DEnv
from controllers.hybrid_controller import HybridController
from controllers.apex_baseline import ApexController


def add_hud(frame, step, info, controller_name, episode):
    """
    Overlay HUD text on frame showing step count, distance, status.
    """
    h, w = frame.shape[:2]
    # Semi-transparent black bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    dist = info.get("dist_to_goal", 0)
    status = "RUNNING"
    color = (255, 255, 255)  # white
    if info.get("success"):
        status = "SUCCESS"
        color = (0, 255, 0)
    elif info.get("collision"):
        status = "COLLISION"
        color = (0, 0, 255)

    text = f"{controller_name.upper()} | Ep {episode} | Step {step} | Dist: {dist:.2f}m | {status}"
    cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def run_evaluation(controller_name, episodes, gui=False, max_steps=500,
                   record=True, fps=20, camera="top"):
    """
    Run evaluation episodes, record videos, and collect metrics.

    Args:
        controller_name: 'hybrid' or 'apex'
        episodes: Number of episodes to run
        gui: Whether to show PyBullet GUI
        max_steps: Max steps per episode
        record: Whether to record MP4 videos
        fps: Video frame rate
        camera: Camera view ('top' or 'follow')

    Returns:
        dict with aggregated metrics
    """
    env = DroneNav3DEnv(gui=gui)

    if controller_name == "hybrid":
        ctrl = HybridController()
        print(f"\n{'='*60}")
        print(f"  HYBRID Controller — Safety Shield + Rollout")
        print(f"{'='*60}")
    else:
        ctrl = ApexController()
        print(f"\n{'='*60}")
        print(f"  APEX Baseline Controller")
        print(f"{'='*60}")

    # Setup video output directory
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_dir = os.path.join(base_path, "videos")
    os.makedirs(video_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if record:
        print(f"  Recording videos to: {video_dir}")
        print(f"  Camera: {camera} | FPS: {fps}")

    successes = 0
    collisions = 0
    timeouts = 0
    total_steps = []
    goal_distances = []
    video_files = []

    # Render frequency: dt=0.05 → 20 steps/sec, render every step for 20fps video
    render_every = 1

    for ep in tqdm(range(episodes), desc=f"{controller_name.upper()} episodes"):
        obs = env.reset()
        done = False
        steps = 0
        info = {}

        # Setup video writer for this episode
        video_writer = None
        video_path = None
        if record:
            video_path = os.path.join(
                video_dir,
                f"3d_{controller_name}_ep{ep+1}_{timestamp}.mp4"
            )
            video_writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (640, 480)
            )

        while not done and steps < max_steps:
            action = ctrl.act(obs)
            obs, rew, done, info = env.step(action)
            steps += 1

            # Record frame
            if record and video_writer and steps % render_every == 0:
                frame = env.render(width=640, height=480, camera=camera)
                frame = add_hud(frame, steps, info, controller_name, ep + 1)
                video_writer.write(frame)

        # Write final frames for end state (hold for 1 sec)
        if record and video_writer:
            frame = env.render(width=640, height=480, camera=camera)
            frame = add_hud(frame, steps, info, controller_name, ep + 1)
            for _ in range(fps):  # Hold 1 second on final frame
                video_writer.write(frame)
            video_writer.release()

        # Track outcomes
        outcome = "timeout"
        if info.get("success", False):
            successes += 1
            total_steps.append(steps)
            outcome = "success"
        elif info.get("collision", False):
            collisions += 1
            outcome = "collision"
        else:
            timeouts += 1

        goal_distances.append(info.get("dist_to_goal", float('inf')))

        if record and video_path:
            video_files.append(video_path)
            tqdm.write(f"  Ep {ep+1}: {outcome} | {steps} steps | "
                       f"dist={info.get('dist_to_goal', 0):.2f}m | "
                       f"video: {os.path.basename(video_path)}")

    env.close()

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"  RESULTS — {controller_name.upper()} ({episodes} episodes)")
    print(f"{'='*60}")
    print(f"  Success Rate:   {successes}/{episodes} ({100*successes/episodes:.1f}%)")
    print(f"  Collision Rate: {collisions}/{episodes} ({100*collisions/episodes:.1f}%)")
    print(f"  Timeout Rate:   {timeouts}/{episodes} ({100*timeouts/episodes:.1f}%)")

    if total_steps:
        print(f"  Avg Steps (success): {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
    else:
        print(f"  Avg Steps (success): N/A (no successes)")

    avg_dist = np.mean(goal_distances)
    print(f"  Avg Final Dist to Goal: {avg_dist:.3f}m")

    if record:
        print(f"\n  Videos saved to: {video_dir}")
        for vf in video_files:
            print(f"    → {os.path.basename(vf)}")

    print(f"{'='*60}\n")

    return {
        "controller": controller_name,
        "episodes": episodes,
        "successes": successes,
        "collisions": collisions,
        "timeouts": timeouts,
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "avg_steps": np.mean(total_steps) if total_steps else None,
        "avg_final_dist": avg_dist,
        "video_files": video_files,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Drone Navigation — Hybrid vs APEX Evaluation"
    )
    parser.add_argument(
        "--controller", type=str, default="hybrid",
        choices=["hybrid", "apex"],
        help="Controller to evaluate: 'hybrid' or 'apex'"
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Show PyBullet GUI (slows execution)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=500,
        help="Max steps per episode before timeout"
    )
    parser.add_argument(
        "--no-record", action="store_true",
        help="Disable video recording"
    )
    parser.add_argument(
        "--fps", type=int, default=20,
        help="Video frame rate (default: 20)"
    )
    parser.add_argument(
        "--camera", type=str, default="pov",
        choices=["pov", "front", "top", "follow"],
        help="Camera view: 'pov' (robot POV, default), 'front' (perspective), 'top' (overhead), 'follow' (behind)"
    )
    args = parser.parse_args()

    results = run_evaluation(
        controller_name=args.controller,
        episodes=args.episodes,
        gui=args.gui,
        max_steps=args.max_steps,
        record=not args.no_record,
        fps=args.fps,
        camera=args.camera,
    )
