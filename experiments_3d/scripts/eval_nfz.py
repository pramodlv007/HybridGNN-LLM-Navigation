"""
NFZ Evaluation Runner
========================
Runs the 3D drone navigation with an irregular L-shaped No-Fly Zone.
Records POV videos and reports metrics including NFZ violations.
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.drone_nav_nfz import DroneNavNFZEnv
from controllers.hybrid_nfz_controller import HybridNFZController
from controllers.apex_baseline import ApexController


def add_hud(frame, step, info, controller_name, episode, in_nfz=False):
    """Add HUD overlay to the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    dist = info.get("dist_to_goal", 0)
    status = "SUCCESS" if info.get("success") else \
             "COLLISION" if info.get("collision") else \
             "IN NFZ!" if in_nfz else "navigating..."

    color = (0, 255, 0) if info.get("success") else \
            (0, 0, 255) if info.get("collision") else \
            (0, 100, 255) if in_nfz else (255, 255, 255)

    cv2.putText(frame, f"NFZ Experiment | {controller_name.upper()} | Ep {episode}",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Step: {step:4d} | Dist: {dist:.2f}m | {status}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # NFZ warning banner
    if in_nfz:
        cv2.putText(frame, "!! NFZ VIOLATION !!", (w//2 - 80, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame


def run_nfz_eval(controller_name="hybrid", episodes=5, max_steps=500,
                 camera="pov", fps=20, record=True):
    """Run NFZ evaluation."""
    env = DroneNavNFZEnv(gui=False)

    if controller_name == "hybrid":
        ctrl = HybridNFZController()
    else:
        ctrl = ApexController()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "videos")
    os.makedirs(video_dir, exist_ok=True)

    print("=" * 50)
    print(f"  3D NFZ EXPERIMENT")
    print(f"  Controller: {controller_name.upper()}")
    print(f"  Episodes: {episodes} | Max Steps: {max_steps}")
    print(f"  Recording videos to: {video_dir}")
    print(f"  Camera: {camera} | FPS: {fps}")
    print("=" * 50)

    successes = 0
    collisions = 0
    timeouts = 0
    nfz_violations = 0
    total_steps = []
    goal_distances = []

    render_every = 1

    for ep in tqdm(range(episodes), desc=f"NFZ {controller_name.upper()} episodes"):
        obs = env.reset()
        done = False
        steps = 0
        info = {}
        ep_nfz_violation = False

        video_writer = None
        video_path = None
        if record:
            video_path = os.path.join(
                video_dir,
                f"3d_nfz_{controller_name}_ep{ep+1}_{timestamp}.mp4"
            )
            video_writer = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (640, 480)
            )

        while not done and steps < max_steps:
            action = ctrl.act(obs)
            obs, rew, done, info = env.step(action)
            steps += 1

            if info.get("nfz_violation"):
                ep_nfz_violation = True

            in_nfz = info.get("in_nfz", False)

            if record and video_writer and steps % render_every == 0:
                frame = env.render(width=640, height=480, camera=camera)
                frame = add_hud(frame, steps, info, controller_name, ep + 1,
                                in_nfz=in_nfz)
                video_writer.write(frame)

        if video_writer:
            video_writer.release()

        # Tally
        if info.get("success"):
            successes += 1
            total_steps.append(steps)
            outcome = "success"
        elif info.get("collision"):
            collisions += 1
            outcome = "collision"
        else:
            timeouts += 1
            outcome = "timeout"

        if ep_nfz_violation:
            nfz_violations += 1

        goal_distances.append(info.get("dist_to_goal", 0))

        nfz_str = " | NFZ VIOLATION" if ep_nfz_violation else ""
        tqdm.write(f"  Ep {ep+1}: {outcome} | {steps} steps | "
                   f"dist={info.get('dist_to_goal',0):.2f}m{nfz_str}"
                   f" | video: {os.path.basename(video_path) if video_path else 'none'}")

    # Summary
    print("\n" + "=" * 50)
    print(f"  NFZ RESULTS ({controller_name.upper()})")
    print("=" * 50)
    print(f"  Success Rate:      {successes}/{episodes} ({100*successes/episodes:.1f}%)")
    print(f"  Collision Rate:    {collisions}/{episodes} ({100*collisions/episodes:.1f}%)")
    print(f"  Timeout Rate:      {timeouts}/{episodes} ({100*timeouts/episodes:.1f}%)")
    print(f"  NFZ Violations:    {nfz_violations}/{episodes} ({100*nfz_violations/episodes:.1f}%)")
    if total_steps:
        print(f"  Avg Steps (succ):  {np.mean(total_steps):.0f}")
    print(f"  Avg Final Dist:    {np.mean(goal_distances):.3f}m")
    print("=" * 50)

    if record:
        print(f"\n  Videos saved to: {video_dir}")
        for ep in range(episodes):
            vname = f"3d_nfz_{controller_name}_ep{ep+1}_{timestamp}.mp4"
            print(f"    → {vname}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D NFZ Drone Navigation Experiment")
    parser.add_argument("--controller", type=str, default="hybrid",
                        choices=["hybrid", "apex"],
                        help="Controller: 'hybrid' (NFZ-aware) or 'apex' (baseline)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera", type=str, default="pov",
                        choices=["pov", "front", "top", "follow"])
    args = parser.parse_args()

    run_nfz_eval(
        controller_name=args.controller,
        episodes=args.episodes,
        max_steps=args.max_steps,
        camera=args.camera,
        fps=args.fps,
        record=not args.no_record,
    )
