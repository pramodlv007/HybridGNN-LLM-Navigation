"""
VLM Evaluation Runner — 3D Drone Navigation
=============================================
Runs the VLM controller (GPT-4o vision) in EITHER:
  - Standard environment  (20 obstacles, no NFZ)
  - NFZ environment       (25 obstacles + L-shaped hard wall)

Uses existing VLMController, DroneNav3DEnv, DroneNavNFZEnv unchanged.
The VLM controller needs a rendered camera frame each step; this script
renders and passes frames via ctrl.set_frame() before calling ctrl.act().

Usage:
    python -m scripts.eval_vlm --episodes 15
    python -m scripts.eval_vlm --episodes 15 --use-nfz
    python -m scripts.eval_vlm --episodes 15 --camera top --no-record
"""

import argparse
import os
import sys
import numpy as np
import cv2
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.vlm_controller import VLMController


# ---------------------------------------------------------------------------
# HUD helpers (mirror style from eval.py / eval_nfz.py)
# ---------------------------------------------------------------------------

def _add_hud_standard(frame, step, info, ep):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    dist = info.get("dist_to_goal", 0)
    if info.get("success"):
        status, color = "SUCCESS", (0, 255, 0)
    elif info.get("collision"):
        status, color = "COLLISION", (0, 0, 255)
    else:
        status, color = "RUNNING", (255, 255, 255)
    cv2.putText(frame,
                f"VLM | Ep {ep} | Step {step} | Dist: {dist:.2f}m | {status}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def _add_hud_nfz(frame, step, info, ep, in_nfz=False):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    dist = info.get("dist_to_goal", 0)
    if info.get("success"):
        status, color = "SUCCESS", (0, 255, 0)
    elif info.get("collision"):
        status, color = "COLLISION", (0, 0, 255)
    elif in_nfz:
        status, color = "IN NFZ!", (0, 100, 255)
    else:
        status, color = "navigating...", (255, 255, 255)
    cv2.putText(frame, f"NFZ Experiment | VLM | Ep {ep}",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Step: {step:4d} | Dist: {dist:.2f}m | {status}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if in_nfz:
        cv2.putText(frame, "!! NFZ VIOLATION !!",
                    (w // 2 - 80, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_vlm_eval(episodes=15, max_steps=500, camera="follow", fps=20,
                 record=True, use_nfz=False):
    """
    Run VLM controller evaluation.

    Returns a dict of aggregated metrics (same schema as the other eval scripts).
    """
    if use_nfz:
        from envs.drone_nav_nfz import DroneNavNFZEnv
        env = DroneNavNFZEnv(gui=False)
        env_label = "VLM_NFZ"
        prefix = "3d_vlm_nfz"
    else:
        from envs.drone_nav_3d import DroneNav3DEnv
        env = DroneNav3DEnv(gui=False)
        env_label = "VLM"
        prefix = "3d_vlm"

    ctrl = VLMController()

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_dir = os.path.join(base_path, "videos", "final_4conditions_10eps")
    os.makedirs(video_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print(f"  3D VLM EXPERIMENT — {env_label}")
    print(f"  Episodes: {episodes} | Max Steps: {max_steps}")
    print(f"  Camera: {camera} | FPS: {fps} | Recording: {record}")
    print("=" * 60)

    successes = 0
    collisions = 0
    timeouts = 0
    nfz_violations = 0
    all_steps = []
    succ_steps = []
    goal_distances = []
    min_obs_dists = []
    vlm_calls_total = 0
    vlm_failures_total = 0
    video_files = []

    for ep in tqdm(range(episodes), desc=f"VLM {'NFZ' if use_nfz else 'standard'} episodes"):
        obs = env.reset()

        # Reset per-episode controller state without touching internals
        ctrl.step_count = 0
        ctrl.last_vlm_time = -10.0
        ctrl.bearing = np.array([1.0, 1.0, 0.0])
        ctrl.vlm_calls = 0
        ctrl.vlm_failures = 0

        done = False
        steps = 0
        info = {}
        ep_nfz_violation = False
        ep_min_obs_dist = float("inf")

        video_writer = None
        video_path = None
        ep_record = record   # Record all episodes
        if ep_record:
            video_path = os.path.join(
                video_dir, f"{prefix}_ep{ep+1}_{timestamp}.mp4")
            video_writer = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (640, 480))

        while not done and steps < max_steps:
            # Render BEFORE act so VLM sees the current frame
            if ep_record:
                frame = env.render(width=640, height=480, camera=camera)
                ctrl.set_frame(frame)
            else:
                ctrl.set_frame(None)

            action = ctrl.act(obs)
            obs, rew, done, info = env.step(action)
            steps += 1

            if info.get("nfz_violation") or info.get("nfz_blocked"):
                ep_nfz_violation = True

            pos = obs.get("pos", np.zeros(3))
            for ob in obs.get("obstacles", []):
                d = np.linalg.norm(pos - ob["pos"]) - ob.get("radius", 0.1)
                ep_min_obs_dist = min(ep_min_obs_dist, d)

            in_nfz = info.get("in_nfz", False)

            if ep_record and video_writer:
                if use_nfz:
                    frame = _add_hud_nfz(frame, steps, info, ep + 1, in_nfz)
                else:
                    frame = _add_hud_standard(frame, steps, info, ep + 1)
                video_writer.write(frame)

        # Hold final frame 1 s
        if ep_record and video_writer:
            frame = env.render(width=640, height=480, camera=camera)
            if use_nfz:
                frame = _add_hud_nfz(frame, steps, info, ep + 1,
                                     info.get("in_nfz", False))
            else:
                frame = _add_hud_standard(frame, steps, info, ep + 1)
            for _ in range(fps):
                video_writer.write(frame)
            video_writer.release()
            video_files.append(video_path)

        all_steps.append(steps)
        goal_distances.append(info.get("dist_to_goal", 0))
        if ep_min_obs_dist < float("inf"):
            min_obs_dists.append(ep_min_obs_dist)
        if ep_nfz_violation:
            nfz_violations += 1
        vlm_calls_total += ctrl.vlm_calls
        vlm_failures_total += ctrl.vlm_failures

        if info.get("success"):
            successes += 1
            succ_steps.append(steps)
            outcome = "SUCCESS"
        elif info.get("collision"):
            collisions += 1
            outcome = "COLLISION"
        else:
            timeouts += 1
            outcome = "TIMEOUT"

        nfz_str = " | NFZ!" if ep_nfz_violation else ""
        tqdm.write(
            f"  Ep {ep+1:2d}: {outcome:9s} | {steps:4d} steps | "
            f"dist={info.get('dist_to_goal', 0):.2f}m"
            f" | vlm_calls={ctrl.vlm_calls}{nfz_str}"
        )

    env.close()

    print("\n" + "=" * 60)
    print(f"  RESULTS — {env_label} ({episodes} episodes)")
    print("=" * 60)
    print(f"  Success Rate:      {successes}/{episodes} ({100*successes/episodes:.1f}%)")
    print(f"  Collision Rate:    {collisions}/{episodes} ({100*collisions/episodes:.1f}%)")
    print(f"  Timeout Rate:      {timeouts}/{episodes} ({100*timeouts/episodes:.1f}%)")
    if use_nfz:
        print(f"  NFZ Violations:    {nfz_violations}/{episodes} ({100*nfz_violations/episodes:.1f}%)")
    print(f"  Avg Steps (all):   {np.mean(all_steps):.1f}")
    if succ_steps:
        print(f"  Avg Steps (succ):  {np.mean(succ_steps):.1f} ± {np.std(succ_steps):.1f}")
    print(f"  Avg Final Dist:    {np.mean(goal_distances):.3f}m")
    if min_obs_dists:
        print(f"  Min Obs Dist:      {np.mean(min_obs_dists):.3f}m")
    print(f"  VLM Calls:         {vlm_calls_total} total | {vlm_failures_total} failures")
    print("=" * 60)

    return {
        "method": env_label,
        "episodes": episodes,
        "successes": successes,
        "collisions": collisions,
        "timeouts": timeouts,
        "nfz_violations": nfz_violations,
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "timeout_rate": timeouts / episodes,
        "nfz_violation_rate": nfz_violations / episodes if use_nfz else 0.0,
        "avg_steps_all": float(np.mean(all_steps)),
        "avg_steps_succ": float(np.mean(succ_steps)) if succ_steps else None,
        "avg_final_dist": float(np.mean(goal_distances)),
        "min_obs_dist": float(np.mean(min_obs_dists)) if min_obs_dists else None,
        "vlm_calls": vlm_calls_total,
        "vlm_failures": vlm_failures_total,
        "video_files": video_files,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D VLM Drone Navigation Evaluation")
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--camera", type=str, default="follow",
                        choices=["pov", "front", "top", "follow"])
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--use-nfz", action="store_true",
                        help="Use NFZ environment (25 obs + hard wall)")
    args = parser.parse_args()

    run_vlm_eval(
        episodes=args.episodes,
        max_steps=args.max_steps,
        camera=args.camera,
        fps=args.fps,
        record=not args.no_record,
        use_nfz=args.use_nfz,
    )
