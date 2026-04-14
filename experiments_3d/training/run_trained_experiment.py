"""
Trained GNN vs Heuristic Comparison Experiment
================================================
Runs the 3D drone experiment with both:
  1. Heuristic GNN (original, default)
  2. Trained Physics-Informed GNN

Compares success rates, collision rates, and average steps.

Usage:
  python training/run_trained_experiment.py
"""

import os
import sys
import numpy as np
import cv2
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from envs.drone_nav_3d import DroneNav3DEnv
from controllers.hybrid_gnn_llm_controller import HybridGNNLLMController


def add_hud(frame, step, info, label, ep):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    dist = info.get('dist_to_goal', 0)
    text = f"[{label}] Ep {ep} | Step {step} | Dist: {dist:.2f}m"
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


def run_episodes(ctrl, env, label, episodes=5, max_steps=600, save_videos=True,
                 video_dir=".", timestamp=""):
    """Run episodes with a given controller and collect results."""
    results = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        steps = 0
        info = {}
        frames = []

        while not done and steps < max_steps:
            action = ctrl.act(obs)
            obs, reward, done, info = env.step(action)
            steps += 1

            if save_videos:
                frame = env.render(camera="chase")
                frame = add_hud(frame, steps, info, label, ep)
                frames.append(frame)

        status = "SUCCESS" if info.get("success") else "COLLISION" if info.get("collision") else "TIMEOUT"
        dist = info.get('dist_to_goal', 0)

        print(f"    Ep {ep}: {status:9s} | {steps:3d} steps | dist={dist:.2f}m")

        results.append({"status": status, "steps": steps, "dist": dist})

        if save_videos and frames:
            safe_label = label.replace(" ", "_").replace("+", "").lower()
            v_name = f"compare_{safe_label}_ep{ep}_{timestamp}.mp4"
            v_path = os.path.join(video_dir, v_name)
            h, w = frames[0].shape[:2]
            out = cv2.VideoWriter(v_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
            for f in frames:
                out.write(f)
            out.release()

    return results


def print_summary(label, results):
    """Print summary for one method."""
    total = len(results)
    successes = sum(1 for r in results if r['status'] == 'SUCCESS')
    collisions = sum(1 for r in results if r['status'] == 'COLLISION')
    timeouts = sum(1 for r in results if r['status'] == 'TIMEOUT')
    succ_steps = [r['steps'] for r in results if r['status'] == 'SUCCESS']

    print(f"  {label}:")
    print(f"    Success:   {successes}/{total} ({successes/total*100:.0f}%)")
    print(f"    Collision: {collisions}/{total} ({collisions/total*100:.0f}%)")
    print(f"    Timeout:   {timeouts}/{total} ({timeouts/total*100:.0f}%)")
    if succ_steps:
        print(f"    Avg Steps: {np.mean(succ_steps):.0f}")
    return successes / total


def main():
    episodes = 5
    max_steps = 600

    video_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "3d expt"))
    os.makedirs(video_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    env = DroneNav3DEnv(gui=False)

    print("=" * 60)
    print("COMPARISON: Heuristic vs Trained GNN")
    print(f"Episodes: {episodes} | Max steps: {max_steps}")
    print("=" * 60)

    # --- Method 1: Heuristic (original) ---
    print(f"\n--- Heuristic GNN (original) ---")
    ctrl_heuristic = HybridGNNLLMController(use_trained_model=False)
    results_heuristic = run_episodes(
        ctrl_heuristic, env, "Heuristic", episodes, max_steps,
        save_videos=True, video_dir=video_dir, timestamp=timestamp)

    # --- Method 2: Trained GNN ---
    print(f"\n--- Trained Physics-Informed GNN ---")
    ctrl_trained = HybridGNNLLMController(use_trained_model=True)
    results_trained = run_episodes(
        ctrl_trained, env, "Trained GNN", episodes, max_steps,
        save_videos=True, video_dir=video_dir, timestamp=timestamp)

    # --- Comparison ---
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")

    rate_h = print_summary("Heuristic (original)", results_heuristic)
    print()
    rate_t = print_summary("Trained GNN (physics)", results_trained)

    print(f"\n  Improvement: {(rate_t - rate_h)*100:+.0f}% success rate")
    print(f"  Videos saved in: {video_dir}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
