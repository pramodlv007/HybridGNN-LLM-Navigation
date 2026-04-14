"""
Master 3D Benchmark Runner
============================
Runs 4 methods × N episodes and collects metrics.

Methods: Hybrid GNN+LLM, LLM-Only, VLM, APEX
(3D env has no NFZ support, so only obstacle avoidance is tested)

Usage:
  python run_benchmark_3d.py --episodes 10
  python run_benchmark_3d.py --episodes 1   # smoke test
"""

import os
import sys
import csv
import numpy as np
import cv2
import math
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.drone_nav_3d import DroneNav3DEnv
from controllers.hybrid_gnn_llm_controller import HybridGNNLLMController
from controllers.apex_baseline import ApexController
from controllers.explicit_llm_controller import ExplicitLLMController
from controllers.vlm_controller import VLMController


METHODS = ["Hybrid_GNN_LLM", "LLM_Only", "VLM", "APEX"]


def add_hud(frame, step, info, method, ep):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    dist = info.get('dist_to_goal', 0)
    text = f"[{method}] Ep{ep} Step{step} Dist:{dist:.2f}m"
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)
    return frame


def create_controller(method_name):
    if method_name == "Hybrid_GNN_LLM":
        return HybridGNNLLMController()
    elif method_name == "LLM_Only":
        return ExplicitLLMController(fps=20)
    elif method_name == "VLM":
        return VLMController(fps=20)
    elif method_name == "APEX":
        return ApexController()
    else:
        raise ValueError(f"Unknown method: {method_name}")


def run_single_episode(env, ctrl, method_name, ep_num,
                       max_steps=600, video_dir=None, timestamp=""):
    obs = env.reset()
    done = False
    steps = 0
    info = {}
    frames = []
    path_length = 0.0
    min_obs_dist = float('inf')
    prev_pos = obs["pos"].copy()

    while not done and steps < max_steps:
        # For VLM: provide the camera frame before acting
        if method_name == "VLM" and hasattr(ctrl, 'set_frame'):
            frame = env.render(camera="chase")
            ctrl.set_frame(frame)

        action = ctrl.act(obs)
        obs, reward, done, info = env.step(action)
        steps += 1

        # Metrics
        curr_pos = obs["pos"]
        path_length += np.linalg.norm(curr_pos - prev_pos)
        prev_pos = curr_pos.copy()

        for ob in obs["obstacles"]:
            d = np.linalg.norm(curr_pos - ob["pos"]) - ob["radius"]
            if d < min_obs_dist:
                min_obs_dist = d

        # Render
        frame = env.render(camera="chase")
        frame = add_hud(frame, steps, info, method_name, ep_num)
        frames.append(frame)

    status = "SUCCESS" if info.get("success") else \
             "COLLISION" if info.get("collision") else "TIMEOUT"
    dist = info.get('dist_to_goal', 0)

    straight_line = np.linalg.norm(
        np.array([3.6, 3.6, 1.0]) - np.array([-3.6, -3.6, 1.0]))
    path_efficiency = path_length / straight_line if straight_line > 0 else 0

    result = {
        "method": method_name,
        "episode": ep_num,
        "status": status,
        "steps": steps,
        "dist_to_goal": round(dist, 3),
        "path_length": round(path_length, 3),
        "path_efficiency": round(path_efficiency, 3),
        "min_obs_dist": round(max(min_obs_dist, 0), 3),
        "success": 1 if status == "SUCCESS" else 0,
        "collision": 1 if status == "COLLISION" else 0,
    }

    # Save video
    if video_dir and frames:
        v_dir = os.path.join(video_dir, "3D", method_name)
        os.makedirs(v_dir, exist_ok=True)
        v_name = f"run_{ep_num}_{timestamp}.mp4"
        v_path = os.path.join(v_dir, v_name)
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(v_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
        for f in frames:
            out.write(f)
        out.release()

    return result


def print_summary_table(all_results):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_results:
        groups[r["method"]].append(r)

    print(f"\n{'='*85}")
    print(f"  3D BENCHMARK RESULTS")
    print(f"{'='*85}")
    print(f"  {'Method':<18} {'Success':>8} {'Collision':>10} "
          f"{'Timeout':>8} {'Avg Steps':>10} {'Path Eff':>9} {'Min Dist':>9}")
    print(f"  {'-'*18} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*9} {'-'*9}")

    for method in METHODS:
        results = groups.get(method, [])
        if not results:
            continue
        n = len(results)
        succ = sum(r["success"] for r in results)
        coll = sum(r["collision"] for r in results)
        tout = n - succ - coll
        avg_steps = np.mean([r["steps"] for r in results])
        avg_pe = np.mean([r["path_efficiency"] for r in results])
        avg_md = np.mean([r["min_obs_dist"] for r in results])

        print(f"  {method:<18} {succ:>3}/{n:<4} {coll:>5}/{n:<4} "
              f"{tout:>3}/{n:<4} {avg_steps:>10.1f} {avg_pe:>9.2f} {avg_md:>9.3f}")

    print(f"{'='*85}")


def main():
    parser = argparse.ArgumentParser(description="3D Benchmark Runner")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    video_dir = os.path.join(base_dir, "results", "benchmarks")
    csv_dir = os.path.join(video_dir, "3D")
    os.makedirs(csv_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env = DroneNav3DEnv(gui=False)

    total_runs = len(args.methods) * args.episodes
    print("=" * 70)
    print("  3D BENCHMARK: 4-Method Comparison")
    print(f"  Methods: {', '.join(args.methods)}")
    print(f"  Episodes per method: {args.episodes}")
    print(f"  Total runs: {total_runs}")
    print(f"  Videos: {video_dir}")
    print("=" * 70)

    all_results = []
    run_count = 0

    for method in args.methods:
        print(f"\n--- {method} ---")
        for ep in range(1, args.episodes + 1):
            run_count += 1
            print(f"  [{run_count}/{total_runs}] Ep {ep}...", end=" ", flush=True)

            try:
                ctrl = create_controller(method)
                result = run_single_episode(
                    env, ctrl, method, ep,
                    max_steps=args.max_steps,
                    video_dir=video_dir, timestamp=timestamp)

                status = result["status"]
                steps = result["steps"]
                dist = result["dist_to_goal"]
                print(f"{status:9s} | {steps:3d} steps | dist={dist:.2f}m")

                all_results.append(result)

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback; traceback.print_exc()
                all_results.append({
                    "method": method, "episode": ep,
                    "status": "ERROR", "steps": 0, "dist_to_goal": 99,
                    "path_length": 0, "path_efficiency": 0,
                    "min_obs_dist": 0, "success": 0, "collision": 0,
                })

    # Save CSV
    csv_path = os.path.join(csv_dir, f"benchmark_3d_{timestamp}.csv")
    fieldnames = ["method", "episode", "status", "steps", "dist_to_goal",
                  "path_length", "path_efficiency", "min_obs_dist",
                  "success", "collision"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nCSV saved: {csv_path}")
    print_summary_table(all_results)
    env.close()
    return all_results


if __name__ == "__main__":
    main()
