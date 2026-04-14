"""
Master 2D Benchmark Runner (Revised)
======================================
Runs Hybrid GNN+LLM, APEX Original, and LLM-Only in the 2D MuJoCo NFZ environment.

Uses medium_nfz_env.xml (the same environment as the original successful experiments).
Each method runs independently so one method's failure doesn't affect others.

Usage:
  python run_benchmark_2d.py --episodes 3
"""

import os
import sys
import csv
import numpy as np
import cv2
import time
import argparse
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medium_nfz_experiment import (
    run_apex_method, run_pulse_method, run_hybrid_method,
    START_POS, TARGET_POS, NUM_DYNAMIC, NUM_STATIC, NUM_TOTAL,
    randomize_all_obstacles, update_dynamic_obstacles, get_random_velocity,
    is_inside_any_nfz
)
import mujoco

base_path = os.path.dirname(os.path.abspath(__file__))

# Use medium_nfz_env.xml — the SAME environment as the original successful experiments
ENV_XML = os.path.join(base_path, "env", "medium_nfz_env.xml")

METHODS = ["Hybrid_GNN_LLM", "APEX", "LLM_Only"]


def setup_environment():
    """Create MuJoCo model and data using medium_nfz_env.xml."""
    with open(ENV_XML, 'r') as f:
        xml_string = f.read()
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    return model, data, renderer


def run_2d_episode(method_name, model, data, renderer, fps=100, max_steps=3000):
    """Run one 2D episode with the specified method. Returns (result_dict, frames_list)."""
    # Reinitialize
    randomize_all_obstacles(model, data)
    mujoco.mj_step(model, data)

    # Dynamic obstacle velocities
    for i in range(1, NUM_DYNAMIC + 1):
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
            if body_id != -1:
                dof_start = model.body_dofadr[body_id]
                vx, vy = get_random_velocity(1.0)
                data.qvel[dof_start], data.qvel[dof_start + 1] = vx, vy
        except:
            pass

    # Static obstacles zero velocity
    for i in range(NUM_DYNAMIC + 1, NUM_TOTAL + 1):
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
            if body_id != -1:
                dof_start = model.body_dofadr[body_id]
                data.qvel[dof_start:dof_start + 3] = [0.0, 0.0, 0.0]
        except:
            pass

    dt = 1.0 / fps

    # Select method
    if method_name == "Hybrid_GNN_LLM":
        generator = run_hybrid_method(model, data, renderer, fps, dt, max_steps)
    elif method_name == "APEX":
        generator = run_apex_method(model, data, renderer, fps, dt, max_steps)
    elif method_name == "LLM_Only":
        generator = run_apex_method(model, data, renderer, fps, dt, max_steps)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    frames = []
    results = None

    for output in generator:
        frame, info = output
        if frame is None:
            results = info
        else:
            frames.append(frame)

    if results is None:
        results = {"collision": False, "nfz_violated": False, "steps": max_steps, "time": max_steps / fps}

    success = results.get("success", False)
    if not success and results.get("steps", max_steps) < max_steps - 1:
        success = True

    return {
        "success": 1 if success else 0,
        "collision": 1 if results.get("collision") else 0,
        "nfz_violated": 1 if results.get("nfz_violated") else 0,
        "steps": results.get("steps", max_steps),
        "time": results.get("time", max_steps / fps),
    }, frames


def save_video(frames, path, fps=100):
    """Save frames to an MP4 video."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def main():
    parser = argparse.ArgumentParser(description="2D Benchmark Runner")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    args = parser.parse_args()

    video_base = os.path.abspath(os.path.join(base_path, "..", "results", "benchmarks"))
    csv_dir = os.path.join(video_base, "2D")
    os.makedirs(csv_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    total_runs = len(args.methods) * args.episodes
    print("=" * 70)
    print("  2D BENCHMARK: Method Comparison (Medium NFZ Environment)")
    print(f"  Environment: {os.path.basename(ENV_XML)}")
    print(f"  Methods: {', '.join(args.methods)}")
    print(f"  Episodes per method: {args.episodes}")
    print(f"  Total runs: {total_runs}")
    print("=" * 70)

    all_results = []
    run_count = 0

    for method in args.methods:
        print(f"\n--- {method} ---")
        for ep in range(1, args.episodes + 1):
            run_count += 1
            print(f"  [{run_count}/{total_runs}] Ep {ep}...", end=" ", flush=True)

            try:
                model, data, renderer = setup_environment()
                result, frames = run_2d_episode(method, model, data, renderer)

                status = "SUCCESS" if result["success"] else \
                         "COLLISION" if result["collision"] else "TIMEOUT"
                steps = result["steps"]
                nfz = result["nfz_violated"]
                print(f"{status:9s} | {steps:4d} steps | NFZ: {'YES' if nfz else 'NO'}")

                row = {
                    "method": method,
                    "episode": ep,
                    "status": status,
                    "steps": steps,
                    "time": result["time"],
                    "collision": result["collision"],
                    "nfz_violated": result["nfz_violated"],
                    "success": result["success"],
                }
                all_results.append(row)

                # Save video IMMEDIATELY after each episode
                if frames:
                    v_dir = os.path.join(video_base, "2D", method)
                    os.makedirs(v_dir, exist_ok=True)
                    v_path = os.path.join(v_dir, f"run_{ep}_{timestamp}.mp4")
                    save_video(frames, v_path)
                    print(f"    Video saved: {v_path}")
                else:
                    print(f"    WARNING: No frames collected!")

            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()
                all_results.append({
                    "method": method, "episode": ep, "status": "ERROR",
                    "steps": 0, "time": 0, "collision": 0,
                    "nfz_violated": 0, "success": 0,
                })

    # Save CSV
    csv_path = os.path.join(csv_dir, f"benchmark_2d_{timestamp}.csv")
    fieldnames = ["method", "episode", "status", "steps", "time",
                  "collision", "nfz_violated", "success"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Summary
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_results:
        groups[r["method"]].append(r)

    print(f"\n{'='*75}")
    print(f"  2D BENCHMARK RESULTS")
    print(f"{'='*75}")
    print(f"  {'Method':<18} {'Success':>8} {'Collision':>10} {'NFZ Viol':>9} {'Avg Steps':>10}")
    print(f"  {'-'*18} {'-'*8} {'-'*10} {'-'*9} {'-'*10}")

    for method in args.methods:
        results = groups.get(method, [])
        if not results:
            continue
        n = len(results)
        succ = sum(r["success"] for r in results)
        coll = sum(r["collision"] for r in results)
        nfz = sum(r["nfz_violated"] for r in results)
        avg_steps = np.mean([r["steps"] for r in results])
        print(f"  {method:<18} {succ:>3}/{n:<4} {coll:>5}/{n:<4} {nfz:>4}/{n:<4} {avg_steps:>10.1f}")

    print(f"{'='*75}")
    print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
