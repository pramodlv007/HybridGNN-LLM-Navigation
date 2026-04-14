"""
3D Drone Navigation Experiment Runner
Runs the Hybrid GNN-LLM controller and saves video to '3d expt' folder.
"""

import os
import sys
import numpy as np
import cv2
from datetime import datetime

# Ensure experiments_3d is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.drone_nav_3d import DroneNav3DEnv
from controllers.hybrid_gnn_llm_controller import HybridGNNLLMController


def add_hud(frame, step, info, label, ep):
    """Simple HUD overlay."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    dist = info.get('dist_to_goal', 0)
    text = f"[{label}] Ep {ep} | Step {step} | Dist: {dist:.2f}m"
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


def run_experiment(episodes=3, max_steps=500, camera="chase", fps=20):
    """Run 3D experiment with Hybrid GNN-LLM controller."""
    
    # Output directory
    video_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "3d expt"))
    os.makedirs(video_dir, exist_ok=True)
    
    env = DroneNav3DEnv(gui=False)
    ctrl = HybridGNNLLMController()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("3D DRONE NAVIGATION — Hybrid GNN+LLM Controller")
    print(f"Episodes: {episodes} | Max Steps: {max_steps}")
    print(f"Camera: {camera} | FPS: {fps}")
    print(f"Video output: {video_dir}")
    print("=" * 60)

    all_results = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        steps = 0
        info = {}
        frames = []

        print(f"\n--- Episode {ep} ---")

        while not done and steps < max_steps:
            action = ctrl.act(obs)
            obs, reward, done, info = env.step(action)
            steps += 1

            # Log every 50 steps
            if steps % 50 == 0:
                d = info.get('dist_to_goal', 0)
                pos = obs['pos']
                print(f"  Step {steps:3d} | Pos: [{pos[0]:.2f}, {pos[1]:.2f}] | Dist: {d:.2f}m")

            # Render frame
            frame = env.render(camera=camera)
            frame = add_hud(frame, steps, info, "GNN+LLM", ep)
            frames.append(frame)

        # Determine outcome
        status = "SUCCESS" if info.get("success") else "COLLISION" if info.get("collision") else "TIMEOUT"
        dist = info.get('dist_to_goal', 0)
        print(f"  Result: {status} | Steps: {steps} | Final dist: {dist:.2f}m")

        all_results.append({
            "episode": ep,
            "status": status,
            "steps": steps,
            "dist": dist
        })

        # Save video
        if frames:
            v_name = f"3d_hybrid_gnn_llm_ep{ep}_{timestamp}.mp4"
            v_path = os.path.join(video_dir, v_name)
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(v_path, fourcc, fps, (w, h))
            for f in frames:
                out.write(f)
            out.release()
            print(f"  Video saved: {v_path}")

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    successes = sum(1 for r in all_results if r['status'] == 'SUCCESS')
    collisions = sum(1 for r in all_results if r['status'] == 'COLLISION')
    timeouts = sum(1 for r in all_results if r['status'] == 'TIMEOUT')
    print(f"  Success:   {successes}/{episodes}")
    print(f"  Collision: {collisions}/{episodes}")
    print(f"  Timeout:   {timeouts}/{episodes}")
    succ_steps = [r['steps'] for r in all_results if r['status'] == 'SUCCESS']
    if succ_steps:
        print(f"  Avg Steps (success): {np.mean(succ_steps):.1f}")
    print(f"  Videos saved in: {video_dir}")
    print("=" * 60)

    env.close()
    return all_results


if __name__ == "__main__":
    run_experiment(episodes=3, max_steps=500, camera="chase", fps=20)
