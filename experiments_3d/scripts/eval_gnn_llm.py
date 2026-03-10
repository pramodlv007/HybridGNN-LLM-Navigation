"""
Hybrid GNN-LLM Evaluation Runner (3D)
=====================================
Runs the Hybrid GNN-LLM controller in EITHER:
  - Standard environment (20 obstacles, no NFZ)
  - NFZ environment (25 obstacles + L-shaped hard wall)
Tracks success, collisions, and NFZ violations.
"""

import argparse
import numpy as np
import cv2
import os
from datetime import datetime


def add_hud_simple(frame, step, info, label, ep):
    """Simple HUD overlay for standard env."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    text = f"[{label}] Ep {ep} | Step {step} | Dist: {info.get('dist_to_goal', 0):.2f}m"
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


def run_gnn_llm_eval(episodes=5, max_steps=500, camera="pov", fps=20, record=True, use_nfz=True):
    """Main evaluation loop for GNN-LLM."""
    from controllers.hybrid_gnn_llm_controller import HybridGNNLLMController

    if use_nfz:
        from envs.drone_nav_nfz import DroneNavNFZEnv
        env = DroneNavNFZEnv(gui=False)
        env_label = "GNN-LLM (NFZ)"
        prefix = "3d_gnn_llm_nfz"
    else:
        from envs.drone_nav_3d import DroneNav3DEnv
        env = DroneNav3DEnv(gui=False)
        env_label = "GNN-LLM"
        prefix = "3d_gnn_llm"

    ctrl = HybridGNNLLMController()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = "videos"
    os.makedirs(video_dir, exist_ok=True)

    results = {
        "success": 0,
        "collision": 0,
        "timeout": 0,
        "nfz_blocked": 0,
        "steps": [],
        "dist": []
    }

    print(f"\n  3D {env_label} EXPERIMENT")
    print(f"  Episodes: {episodes} | Max Steps: {max_steps}")
    print(f"  Recording: {record} | Camera: {camera} | FPS: {fps}")
    print("=" * 50)

    for ep in range(episodes):
        obs = env.reset()
        done = False
        steps = 0
        info = {}
        ep_nfz_blocked = False
        
        frames = []
        
        while not done and steps < max_steps:
            action = ctrl.act(obs)
            obs, rew, done, info = env.step(action)
            steps += 1
            
            if info.get("nfz_blocked"):
                ep_nfz_blocked = True
                
            if record:
                frame = env.render(camera=camera)
                if use_nfz:
                    from scripts.eval_nfz import add_hud
                    frame = add_hud(frame, steps, info, env_label, ep+1, 
                                   in_nfz=info.get("in_nfz", False))
                else:
                    frame = add_hud_simple(frame, steps, info, env_label, ep+1)
                frames.append(frame)

        # Collect results
        if info.get("success"): results["success"] += 1
        elif info.get("collision"): results["collision"] += 1
        else: results["timeout"] += 1
        
        if ep_nfz_blocked: results["nfz_blocked"] += 1
        
        results["steps"].append(steps)
        results["dist"].append(info.get("dist_to_goal", 0))

        status = "SUCCESS" if info.get("success") else "COLLISION" if info.get("collision") else "TIMEOUT"
        
        if record and frames:
            v_name = f"{prefix}_ep{ep+1}_{timestamp}.mp4"
            v_path = os.path.join(video_dir, v_name)
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(v_path, fourcc, fps, (w, h))
            for f in frames: out.write(f)
            out.release()
            print(f"  Ep {ep+1}: {status:9} | {steps:3} steps | dist={info.get('dist_to_goal',0):.2f}m | video: {v_name}")
        else:
            print(f"  Ep {ep+1}: {status:9} | {steps:3} steps | dist={info.get('dist_to_goal',0):.2f}m")

    # Final summary
    print(f"\nSummary Results ({env_label})")
    print("-------------------------")
    print(f"Success Rate:      {results['success']}/{episodes} ({results['success']/episodes*100:.1f}%)")
    print(f"Collision Rate:    {results['collision']}/{episodes} ({results['collision']/episodes*100:.1f}%)")
    print(f"Timeout Rate:      {results['timeout']}/{episodes} ({results['timeout']/episodes*100:.1f}%)")
    if use_nfz:
        print(f"NFZ Blocked Eps:   {results['nfz_blocked']}/{episodes}")
    succ_steps = [s for i,s in enumerate(results['steps']) if results['dist'][i] < 0.6]
    if succ_steps:
        print(f"Avg Steps (Succ):  {np.mean(succ_steps):.1f}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--camera", type=str, default="pov")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--no-nfz", action="store_true", help="Use standard env without NFZ")
    args = parser.parse_args()

    run_gnn_llm_eval(
        episodes=args.episodes,
        max_steps=args.max_steps,
        camera=args.camera,
        record=not args.no_record,
        use_nfz=not args.no_nfz
    )
