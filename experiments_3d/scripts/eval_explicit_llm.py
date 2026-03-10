"""
Explicit LLM Evaluation Runner (3D)
=====================================
Runs the Explicit LLM Controller in EITHER:
  - Standard environment (20 obstacles, no NFZ)
  - NFZ environment (25 obstacles + L-shaped hard wall)
Saves an `action_log.txt` and overlays current action on the video HUD.
"""

import argparse
import numpy as np
import cv2
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

def add_hud(frame, step, fps, stats, env_label, ep):
    """Draw a dark banner HUD at the top of the video."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
    
    current_action = stats.get("action")
    act_str = current_action.get('description', 'Hovering / Waiting') if current_action else 'Starting'
    
    cv2.putText(frame, f"Action: {act_str}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    risk = stats.get("risk", 0.0)
    text = f"[{env_label}] Ep {ep} | Time: {step/fps:.1f}s | Risk: {risk:.1f}"
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame

def run_explicit_llm_eval(episodes=5, max_steps=500, camera="pov", fps=20, record=True, use_nfz=True):
    from hybrid3d_drone_nav.controllers.explicit_llm_controller import ExplicitLLMController

    if use_nfz:
        from hybrid3d_drone_nav.envs.drone_nav_nfz import DroneNavNFZEnv
        env = DroneNavNFZEnv(gui=False)
        env_label = "Explicit-LLM (NFZ)"
        prefix = "3d_explicit_llm_nfz"
    else:
        from hybrid3d_drone_nav.envs.drone_nav_3d import DroneNav3DEnv
        env = DroneNav3DEnv(gui=False)
        env_label = "Explicit-LLM"
        prefix = "3d_explicit_llm"

    ctrl = ExplicitLLMController(fps=fps)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = os.path.join(os.path.dirname(__file__), "../videos/LLM_Actions")
    os.makedirs(video_dir, exist_ok=True)

    results = {
        "success": 0,
        "collision": 0,
        "timeout": 0,
        "nfz_blocked": 0,
        "steps": [],
        "dist": []
    }

    print(f"\n  3D {env_label} EXPERIMENT (Explicit Actions + HUD)")
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
        
        # Open action log for this episode
        log_name = f"{prefix}_ep{ep+1}_{timestamp}_action_log.txt"
        log_path = os.path.join(video_dir, log_name)
        log_file = open(log_path, "w")
        ctrl.set_log_file(log_file)
        
        while not done and steps < max_steps:
            action = ctrl.act(obs)
            obs, rew, done, info = env.step(action)
            steps += 1
            
            if info.get("nfz_blocked"):
                ep_nfz_blocked = True
                
            if record:
                frame = env.render(camera=camera)
                stats = ctrl.get_hud_stats()
                frame = add_hud(frame, steps, fps, stats, env_label, ep+1)
                frames.append(frame)

        log_file.close()

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
            print(f"  Ep {ep+1}: Log saved to: {log_name}")
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
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--camera", type=str, default="chase")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--no-nfz", action="store_true", help="Use standard env without NFZ")
    args = parser.parse_args()

    # Set episodes to 1 by default for demonstration quickly
    run_explicit_llm_eval(
        episodes=args.episodes,
        max_steps=args.max_steps,
        camera=args.camera,
        record=not args.no_record,
        use_nfz=not args.no_nfz
    )
