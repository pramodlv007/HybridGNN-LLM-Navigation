"""
3D Drone Experiment - Loop until >70% success rate
====================================================
Runs batches of episodes. After each batch, checks cumulative success rate.
Stops when success rate > 70% (across all episodes so far).
Saves ALL videos to '3d expt' folder.
"""

import os
import sys
import numpy as np
import cv2
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.drone_nav_3d import DroneNav3DEnv
from controllers.hybrid_gnn_llm_controller import HybridGNNLLMController


def add_hud(frame, step, info, ep):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    dist = info.get('dist_to_goal', 0)
    text = f"[GNN+LLM] Ep {ep} | Step {step} | Dist: {dist:.2f}m"
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


def run_single_episode(env, ctrl, ep_num, video_dir, timestamp, camera="chase", fps=20, max_steps=600):
    """Run one episode, save video, return result dict."""
    obs = env.reset()
    done = False
    steps = 0
    info = {}
    frames = []

    while not done and steps < max_steps:
        action = ctrl.act(obs)
        obs, reward, done, info = env.step(action)
        steps += 1

        if steps % 100 == 0:
            d = info.get('dist_to_goal', 0)
            pos = obs['pos']
            print(f"    Step {steps:3d} | Pos: [{pos[0]:.2f}, {pos[1]:.2f}] | Dist: {d:.2f}m")

        frame = env.render(camera=camera)
        frame = add_hud(frame, steps, info, ep_num)
        frames.append(frame)

    status = "SUCCESS" if info.get("success") else "COLLISION" if info.get("collision") else "TIMEOUT"
    dist = info.get('dist_to_goal', 0)

    # Save video
    v_name = f"3d_gnn_llm_ep{ep_num}_{status}_{timestamp}.mp4"
    v_path = os.path.join(video_dir, v_name)
    if frames:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(v_path, fourcc, fps, (w, h))
        for f in frames:
            out.write(f)
        out.release()

    return {
        "episode": ep_num,
        "status": status,
        "steps": steps,
        "dist": dist,
        "video": v_name
    }


def main():
    video_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "3d expt"))
    os.makedirs(video_dir, exist_ok=True)

    env = DroneNav3DEnv(gui=False)
    ctrl = HybridGNNLLMController()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    target_rate = 0.70
    batch_size = 5
    max_batches = 10  # Safety cap: max 50 episodes total
    max_steps_per_ep = 600  # Slightly more time to reach goal

    all_results = []
    ep_counter = 0

    print("=" * 60)
    print("3D EXPERIMENT LOOP - Target: >70% Success Rate")
    print(f"Batch size: {batch_size} | Max batches: {max_batches}")
    print(f"Max steps/episode: {max_steps_per_ep}")
    print(f"Videos -> {video_dir}")
    print("=" * 60)

    for batch in range(1, max_batches + 1):
        print(f"\n{'-'*60}")
        print(f"BATCH {batch}")
        print(f"{'-'*60}")

        for i in range(batch_size):
            ep_counter += 1
            print(f"\n  Episode {ep_counter}:")

            result = run_single_episode(
                env, ctrl, ep_counter, video_dir, timestamp,
                camera="chase", fps=20, max_steps=max_steps_per_ep
            )
            all_results.append(result)

            print(f"    -> {result['status']} | {result['steps']} steps | dist={result['dist']:.2f}m | {result['video']}")

        # Check cumulative success rate
        successes = sum(1 for r in all_results if r['status'] == 'SUCCESS')
        total = len(all_results)
        rate = successes / total

        print(f"")
        print(f"  >> Cumulative: {successes}/{total} = {rate*100:.1f}% success <<")
        print(f"")

        if rate > target_rate:
            print(f"  [OK] TARGET ACHIEVED! {rate*100:.1f}% > {target_rate*100:.0f}%")
            break
        else:
            print(f"  ... Below target ({rate*100:.1f}% <= {target_rate*100:.0f}%). Running next batch...")

    # Final summary
    successes = sum(1 for r in all_results if r['status'] == 'SUCCESS')
    collisions = sum(1 for r in all_results if r['status'] == 'COLLISION')
    timeouts = sum(1 for r in all_results if r['status'] == 'TIMEOUT')
    total = len(all_results)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({total} episodes)")
    print(f"{'='*60}")
    print(f"  Success:   {successes}/{total} ({successes/total*100:.1f}%)")
    print(f"  Collision: {collisions}/{total} ({collisions/total*100:.1f}%)")
    print(f"  Timeout:   {timeouts}/{total} ({timeouts/total*100:.1f}%)")

    succ_steps = [r['steps'] for r in all_results if r['status'] == 'SUCCESS']
    if succ_steps:
        print(f"  Avg steps (success): {np.mean(succ_steps):.1f}")

    print(f"\n  Episode Details:")
    for r in all_results:
        icon = "[OK]" if r['status'] == 'SUCCESS' else "[XX]" if r['status'] == 'COLLISION' else "[--]"
        print(f"    {icon} Ep {r['episode']:2d}: {r['status']:9s} | {r['steps']:3d} steps | {r['dist']:.2f}m")

    print(f"\n  Videos saved in: {video_dir}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
