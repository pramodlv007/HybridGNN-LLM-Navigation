"""Quick debug: verify drone movement after TTC fix."""
import sys, os
sys.path.insert(0, '.')
from envs.drone_nav_3d import DroneNav3DEnv
from controllers.hybrid_controller import HybridController
import numpy as np

env = DroneNav3DEnv(gui=False)
ctrl = HybridController()
obs = env.reset()

print(f"Start: {obs['pos']}  Goal: {obs['goal']}")
print(f"Initial dist: {np.linalg.norm(obs['pos'] - obs['goal']):.2f}m")
print()

for step in range(300):
    action = ctrl.act(obs)
    obs, rew, done, info = env.step(action)
    if step % 30 == 0:
        pos = obs["pos"]
        speed = np.linalg.norm(obs["vel"])
        dist = info["dist_to_goal"]
        displacement = np.linalg.norm(pos - np.array([-3.6,-3.6,1.0]))
        print(f"Step {step:4d}: pos=[{pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:.2f}]  "
              f"speed={speed:.2f} dist_goal={dist:.2f} displaced={displacement:.2f}")
    if done:
        outcome = "SUCCESS" if info.get("success") else "COLLISION"
        print(f"\n>>> {outcome} at step {step}! pos={obs['pos']} dist={info['dist_to_goal']:.3f}")
        break

if not done:
    disp = np.linalg.norm(obs['pos'] - np.array([-3.6,-3.6,1.0]))
    print(f"\nAfter 300 steps: displaced={disp:.3f}m, dist_goal={info['dist_to_goal']:.3f}m")

env.close()
