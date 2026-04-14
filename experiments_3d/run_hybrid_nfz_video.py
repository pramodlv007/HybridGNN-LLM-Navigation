"""
Run Hybrid GNN-LLM in NFZ environment — 10 episodes with full video recording.
Saves videos to: videos/final_4conditions_10eps/
HUD shows: step, dist-to-goal, outcome, LLM bearing direction, GNN risk
"""
import os, sys, random, math
import numpy as np
import cv2
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.drone_nav_nfz import DroneNavNFZEnv
from controllers.hybrid_gnn_llm_controller import HybridGNNLLMController

EPISODES  = 10
MAX_STEPS = 1000
CAMERA    = "follow"   # follow camera — walls removed so view is clear
FPS       = 20

base_path = os.path.dirname(os.path.abspath(__file__))
video_dir = os.path.join(base_path, "..", "videos", "final_4conditions_10eps")
os.makedirs(video_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

ctrl = HybridGNNLLMController()

successes = collisions = timeouts = nfz_violations = 0
all_steps, succ_steps, goal_dists = [], [], []

# ── bearing direction label ──────────────────────────────────────────────────
def bearing_label(bx, by):
    angle = math.degrees(math.atan2(by, bx))
    dirs = [
        (157.5,  "← West"),   (112.5,  "↖ NW"),
        ( 67.5,  "↑ North"),  ( 22.5,  "↗ NE"),
        (-22.5,  "→ East"),   (-67.5,  "↘ SE"),
        (-112.5, "↓ South"),  (-157.5, "↙ SW"),
    ]
    for thresh, label in sorted(dirs, key=lambda x: x[0], reverse=True):
        if angle >= thresh:
            return label
    return "← West"

# ── HUD overlay ──────────────────────────────────────────────────────────────
def draw_hud(frame, step, info, ep, bearing, stagnating, near_nfz):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    dist = info.get("dist_to_goal", 0)
    if info.get("success"):
        status, sc = "SUCCESS", (0, 255, 80)
    elif info.get("collision"):
        status, sc = "COLLISION", (0, 60, 255)
    else:
        status, sc = "navigating", (220, 220, 220)

    # Row 1 — basic info
    cv2.putText(frame,
        f"Hybrid GNN-LLM | NFZ Env | Ep {ep}/{EPISODES} | Step {step:4d}",
        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

    # Row 2 — dist + status
    cv2.putText(frame,
        f"Dist-to-Goal: {dist:.2f}m    {status}",
        (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sc, 2)

    # Row 3 — LLM bearing + flags
    bx, by = bearing[0], bearing[1]
    blabel = bearing_label(bx, by)
    flag_nfz  = "  [NEAR NFZ]" if near_nfz else ""
    flag_stag = "  [ESCAPE]"   if stagnating else ""
    cv2.putText(frame,
        f"LLM Bearing: {blabel}  ({bx:+.2f}, {by:+.2f}){flag_nfz}{flag_stag}",
        (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
        (0, 200, 255) if not near_nfz else (0, 120, 255), 1)

    # NFZ warning banner
    if info.get("in_nfz") or info.get("nfz_blocked"):
        cv2.rectangle(frame, (0, h-30), (w, h), (0, 0, 180), -1)
        cv2.putText(frame, "!! NFZ VIOLATION !!",
            (w//2 - 95, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    return frame

# ── Main loop ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  HYBRID GNN-LLM — NFZ Environment  (10 episodes + video)")
print(f"  Camera: follow | Walls: invisible | HUD: LLM bearing")
print(f"  Saving to: videos/final_4conditions_10eps/")
print("="*60)

for ep in tqdm(range(EPISODES), desc="Hybrid NFZ"):
    seed = ep * 1000 + 500
    random.seed(seed);  np.random.seed(seed)
    env = DroneNavNFZEnv(gui=False)
    obs = env.reset()

    vp = os.path.join(video_dir, f"3d_hybrid_nfz_ep{ep+1}_{timestamp}.mp4")
    vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 480))

    done = False;  steps = 0;  info = {};  ep_nfz = False

    while not done and steps < MAX_STEPS:
        action = ctrl.act(obs)
        obs, _, done, info = env.step(action)
        steps += 1
        if info.get("nfz_violation") or info.get("nfz_blocked"):
            ep_nfz = True

        # Read internal controller state for HUD
        bearing    = ctrl.navigator.bearing if hasattr(ctrl, 'navigator') else np.array([1,0,0])
        stagnating = ctrl.stagnation_mode   if hasattr(ctrl, 'stagnation_mode') else False
        near_nfz   = getattr(ctrl, '_near_nfz', False)

        frame = env.render(width=640, height=480, camera=CAMERA)
        frame = draw_hud(frame, steps, info, ep+1, bearing, stagnating, near_nfz)
        vw.write(frame)

    # Hold final frame 1 s
    frame = env.render(width=640, height=480, camera=CAMERA)
    frame = draw_hud(frame, steps, info, ep+1, bearing, stagnating, near_nfz)
    for _ in range(FPS): vw.write(frame)
    vw.release()
    env.close()

    all_steps.append(steps)
    goal_dists.append(info.get("dist_to_goal", 0))
    if ep_nfz: nfz_violations += 1

    if info.get("success"):
        successes += 1;  succ_steps.append(steps);  outcome = "SUCCESS  "
    elif info.get("collision"):
        collisions += 1;  outcome = "COLLISION"
    else:
        timeouts += 1;  outcome = "TIMEOUT  "

    tqdm.write(f"  Ep {ep+1:2d}: {outcome} | {steps:4d} steps | "
               f"dist={info.get('dist_to_goal',0):.2f}m"
               + (" | NFZ!" if ep_nfz else ""))

print("\n" + "="*60)
print(f"  Success Rate : {100*successes/EPISODES:.1f}%  ({successes}/{EPISODES})")
print(f"  Collision    : {100*collisions/EPISODES:.1f}%")
print(f"  Timeout      : {100*timeouts/EPISODES:.1f}%")
print(f"  NFZ Viol.    : {100*nfz_violations/EPISODES:.1f}%")
print(f"  Videos saved : {video_dir}")
print("="*60)
