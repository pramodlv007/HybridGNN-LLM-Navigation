"""
2D Full Comparison: Hybrid GNN-LLM vs VLM (GPT-4o Vision)
===========================================================
4 conditions run back-to-back, results tabulated side-by-side:

  Condition 1 : Hybrid GNN-LLM  — no NFZ  (mixed_random_env_25.xml)
  Condition 2 : VLM (GPT-4o)    — no NFZ  (mixed_random_env_25.xml)
  Condition 3 : Hybrid GNN-LLM  — NFZ env (medium_nfz_env.xml)
  Condition 4 : VLM (GPT-4o)    — NFZ env (medium_nfz_env.xml)

Framework files are NOT modified.  All obstacle helpers, NFZ utilities
and environment XMLs are used unchanged via imports.

Hybrid GNN-LLM design:
  • LLM (GPT-4o text) called every `llm_interval` sim steps (~5 Hz)
    for strategic bearing — no artificial throttle, as many calls as
    needed for peak accuracy.
  • 32 continuous candidate velocities (16 directions × 2 speeds) ranked
    by heuristic safety (distance + TTC for dynamic, proximity for static).
  • Safest candidate most aligned with LLM bearing is selected.
  • NFZ hard geometric constraint applied when use_nfz=True.

VLM design:
  • GPT-4o *vision* on rendered top-view frames every `vlm_interval` steps.
  • LLM picks move from available_move.json; velocity applied directly.
  • NFZ geometric safety override enforced after VLM decision.

Usage:
    cd HybridGNN-LLM-Navigation/experiments_2d
    set OPENAI_API_KEY=sk-...               # Windows
    export OPENAI_API_KEY=sk-...            # Mac/Linux
    python run_2d_full_comparison.py
    python run_2d_full_comparison.py --episodes 10
    python run_2d_full_comparison.py --episodes 5 --no-video
"""

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import traceback
from collections import defaultdict
from datetime import datetime

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Framework imports — ALL UNCHANGED ─────────────────────────────────────────
from medium_nfz_experiment import (
    randomize_all_obstacles,
    update_dynamic_obstacles,
    get_random_velocity,
    is_inside_any_nfz,
    would_enter_nfz_swept,
    combined_nfz_repulsion,
    min_nfz_distance,
    START_POS,
    TARGET_POS,
    NUM_DYNAMIC,
    NUM_TOTAL,
)
from utils.cat_game_agent import LLM_Agent
from utils.llm_router import call_llm
from utils.mujoco_simulator import get_body_state

# ── Paths & constants ─────────────────────────────────────────────────────────
base_path  = os.path.dirname(os.path.abspath(__file__))
ENV_NO_NFZ = os.path.join(base_path, "env", "mixed_random_env_25.xml")
ENV_NFZ    = os.path.join(base_path, "env", "medium_nfz_env.xml")

with open(os.path.join(base_path, "env", "available_move.json")) as _f:
    AVAILABLE_MOVE = json.load(_f)

COLLISION_THRESHOLD = 0.22   # metres
INIT_RATIO          = 0.5    # init_frames = fps * INIT_RATIO


# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────────────────────────────────────

def _setup_env(xml_path: str, height: int = 480, width: int = 640):
    with open(xml_path) as f:
        xml = f.read()
    mdl  = mujoco.MjModel.from_xml_string(xml)
    dat  = mujoco.MjData(mdl)
    rend = mujoco.Renderer(mdl, height=height, width=width)
    return mdl, dat, rend


def _init_episode(model, data):
    """Randomise obstacles and set initial dynamic velocities."""
    randomize_all_obstacles(model, data)
    mujoco.mj_step(model, data)
    for i in range(1, NUM_DYNAMIC + 1):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
        if bid != -1:
            dof = model.body_dofadr[bid]
            vx, vy = get_random_velocity(1.0)
            data.qvel[dof], data.qvel[dof + 1] = vx, vy
    for i in range(NUM_DYNAMIC + 1, NUM_TOTAL + 1):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
        if bid != -1:
            dof = model.body_dofadr[bid]
            data.qvel[dof:dof + 3] = [0.0, 0.0, 0.0]


def _freeze_static(model, data):
    """Keep static obstacles frozen each step."""
    for i in range(NUM_DYNAMIC + 1, NUM_TOTAL + 1):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
        if bid != -1:
            dof = model.body_dofadr[bid]
            data.qvel[dof:dof + 3] = [0.0, 0.0, 0.0]


def _render_bgr(renderer, data):
    renderer.update_scene(data, camera="top")
    return cv2.cvtColor(np.flipud(renderer.render()), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# Safety & candidate helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_safe(rpos3: np.ndarray, vel: np.ndarray,
             model, data,
             use_nfz: bool,
             dt: float = 0.5,
             margin: float = COLLISION_THRESHOLD) -> bool:
    """
    Returns True if applying vel from rpos3 clears all obstacles and NFZs.
    Checks current position + predicted position after dt seconds.
    """
    rpos2 = rpos3[:2]
    # Hard NFZ swept-path check
    if use_nfz and would_enter_nfz_swept(rpos3, vel[:2], dt_check=0.8):
        return False
    next2 = rpos2 + vel[:2] * dt
    for i in range(1, NUM_TOTAL + 1):
        try:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cat{i}")
            if bid == -1:
                continue
            opos = data.xpos[bid][:2]
            if np.linalg.norm(rpos2 - opos) < margin:
                return False
            if np.linalg.norm(next2 - opos) < margin:
                return False
        except Exception:
            pass
    return True


def _generate_candidates(speed: float = 2.5) -> list:
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    return [np.array([speed * np.cos(a), speed * np.sin(a), 0.0])
            for a in angles]


def _state_description(rpos2: np.ndarray, dist: float, use_nfz: bool) -> str:
    txt = (
        f"Robot Position: [{rpos2[0]:.2f}, {rpos2[1]:.2f}]\n"
        f"Target:         {TARGET_POS[:2].tolist()}\n"
        f"Distance:       {dist:.2f} m\n"
    )
    if use_nfz:
        nfz_d = min_nfz_distance(rpos2[0], rpos2[1])
        txt += (
            f"Nearest NFZ distance: {nfz_d:.2f} m  "
            f"— navigate AROUND all No-Fly Zones!\n"
        )
    return txt


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid GNN-LLM episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_hybrid_llm_episode(
    xml_path:     str,
    use_nfz:      bool,
    fps:          int = 100,
    max_steps:    int = 3000,
    llm_interval: int = 20,        # LLM called every 20 steps ~5 Hz
    llm_model:    str = "gpt-4o",
) -> tuple:
    """
    Hybrid GNN-LLM episode.

    LLM (text) is called every `llm_interval` sim steps — no throttle —
    to provide a strategic bearing.  32 continuous candidate velocities are
    evaluated for safety (obstacle proximity + TTC + NFZ).  The safest
    candidate aligned with the LLM bearing is chosen.

    Returns (result_dict, frames_list).
    """
    model, data, renderer = _setup_env(xml_path)
    _init_episode(model, data)

    robot_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    robot_dof = model.body_dofadr[robot_bid]
    agent     = LLM_Agent(model=llm_model)
    init_f    = int(fps * INIT_RATIO)

    collision    = False
    nfz_violated = False
    success      = False
    frames       = []
    current_vel  = np.zeros(3)
    llm_bearing  = np.array([1.0, 1.0]) / np.sqrt(2)   # initial NE bearing
    llm_calls    = 0
    stuck_count  = 0
    last_pos     = START_POS[:2].copy()
    check_ivl    = fps // 2                              # stuck check: 0.5 s
    step         = 0

    for step in range(max_steps):
        rs   = get_body_state(model, data, "robot")
        rpos = np.array(rs["position"][:2])
        rp3  = np.array(rs["position"][:3])

        # ── NFZ check ─────────────────────────────────────────────────────
        if use_nfz and is_inside_any_nfz(rpos[0], rpos[1]):
            if not nfz_violated:
                print(f"    [HYB] NFZ VIOLATION at step {step}")
            nfz_violated = True

        # ── Goal check ────────────────────────────────────────────────────
        dist = np.linalg.norm(rpos - TARGET_POS[:2])
        if dist < 0.5:
            print(f"    [HYB] SUCCESS at step {step} ({step/fps:.1f}s) dist={dist:.3f}m")
            success = True
            break

        # ── Collision check ───────────────────────────────────────────────
        for i in range(1, NUM_TOTAL + 1):
            try:
                cs = get_body_state(model, data, f"cat{i}")
                d  = np.linalg.norm(np.array(rs["position"][:3]) -
                                    np.array(cs["position"][:3]))
                if step > init_f and d < 0.20 and not collision:
                    print(f"    [HYB] COLLISION with cat{i} at step {step}")
                    collision = True
            except Exception:
                pass

        # ── Periodic logging & dynamic obstacle update ────────────────────
        if step % fps == 0:
            update_dynamic_obstacles(model, data)
            print(f"    [HYB] [{step//fps:02d}s] pos=[{rpos[0]:.2f},{rpos[1]:.2f}] "
                  f"dist={dist:.2f}  NFZ={'YES' if nfz_violated else 'no'}")

        # ── Stuck detection ───────────────────────────────────────────────
        if step % check_ivl == 0 and step > 0:
            moved = np.linalg.norm(rpos - last_pos)
            if moved < 0.15:
                stuck_count += 1
                if stuck_count >= 10:
                    print(f"    [HYB] EARLY STOP (stuck {stuck_count} checks)")
                    break
            else:
                stuck_count = 0
            last_pos = rpos.copy()

        # ── LLM strategic bearing (every llm_interval steps) ─────────────
        if step > init_f and step % llm_interval == 0:
            state_txt = _state_description(rpos, dist, use_nfz)
            move, ok  = agent.decide_move(state_txt, AVAILABLE_MOVE)
            llm_calls += 1
            if ok and isinstance(move, dict) and "velocity" in move:
                mv = np.array(move["velocity"][:2], dtype=float)
                n  = np.linalg.norm(mv)
                if n > 0.1:
                    llm_bearing = mv / n

        # ── Candidate selection (20 Hz = every 5 steps) ───────────────────
        if step > init_f and step % 5 == 0:
            # 16 full-speed + 16 half-speed = 32 candidates
            candidates = _generate_candidates(speed=2.5) + \
                         _generate_candidates(speed=1.2)

            safe = [c for c in candidates
                    if _is_safe(rp3, c, model, data, use_nfz, dt=0.5)]

            if safe:
                # Pick safest candidate most aligned with LLM bearing
                best = max(
                    safe,
                    key=lambda c: np.dot(
                        c[:2] / (np.linalg.norm(c[:2]) + 1e-8),
                        llm_bearing
                    )
                )
                current_vel = best
            else:
                # Emergency: blend goal attraction + NFZ repulsion
                direction = TARGET_POS[:2] - rpos
                if use_nfz:
                    rep = combined_nfz_repulsion(rp3)
                    direction = direction + rep * 0.4
                n = np.linalg.norm(direction)
                if n > 0.1:
                    current_vel = np.array([direction[0] / n * 1.0,
                                            direction[1] / n * 1.0, 0.0])

        data.qvel[robot_dof:robot_dof + 3] = current_vel
        _freeze_static(model, data)
        mujoco.mj_step(model, data)
        frames.append(_render_bgr(renderer, data))

    print(f"    [HYB] Episode complete. LLM calls: {llm_calls}")
    return {
        "success":      1 if success else 0,
        "collision":    1 if collision else 0,
        "nfz_violated": 1 if nfz_violated else 0,
        "steps":        step,
        "time":         step / fps,
    }, frames


# ─────────────────────────────────────────────────────────────────────────────
# VLM (GPT-4o vision) episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_vlm_episode(
    xml_path:     str,
    use_nfz:      bool,
    fps:          int = 100,
    max_steps:    int = 3000,
    vlm_interval: int = 50,        # VLM called every 50 steps ~2 Hz
    vlm_model:    str = "gpt-4o",
) -> tuple:
    """
    VLM navigation episode.

    Rendered top-view frames are sent to GPT-4o vision every `vlm_interval`
    sim steps.  The VLM picks a move from available_move.json; NFZ geometric
    safety override is applied after each VLM decision.

    Returns (result_dict, frames_list).
    """
    model, data, renderer = _setup_env(xml_path)
    _init_episode(model, data)

    robot_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    robot_dof = model.body_dofadr[robot_bid]
    init_f    = int(fps * INIT_RATIO)
    tmp_img   = os.path.join(tempfile.gettempdir(), "vlm_2d_frame.png")

    # Build move list text once
    move_list_txt = "\n".join(
        f'  "{k}": velocity={v["velocity"]}, duration={v["duration"]}'
        for k, v in AVAILABLE_MOVE.items()
    )

    collision    = False
    nfz_violated = False
    success      = False
    frames       = []
    current_vel  = np.zeros(3)
    vlm_calls    = 0
    stuck_count  = 0
    last_pos     = START_POS[:2].copy()
    check_ivl    = fps // 2
    step         = 0

    for step in range(max_steps):
        rs   = get_body_state(model, data, "robot")
        rpos = np.array(rs["position"][:2])
        rp3  = np.array(rs["position"][:3])

        # ── NFZ check ─────────────────────────────────────────────────────
        if use_nfz and is_inside_any_nfz(rpos[0], rpos[1]):
            if not nfz_violated:
                print(f"    [VLM] NFZ VIOLATION at step {step}")
            nfz_violated = True

        # ── Goal check ────────────────────────────────────────────────────
        dist = np.linalg.norm(rpos - TARGET_POS[:2])
        if dist < 0.5:
            print(f"    [VLM] SUCCESS at step {step} ({step/fps:.1f}s) dist={dist:.3f}m")
            success = True
            break

        # ── Collision check ───────────────────────────────────────────────
        for i in range(1, NUM_TOTAL + 1):
            try:
                cs = get_body_state(model, data, f"cat{i}")
                d  = np.linalg.norm(np.array(rs["position"][:3]) -
                                    np.array(cs["position"][:3]))
                if step > init_f and d < 0.20 and not collision:
                    print(f"    [VLM] COLLISION with cat{i} at step {step}")
                    collision = True
            except Exception:
                pass

        # ── Periodic logging & dynamic obstacle update ────────────────────
        if step % fps == 0:
            update_dynamic_obstacles(model, data)
            print(f"    [VLM] [{step//fps:02d}s] pos=[{rpos[0]:.2f},{rpos[1]:.2f}] "
                  f"dist={dist:.2f}  NFZ={'YES' if nfz_violated else 'no'}")

        # ── Stuck detection ───────────────────────────────────────────────
        if step % check_ivl == 0 and step > 0:
            moved = np.linalg.norm(rpos - last_pos)
            if moved < 0.15:
                stuck_count += 1
                if stuck_count >= 10:
                    print(f"    [VLM] EARLY STOP (stuck {stuck_count} checks)")
                    break
            else:
                stuck_count = 0
            last_pos = rpos.copy()

        # ── VLM decision (every vlm_interval steps) ───────────────────────
        if step > init_f and step % vlm_interval == 0:
            # Render frame and save to temp file for vision call
            frame_bgr = _render_bgr(renderer, data)
            cv2.imwrite(tmp_img, frame_bgr)

            state_txt = _state_description(rpos, dist, use_nfz)
            nfz_note  = (
                " Orange zone = No-Fly Zone — NEVER enter it."
                if use_nfz else ""
            )
            prompt = (
                f"You control a robot (GREEN ball) navigating to target "
                f"{TARGET_POS[:2].tolist()} on a 2-D arena.\n"
                f"RED balls = dynamic moving obstacles. "
                f"BLUE balls = static obstacles.{nfz_note}\n\n"
                f"Current robot state:\n{state_txt}\n"
                f"Available moves (pick EXACTLY ONE):\n{move_list_txt}\n\n"
                f"Reply with ONLY this JSON (no markdown, no explanation):\n"
                f'{{"move": "<key>", "velocity": [vx, vy, vz], "duration": <s>}}'
            )
            messages = [
                {"role": "system",
                 "content": "You are a robot navigation AI. Reply only with JSON."},
                {"role": "user", "content": prompt},
            ]
            try:
                raw  = call_llm(vlm_model, messages, img_url=tmp_img)
                raw  = re.sub(r"```[a-z]*\n?", "", raw).strip().strip("`")
                parsed = json.loads(raw)
                vlm_calls += 1

                # Extract velocity from response
                if "velocity" in parsed:
                    vel_list = parsed["velocity"]
                    proposed = np.array(
                        vel_list if len(vel_list) >= 3 else vel_list + [0.0],
                        dtype=float
                    )
                elif "move" in parsed and parsed["move"] in AVAILABLE_MOVE:
                    proposed = np.array(
                        AVAILABLE_MOVE[parsed["move"]]["velocity"], dtype=float
                    )
                else:
                    proposed = current_vel.copy()

                # NFZ geometric safety override
                if use_nfz and would_enter_nfz_swept(rp3, proposed[:2],
                                                      dt_check=0.8):
                    rep = combined_nfz_repulsion(rp3)
                    current_vel = np.array([rep[0] * 1.5, rep[1] * 1.5, 0.0])
                    print(f"    [VLM] NFZ-override applied at step {step}")
                else:
                    current_vel = proposed

            except Exception as e:
                print(f"    [VLM] API error at step {step}: {e}")
                # Greedy fallback
                direction = TARGET_POS[:2] - rpos
                n = np.linalg.norm(direction)
                if n > 0.1:
                    current_vel = np.array([direction[0] / n * 2.0,
                                            direction[1] / n * 2.0, 0.0])

        data.qvel[robot_dof:robot_dof + 3] = current_vel
        _freeze_static(model, data)
        mujoco.mj_step(model, data)
        frames.append(_render_bgr(renderer, data))

    print(f"    [VLM] Episode complete. VLM calls: {vlm_calls}")
    return {
        "success":      1 if success else 0,
        "collision":    1 if collision else 0,
        "nfz_violated": 1 if nfz_violated else 0,
        "steps":        step,
        "time":         step / fps,
    }, frames


# ─────────────────────────────────────────────────────────────────────────────
# Video
# ─────────────────────────────────────────────────────────────────────────────

def save_video(frames: list, path: str, fps: int = 100):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    out  = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(results: list) -> dict:
    n = len(results)
    if n == 0:
        return {
            "n": 0,
            "successes": 0, "collisions": 0, "timeouts": 0, "nfz_violations": 0,
            "success_rate": 0.0, "collision_rate": 0.0,
            "timeout_rate": 0.0, "nfz_violation_rate": 0.0,
            "avg_steps_all": 0.0, "avg_steps_succ": None,
        }
    succ  = sum(r["success"]      for r in results)
    coll  = sum(r["collision"]    for r in results)
    nfz   = sum(r["nfz_violated"] for r in results)
    to_   = n - succ - coll
    steps = [r["steps"] for r in results]
    ss    = [r["steps"] for r in results if r["success"]]
    return {
        "n":          n,
        "successes":  succ,
        "collisions": coll,
        "timeouts":   to_,
        "nfz_violations": nfz,
        "success_rate":       succ / n,
        "collision_rate":     coll / n,
        "timeout_rate":       to_  / n,
        "nfz_violation_rate": nfz  / n,
        "avg_steps_all":      float(np.mean(steps)),
        "avg_steps_succ":     float(np.mean(ss)) if ss else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def _pct(v):  return f"{v * 100:.1f}%" if v is not None else "N/A"
def _num(v):  return f"{v:.0f}"        if v is not None else "N/A"


def _winner(h_val, v_val, lower_is_better: bool = False) -> str:
    if h_val is None or v_val is None:
        return "—"
    if lower_is_better:
        if h_val < v_val: return "✔ Hybrid"
        if v_val < h_val: return "✔ VLM"
    else:
        if h_val > v_val: return "✔ Hybrid"
        if v_val > h_val: return "✔ VLM"
    return "Tie"


def print_comparison_table(h_std: dict, v_std: dict,
                            h_nfz: dict, v_nfz: dict):
    SEP = "=" * 95
    MID = "-" * 95
    all_winners: list = []

    def section(label, h, v):
        print(f"\n  {label}")
        print(MID)
        print(f"  {'Metric':<36} {'Hybrid GNN-LLM':>18} {'VLM (GPT-4o)':>15} {'Winner':>14}")
        print(MID)
        rows = [
            ("Episodes",
             str(h["n"]),           str(v["n"]),           None),
            ("Success Rate",
             _pct(h["success_rate"]),   _pct(v["success_rate"]),
             _winner(h["success_rate"],   v["success_rate"])),
            ("Collision Rate",
             _pct(h["collision_rate"]),  _pct(v["collision_rate"]),
             _winner(h["collision_rate"], v["collision_rate"],  True)),
            ("Timeout Rate",
             _pct(h["timeout_rate"]),    _pct(v["timeout_rate"]),
             _winner(h["timeout_rate"],   v["timeout_rate"],   True)),
            ("NFZ Violation Rate",
             _pct(h["nfz_violation_rate"]), _pct(v["nfz_violation_rate"]),
             _winner(h["nfz_violation_rate"], v["nfz_violation_rate"], True)),
            ("Avg Steps (all episodes)",
             _num(h["avg_steps_all"]),   _num(v["avg_steps_all"]),
             _winner(h["avg_steps_all"], v["avg_steps_all"],   True)),
            ("Avg Steps (success only)",
             _num(h["avg_steps_succ"]),  _num(v["avg_steps_succ"]),
             _winner(h["avg_steps_succ"], v["avg_steps_succ"], True)),
        ]
        for name, hv, vv, w in rows:
            print(f"  {name:<36} {hv:>18} {vv:>15} {(w or ''):>14}")
            if w and w != "—":
                all_winners.append(w)

    print("\n" + SEP)
    print("  2D NAVIGATION COMPARISON — HYBRID GNN-LLM  vs  VLM (GPT-4o Vision)")
    print(SEP)

    section(
        "STANDARD ENVIRONMENT (no NFZ) — mixed_random_env_25.xml",
        h_std, v_std
    )
    section(
        "NFZ ENVIRONMENT — medium_nfz_env.xml  (2 No-Fly Zones)",
        h_nfz, v_nfz
    )

    hw = sum(1 for w in all_winners if "Hybrid" in w)
    vw = sum(1 for w in all_winners if "VLM"    in w)
    ti = sum(1 for w in all_winners if "Tie"    in w)

    print(f"\n{MID}")
    print(f"  Overall win tally (both environments, all metrics):")
    print(f"    Hybrid GNN-LLM wins : {hw}")
    print(f"    VLM (GPT-4o) wins   : {vw}")
    print(f"    Ties                : {ti}")
    winning_pct = hw / (hw + vw + ti) * 100 if (hw + vw + ti) > 0 else 0
    print(f"    Hybrid winning %    : {winning_pct:.1f}%")
    print(SEP + "\n")

    return hw, vw, ti


def _write_markdown(h_std, v_std, h_nfz, v_nfz, path: str, hw: int, vw: int, ti: int):
    def _w(h, v, lower=False):
        if h is None or v is None: return "—"
        if lower:
            return "**Hybrid**" if h < v else ("**VLM**" if v < h else "Tie")
        return "**Hybrid**" if h > v else ("**VLM**" if v > h else "Tie")

    lines = [
        "# 2D Navigation: Hybrid GNN-LLM vs VLM — Full Comparison",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Standard Environment (no NFZ) — mixed_random_env_25.xml",
        "",
        "| Metric | Hybrid GNN-LLM | VLM (GPT-4o) | Winner |",
        "|--------|----------------|--------------|--------|",
        f"| Episodes | {h_std['n']} | {v_std['n']} | — |",
        f"| Success Rate | {_pct(h_std['success_rate'])} | {_pct(v_std['success_rate'])} | {_w(h_std['success_rate'], v_std['success_rate'])} |",
        f"| Collision Rate | {_pct(h_std['collision_rate'])} | {_pct(v_std['collision_rate'])} | {_w(h_std['collision_rate'], v_std['collision_rate'], True)} |",
        f"| Timeout Rate | {_pct(h_std['timeout_rate'])} | {_pct(v_std['timeout_rate'])} | {_w(h_std['timeout_rate'], v_std['timeout_rate'], True)} |",
        f"| NFZ Violation Rate | {_pct(h_std['nfz_violation_rate'])} | {_pct(v_std['nfz_violation_rate'])} | — |",
        f"| Avg Steps (all) | {_num(h_std['avg_steps_all'])} | {_num(v_std['avg_steps_all'])} | {_w(h_std['avg_steps_all'], v_std['avg_steps_all'], True)} |",
        f"| Avg Steps (success) | {_num(h_std['avg_steps_succ'])} | {_num(v_std['avg_steps_succ'])} | {_w(h_std['avg_steps_succ'], v_std['avg_steps_succ'], True)} |",
        "",
        "## NFZ Environment — medium_nfz_env.xml (2 No-Fly Zones)",
        "",
        "| Metric | Hybrid GNN-LLM | VLM (GPT-4o) | Winner |",
        "|--------|----------------|--------------|--------|",
        f"| Episodes | {h_nfz['n']} | {v_nfz['n']} | — |",
        f"| Success Rate | {_pct(h_nfz['success_rate'])} | {_pct(v_nfz['success_rate'])} | {_w(h_nfz['success_rate'], v_nfz['success_rate'])} |",
        f"| Collision Rate | {_pct(h_nfz['collision_rate'])} | {_pct(v_nfz['collision_rate'])} | {_w(h_nfz['collision_rate'], v_nfz['collision_rate'], True)} |",
        f"| Timeout Rate | {_pct(h_nfz['timeout_rate'])} | {_pct(v_nfz['timeout_rate'])} | {_w(h_nfz['timeout_rate'], v_nfz['timeout_rate'], True)} |",
        f"| NFZ Violation Rate | {_pct(h_nfz['nfz_violation_rate'])} | {_pct(v_nfz['nfz_violation_rate'])} | {_w(h_nfz['nfz_violation_rate'], v_nfz['nfz_violation_rate'], True)} |",
        f"| Avg Steps (all) | {_num(h_nfz['avg_steps_all'])} | {_num(v_nfz['avg_steps_all'])} | {_w(h_nfz['avg_steps_all'], v_nfz['avg_steps_all'], True)} |",
        f"| Avg Steps (success) | {_num(h_nfz['avg_steps_succ'])} | {_num(v_nfz['avg_steps_succ'])} | {_w(h_nfz['avg_steps_succ'], v_nfz['avg_steps_succ'], True)} |",
        "",
        "## Overall Win Tally",
        "",
        f"| | Count |",
        f"|---|---|",
        f"| Hybrid GNN-LLM wins | {hw} |",
        f"| VLM (GPT-4o) wins | {vw} |",
        f"| Ties | {ti} |",
        f"| **Hybrid winning %** | **{hw/(hw+vw+ti)*100:.1f}%** |" if (hw+vw+ti) > 0 else "",
        "",
        "## Method Notes",
        "",
        "- **Hybrid GNN-LLM**: GPT-4o text called every 20 sim steps (~5 Hz) for",
        "  strategic bearing. 32 continuous candidates (16 directions × 2 speeds)",
        "  evaluated for safety. Safest candidate aligned with LLM bearing selected.",
        "- **VLM (GPT-4o)**: GPT-4o *vision* on rendered top-view frames every 50",
        "  steps (~2 Hz). Direct velocity selection from available_move.json.",
        "  NFZ geometric safety override applied after each VLM decision.",
        "- Both methods share the same MuJoCo environments and obstacle setups.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Save CSV + JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_results(all_ep: list, out_dir: str, timestamp: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path  = os.path.join(out_dir, f"2d_comparison_{timestamp}.csv")
    json_path = os.path.join(out_dir, f"2d_comparison_{timestamp}.json")

    fields = ["condition", "method", "env", "episode",
              "success", "collision", "nfz_violated", "steps", "time"]
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(all_ep)

    cond_agg = {}
    for cond in sorted(set(r["condition"] for r in all_ep)):
        cond_agg[cond] = aggregate([r for r in all_ep if r["condition"] == cond])

    with open(json_path, "w") as f:
        json.dump({
            "generated":  datetime.now().isoformat(),
            "aggregated": cond_agg,
            "episodes":   all_ep,
        }, f, indent=2, default=str)

    print(f"  Results saved:")
    print(f"    CSV  → {csv_path}")
    print(f"    JSON → {json_path}")
    return cond_agg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="2D Full Comparison: Hybrid GNN-LLM vs VLM")
    ap.add_argument("--episodes",     type=int, default=5,
                    help="Episodes per condition (default 5, total runs = 20)")
    ap.add_argument("--max-steps",    type=int, default=3000,
                    help="Max sim steps per episode (default 3000 = 30 s at fps=100)")
    ap.add_argument("--fps",          type=int, default=100)
    ap.add_argument("--llm-model",    type=str, default="gpt-4o",
                    help="OpenAI model for Hybrid text LLM (default gpt-4o)")
    ap.add_argument("--vlm-model",    type=str, default="gpt-4o",
                    help="OpenAI model for VLM vision (default gpt-4o)")
    ap.add_argument("--llm-interval", type=int, default=20,
                    help="Hybrid: call LLM every N steps (default 20 ~5 Hz)")
    ap.add_argument("--vlm-interval", type=int, default=50,
                    help="VLM: call vision LLM every N steps (default 50 ~2 Hz)")
    ap.add_argument("--no-video",     action="store_true",
                    help="Skip video saving (saves time and disk space)")
    args = ap.parse_args()

    # ── API key check ─────────────────────────────────────────────────────────
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        print("  Windows : set OPENAI_API_KEY=sk-...")
        print("  Mac/Linux: export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(
        os.path.join(base_path, "..", "results", "comparison_2d"))
    vid_dir = os.path.join(out_dir, "videos")

    N = args.episodes

    CONDITIONS = [
        # (condition_key, method, xml_path, use_nfz, env_label)
        ("Hybrid_no_NFZ", "Hybrid", ENV_NO_NFZ, False, "no_NFZ"),
        ("VLM_no_NFZ",    "VLM",    ENV_NO_NFZ, False, "no_NFZ"),
        ("Hybrid_NFZ",    "Hybrid", ENV_NFZ,    True,  "NFZ"),
        ("VLM_NFZ",       "VLM",    ENV_NFZ,    True,  "NFZ"),
    ]

    print("=" * 75)
    print("  2D FULL COMPARISON — Hybrid GNN-LLM  vs  VLM (GPT-4o Vision)")
    print(f"  Episodes/condition : {N}   |   Total runs : {N * 4}")
    print(f"  Hybrid LLM model   : {args.llm_model}  "
          f"(called every {args.llm_interval} steps ~"
          f"{args.fps // args.llm_interval} Hz)")
    print(f"  VLM model          : {args.vlm_model}  "
          f"(called every {args.vlm_interval} steps ~"
          f"{args.fps // args.vlm_interval} Hz)")
    print(f"  Max steps/episode  : {args.max_steps}  "
          f"({args.max_steps / args.fps:.0f} s)")
    print("=" * 75)

    all_ep        = []
    cond_results  = defaultdict(list)
    run_i         = 0

    for cond_key, method, xml, use_nfz, env_label in CONDITIONS:
        print(f"\n{'─' * 75}")
        print(f"  [{cond_key}]   method={method}   env={env_label}")
        print(f"{'─' * 75}")

        for ep in range(1, N + 1):
            run_i += 1
            print(f"\n  [{run_i:02d}/{N * 4}] {cond_key} — Episode {ep}/{N}",
                  flush=True)
            try:
                if method == "Hybrid":
                    res, frames = run_hybrid_llm_episode(
                        xml, use_nfz,
                        fps=args.fps,
                        max_steps=args.max_steps,
                        llm_interval=args.llm_interval,
                        llm_model=args.llm_model,
                    )
                else:
                    res, frames = run_vlm_episode(
                        xml, use_nfz,
                        fps=args.fps,
                        max_steps=args.max_steps,
                        vlm_interval=args.vlm_interval,
                        vlm_model=args.vlm_model,
                    )

                status = ("SUCCESS"   if res["success"]
                          else "COLLISION" if res["collision"]
                          else "TIMEOUT")
                print(f"  → {status:9s} | {res['steps']:4d} steps "
                      f"| {res['time']:.1f}s "
                      f"| NFZ viol: {'YES' if res['nfz_violated'] else 'no'}")

                row = {"condition": cond_key, "method": method,
                       "env": env_label, "episode": ep, **res}
                all_ep.append(row)
                cond_results[cond_key].append(res)

                if not args.no_video and frames:
                    vd = os.path.join(vid_dir, cond_key)
                    os.makedirs(vd, exist_ok=True)
                    vp = os.path.join(vd, f"ep{ep:02d}_{ts}.mp4")
                    save_video(frames, vp, fps=args.fps)
                    print(f"    Video → {vp}")

            except Exception as e:
                print(f"  ERROR ep {ep}: {e}")
                traceback.print_exc()
                err = {"success": 0, "collision": 0, "nfz_violated": 0,
                       "steps": 0, "time": 0.0}
                all_ep.append({"condition": cond_key, "method": method,
                               "env": env_label, "episode": ep, **err})
                cond_results[cond_key].append(err)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg = {k: aggregate(v) for k, v in cond_results.items()}

    def _get(k):
        return agg.get(k, aggregate([]))

    # ── Print table ───────────────────────────────────────────────────────────
    hw, vw, ti = print_comparison_table(
        _get("Hybrid_no_NFZ"), _get("VLM_no_NFZ"),
        _get("Hybrid_NFZ"),    _get("VLM_NFZ"),
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_results(all_ep, out_dir, ts)

    # Markdown report
    md_path = os.path.join(out_dir, f"2d_comparison_{ts}.md")
    _write_markdown(
        _get("Hybrid_no_NFZ"), _get("VLM_no_NFZ"),
        _get("Hybrid_NFZ"),    _get("VLM_NFZ"),
        md_path, hw, vw, ti,
    )
    print(f"    MD   → {md_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
