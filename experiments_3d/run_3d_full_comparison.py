"""
3D Full Comparison Runner
==========================
Runs all 4 experimental conditions back-to-back and prints a
side-by-side performance comparison table:

    Condition 1 : Hybrid GNN-LLM  — standard env (no NFZ)
    Condition 2 : VLM             — standard env (no NFZ)
    Condition 3 : Hybrid GNN-LLM  — NFZ env
    Condition 4 : VLM             — NFZ env

All existing controllers and environments are used UNCHANGED.
Only new eval_vlm.py is new; eval_gnn_llm.py and eval_nfz.py are
called as library functions without modification.

Usage:
    cd experiments_3d
    python run_3d_full_comparison.py
    python run_3d_full_comparison.py --episodes 20 --camera top
    python run_3d_full_comparison.py --no-record          # skip videos
"""

import argparse
import os
import sys
import csv
import json
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Pretty table helpers
# ---------------------------------------------------------------------------

def _pct(rate):
    return f"{rate*100:.1f}%"


def _steps(val):
    return f"{val:.0f}" if val is not None else "N/A"


def _dist(val):
    return f"{val:.3f}m" if val is not None else "N/A"


def print_table(results_no_nfz, results_nfz):
    """Print a rich side-by-side comparison table."""
    sep = "=" * 90
    mid = "-" * 90

    print("\n" + sep)
    print("  3D NAVIGATION COMPARISON — HYBRID GNN-LLM  vs  VLM")
    print(sep)

    # ── Standard env (no NFZ) ──────────────────────────────────────────────
    h, v = results_no_nfz["hybrid"], results_no_nfz["vlm"]
    eps = h["episodes"]

    print(f"\n{'STANDARD ENVIRONMENT (no NFZ)':^90}")
    print(mid)
    header = f"  {'Metric':<30}{'Hybrid GNN-LLM':>20}{'VLM':>20}{'Hybrid Wins?':>18}"
    print(header)
    print(mid)

    rows_std = [
        ("Episodes",            str(eps),                        str(v["episodes"]),                    ""),
        ("Success Rate",        _pct(h["success_rate"]),         _pct(v["success_rate"]),               "[Y] Hybrid" if h["success_rate"] > v["success_rate"] else ("[Y] VLM" if v["success_rate"] > h["success_rate"] else "Tie")),
        ("Collision Rate",      _pct(h["collision_rate"]),       _pct(v["collision_rate"]),             "[Y] Hybrid" if h["collision_rate"] < v["collision_rate"] else ("[Y] VLM" if v["collision_rate"] < h["collision_rate"] else "Tie")),
        ("Timeout Rate",        _pct(h["timeout_rate"]),         _pct(v["timeout_rate"]),               "[Y] Hybrid" if h["timeout_rate"] < v["timeout_rate"] else ("[Y] VLM" if v["timeout_rate"] < h["timeout_rate"] else "Tie")),
        ("Avg Steps (all eps)", _steps(h["avg_steps_all"]),      _steps(v["avg_steps_all"]),            "[Y] Hybrid" if h["avg_steps_all"] < v["avg_steps_all"] else ("[Y] VLM" if v["avg_steps_all"] < h["avg_steps_all"] else "Tie")),
        ("Avg Steps (success)", _steps(h["avg_steps_succ"]),     _steps(v["avg_steps_succ"]),           "[Y] Hybrid" if (h["avg_steps_succ"] or 9999) < (v["avg_steps_succ"] or 9999) else ("[Y] VLM" if (v["avg_steps_succ"] or 9999) < (h["avg_steps_succ"] or 9999) else "Tie")),
        ("Avg Final Dist",      _dist(h["avg_final_dist"]),      _dist(v["avg_final_dist"]),            "[Y] Hybrid" if h["avg_final_dist"] < v["avg_final_dist"] else ("[Y] VLM" if v["avg_final_dist"] < h["avg_final_dist"] else "Tie")),
        ("Min Obs Dist",        _dist(h.get("min_obs_dist")),    _dist(v.get("min_obs_dist")),          ""),
    ]
    for name, hval, vval, winner in rows_std:
        print(f"  {name:<30}{hval:>20}{vval:>20}{winner:>18}")

    # ── NFZ env ───────────────────────────────────────────────────────────
    hn, vn = results_nfz["hybrid"], results_nfz["vlm"]
    eps_n = hn["episodes"]

    print(f"\n{'NFZ ENVIRONMENT (25 obstacles + hard wall)':^90}")
    print(mid)
    header = f"  {'Metric':<30}{'Hybrid GNN-LLM':>20}{'VLM':>20}{'Hybrid Wins?':>18}"
    print(header)
    print(mid)

    rows_nfz = [
        ("Episodes",            str(eps_n),                       str(vn["episodes"]),                    ""),
        ("Success Rate",        _pct(hn["success_rate"]),         _pct(vn["success_rate"]),               "[Y] Hybrid" if hn["success_rate"] > vn["success_rate"] else ("[Y] VLM" if vn["success_rate"] > hn["success_rate"] else "Tie")),
        ("Collision Rate",      _pct(hn["collision_rate"]),       _pct(vn["collision_rate"]),             "[Y] Hybrid" if hn["collision_rate"] < vn["collision_rate"] else ("[Y] VLM" if vn["collision_rate"] < hn["collision_rate"] else "Tie")),
        ("Timeout Rate",        _pct(hn["timeout_rate"]),         _pct(vn["timeout_rate"]),               "[Y] Hybrid" if hn["timeout_rate"] < vn["timeout_rate"] else ("[Y] VLM" if vn["timeout_rate"] < hn["timeout_rate"] else "Tie")),
        ("NFZ Violation Rate",  _pct(hn["nfz_violation_rate"]),  _pct(vn["nfz_violation_rate"]),         "[Y] Hybrid" if hn["nfz_violation_rate"] < vn["nfz_violation_rate"] else ("[Y] VLM" if vn["nfz_violation_rate"] < hn["nfz_violation_rate"] else "Tie")),
        ("Avg Steps (all eps)", _steps(hn["avg_steps_all"]),      _steps(vn["avg_steps_all"]),            "[Y] Hybrid" if hn["avg_steps_all"] < vn["avg_steps_all"] else ("[Y] VLM" if vn["avg_steps_all"] < hn["avg_steps_all"] else "Tie")),
        ("Avg Steps (success)", _steps(hn["avg_steps_succ"]),     _steps(vn["avg_steps_succ"]),           "[Y] Hybrid" if (hn["avg_steps_succ"] or 9999) < (vn["avg_steps_succ"] or 9999) else ("[Y] VLM" if (vn["avg_steps_succ"] or 9999) < (hn["avg_steps_succ"] or 9999) else "Tie")),
        ("Avg Final Dist",      _dist(hn["avg_final_dist"]),      _dist(vn["avg_final_dist"]),            "[Y] Hybrid" if hn["avg_final_dist"] < vn["avg_final_dist"] else ("[Y] VLM" if vn["avg_final_dist"] < hn["avg_final_dist"] else "Tie")),
        ("Min Obs Dist",        _dist(hn.get("min_obs_dist")),    _dist(vn.get("min_obs_dist")),          ""),
    ]
    for name, hval, vval, winner in rows_nfz:
        print(f"  {name:<30}{hval:>20}{vval:>20}{winner:>18}")

    # ── Overall win tally ─────────────────────────────────────────────────
    print("\n" + mid)
    all_winners = [r[3] for r in rows_std + rows_nfz if r[3]]
    hybrid_wins = sum(1 for w in all_winners if "Hybrid" in w)
    vlm_wins    = sum(1 for w in all_winners if "VLM" in w)
    ties        = sum(1 for w in all_winners if "Tie" in w)
    print(f"  Overall (across both envs):   "
          f"Hybrid wins {hybrid_wins}  |  VLM wins {vlm_wins}  |  Ties {ties}")
    print(sep + "\n")


def save_results(results_no_nfz, results_nfz, out_dir):
    """Save CSV and JSON summaries."""
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV
    csv_path = os.path.join(out_dir, f"3d_comparison_{ts}.csv")
    fieldnames = [
        "method", "env", "episodes",
        "success_rate", "collision_rate", "timeout_rate", "nfz_violation_rate",
        "avg_steps_all", "avg_steps_succ", "avg_final_dist", "min_obs_dist",
    ]
    rows = []
    for env_label, mapping in [("no_nfz", results_no_nfz), ("nfz", results_nfz)]:
        for ctrl_label, r in mapping.items():
            rows.append({
                "method": ctrl_label,
                "env": env_label,
                "episodes": r["episodes"],
                "success_rate": f"{r['success_rate']*100:.1f}",
                "collision_rate": f"{r['collision_rate']*100:.1f}",
                "timeout_rate": f"{r['timeout_rate']*100:.1f}",
                "nfz_violation_rate": f"{r['nfz_violation_rate']*100:.1f}" if r.get("nfz_violation_rate") else "0.0",
                "avg_steps_all": f"{r['avg_steps_all']:.1f}",
                "avg_steps_succ": f"{r['avg_steps_succ']:.1f}" if r.get("avg_steps_succ") else "N/A",
                "avg_final_dist": f"{r['avg_final_dist']:.3f}",
                "min_obs_dist": f"{r['min_obs_dist']:.3f}" if r.get("min_obs_dist") else "N/A",
            })
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # JSON (full)
    json_path = os.path.join(out_dir, f"3d_comparison_{ts}.json")
    payload = {
        "generated": datetime.now().isoformat(),
        "no_nfz": {k: {kk: vv for kk, vv in v.items() if kk != "video_files"}
                   for k, v in results_no_nfz.items()},
        "nfz":    {k: {kk: vv for kk, vv in v.items() if kk != "video_files"}
                   for k, v in results_nfz.items()},
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    # Markdown table
    md_path = os.path.join(out_dir, f"3d_comparison_{ts}.md")
    _write_markdown(results_no_nfz, results_nfz, md_path)

    print(f"  Results saved:")
    print(f"    CSV  -> {csv_path}")
    print(f"    JSON -> {json_path}")
    print(f"    MD   -> {md_path}")


def _write_markdown(results_no_nfz, results_nfz, path):
    h  = results_no_nfz["hybrid"]
    v  = results_no_nfz["vlm"]
    hn = results_nfz["hybrid"]
    vn = results_nfz["vlm"]

    lines = [
        "# 3D Navigation: Hybrid GNN-LLM vs VLM — Full Comparison",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Standard Environment (no NFZ)",
        "",
        f"| Metric | Hybrid GNN-LLM | VLM | Winner |",
        f"|--------|---------------|-----|--------|",
    ]

    def winner(a, b, lower_is_better=False):
        if a is None or b is None:
            return "—"
        better = a < b if lower_is_better else a > b
        worse  = a > b if lower_is_better else a < b
        return "**Hybrid**" if better else ("**VLM**" if worse else "Tie")

    lines += [
        f"| Episodes | {h['episodes']} | {v['episodes']} | — |",
        f"| Success Rate | {_pct(h['success_rate'])} | {_pct(v['success_rate'])} | {winner(h['success_rate'], v['success_rate'])} |",
        f"| Collision Rate | {_pct(h['collision_rate'])} | {_pct(v['collision_rate'])} | {winner(h['collision_rate'], v['collision_rate'], lower_is_better=True)} |",
        f"| Timeout Rate | {_pct(h['timeout_rate'])} | {_pct(v['timeout_rate'])} | {winner(h['timeout_rate'], v['timeout_rate'], lower_is_better=True)} |",
        f"| Avg Steps (all) | {_steps(h['avg_steps_all'])} | {_steps(v['avg_steps_all'])} | {winner(h['avg_steps_all'], v['avg_steps_all'], lower_is_better=True)} |",
        f"| Avg Steps (success) | {_steps(h['avg_steps_succ'])} | {_steps(v['avg_steps_succ'])} | {winner(h['avg_steps_succ'], v['avg_steps_succ'], lower_is_better=True)} |",
        f"| Avg Final Dist | {_dist(h['avg_final_dist'])} | {_dist(v['avg_final_dist'])} | {winner(h['avg_final_dist'], v['avg_final_dist'], lower_is_better=True)} |",
        f"| Min Obs Dist | {_dist(h.get('min_obs_dist'))} | {_dist(v.get('min_obs_dist'))} | — |",
        "",
        "## NFZ Environment (25 obstacles + hard wall)",
        "",
        f"| Metric | Hybrid GNN-LLM | VLM | Winner |",
        f"|--------|---------------|-----|--------|",
        f"| Episodes | {hn['episodes']} | {vn['episodes']} | — |",
        f"| Success Rate | {_pct(hn['success_rate'])} | {_pct(vn['success_rate'])} | {winner(hn['success_rate'], vn['success_rate'])} |",
        f"| Collision Rate | {_pct(hn['collision_rate'])} | {_pct(vn['collision_rate'])} | {winner(hn['collision_rate'], vn['collision_rate'], lower_is_better=True)} |",
        f"| Timeout Rate | {_pct(hn['timeout_rate'])} | {_pct(vn['timeout_rate'])} | {winner(hn['timeout_rate'], vn['timeout_rate'], lower_is_better=True)} |",
        f"| NFZ Violation Rate | {_pct(hn['nfz_violation_rate'])} | {_pct(vn['nfz_violation_rate'])} | {winner(hn['nfz_violation_rate'], vn['nfz_violation_rate'], lower_is_better=True)} |",
        f"| Avg Steps (all) | {_steps(hn['avg_steps_all'])} | {_steps(vn['avg_steps_all'])} | {winner(hn['avg_steps_all'], vn['avg_steps_all'], lower_is_better=True)} |",
        f"| Avg Steps (success) | {_steps(hn['avg_steps_succ'])} | {_steps(vn['avg_steps_succ'])} | {winner(hn['avg_steps_succ'], vn['avg_steps_succ'], lower_is_better=True)} |",
        f"| Avg Final Dist | {_dist(hn['avg_final_dist'])} | {_dist(vn['avg_final_dist'])} | {winner(hn['avg_final_dist'], vn['avg_final_dist'], lower_is_better=True)} |",
        f"| Min Obs Dist | {_dist(hn.get('min_obs_dist'))} | {_dist(vn.get('min_obs_dist'))} | — |",
        "",
        "## Key Observations",
        "",
        "- **Hybrid GNN-LLM** uses text-based LLM bearing (GPT-4o text) + GNN safety filter.",
        "- **VLM** uses GPT-4o *vision* on rendered camera frames + same GNN safety filter.",
        "- Both methods share the identical GNN safety checker and post-decision safety shield.",
        "- The only difference is the perception modality feeding the bearing decision.",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run all 4 3D conditions and compare Hybrid vs VLM")
    parser.add_argument("--episodes", type=int, default=15,
                        help="Episodes per condition (default 15)")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--camera", type=str, default="follow",
                        choices=["pov", "front", "top", "follow"])
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()

    record = not args.no_record
    eps    = args.episodes

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "results", "final_4conditions_10eps"
    )

    print("\n" + "=" * 60)
    print("  3D FULL COMPARISON — HYBRID GNN-LLM vs VLM")
    print(f"  {eps} episodes per condition  |  4 conditions total")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Import only what returns a usable dict (eval_vlm)
    # Hybrid conditions use inline capture functions to collect metrics.
    # ------------------------------------------------------------------
    from scripts.eval_vlm import run_vlm_eval

    results_no_nfz = {}
    results_nfz    = {}

    # ── Condition 1: Hybrid — standard env ────────────────────────────
    print(f"\n[1/4] HYBRID GNN-LLM  —  standard env (no NFZ)")
    results_no_nfz["hybrid"] = _run_hybrid_std_capture(
        eps, args.max_steps, args.camera, args.fps, record)

    # ── Condition 2: VLM — standard env ───────────────────────────────
    print(f"\n[2/4] VLM  —  standard env (no NFZ)")
    r2 = run_vlm_eval(
        episodes=eps,
        max_steps=args.max_steps,
        camera=args.camera,
        fps=args.fps,
        record=record,
        use_nfz=False,
    )
    results_no_nfz["vlm"] = r2

    # ── Condition 3: Hybrid — NFZ env ─────────────────────────────────
    print(f"\n[3/4] HYBRID NFZ-aware  —  NFZ env")
    results_nfz["hybrid"] = _run_hybrid_nfz_capture(
        eps, args.max_steps, args.camera, args.fps, record)

    # ── Condition 4: VLM — NFZ env ────────────────────────────────────
    print(f"\n[4/4] VLM  —  NFZ env")
    r4 = run_vlm_eval(
        episodes=eps,
        max_steps=args.max_steps,
        camera=args.camera,
        fps=args.fps,
        record=record,
        use_nfz=True,
    )
    results_nfz["vlm"] = r4

    # ------------------------------------------------------------------
    # Print and save
    # ------------------------------------------------------------------
    print_table(results_no_nfz, results_nfz)
    save_results(results_no_nfz, results_nfz, out_dir)
    
    # Save a text report to the videos folder to group them
    try:
        import sys
        report_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "..", "videos", "final_4conditions_10eps", "benchmark_report.txt"
        )
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            original_stdout = sys.stdout
            sys.stdout = f
            print_table(results_no_nfz, results_nfz)
            sys.stdout = original_stdout
        print(f"\nSaved final benchmark report to: {report_path}")
    except Exception as e:
        print(f"\nCould not save local txt report: {e}")


def _run_hybrid_std_capture(episodes, max_steps, camera, fps, record):
    """Run HybridGNNLLMController in standard env and return metrics dict."""
    import numpy as np
    import random
    from envs.drone_nav_3d import DroneNav3DEnv
    from controllers.hybrid_gnn_llm_controller import HybridGNNLLMController
    from datetime import datetime
    import cv2, os
    from tqdm import tqdm

    env  = DroneNav3DEnv(gui=False)
    ctrl = HybridGNNLLMController()

    base_path = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(base_path, "..", "videos", "final_4conditions_10eps")
    os.makedirs(video_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    successes = collisions = timeouts = 0
    all_steps = []
    succ_steps = []
    goal_dists = []
    min_obs_dists = []

    for ep in tqdm(range(episodes), desc="Hybrid standard"):
        # Deterministic seed per episode
        ep_seed = ep * 1000
        random.seed(ep_seed)
        np.random.seed(ep_seed)
        env = DroneNav3DEnv(gui=False)

        obs = env.reset()
        done = False
        steps = 0
        info = {}
        ep_min = float("inf")

        video_writer = None
        if record:  # Record all episodes
            vp = os.path.join(video_dir,
                              f"3d_hybrid_std_ep{ep+1}_{timestamp}.mp4")
            video_writer = cv2.VideoWriter(
                vp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))

        while not done and steps < max_steps:
            action = ctrl.act(obs)
            obs, rew, done, info = env.step(action)
            steps += 1

            pos = obs.get("pos", np.zeros(3))
            for ob in obs.get("obstacles", []):
                d = np.linalg.norm(pos - ob["pos"]) - ob.get("radius", 0.1)
                ep_min = min(ep_min, d)

            if record and video_writer:
                frame = env.render(width=640, height=480, camera=camera)
                video_writer.write(frame)

        if video_writer:
            video_writer.release()

        all_steps.append(steps)
        goal_dists.append(info.get("dist_to_goal", 0))
        if ep_min < float("inf"):
            min_obs_dists.append(ep_min)

        if info.get("success"):
            successes += 1
            succ_steps.append(steps)
            outcome = "SUCCESS"
        elif info.get("collision"):
            collisions += 1
            outcome = "COLLISION"
        else:
            timeouts += 1
            outcome = "TIMEOUT"

        tqdm.write(f"  Ep {ep+1:2d}: {outcome:9s} | {steps:4d} steps | "
                   f"dist={info.get('dist_to_goal', 0):.2f}m")

        env.close()

    print(f"\n  Hybrid standard: {successes}/{episodes} success "
          f"({100*successes/episodes:.1f}%)")

    return {
        "method": "Hybrid_GNN_LLM",
        "episodes": episodes,
        "successes": successes,
        "collisions": collisions,
        "timeouts": timeouts,
        "nfz_violations": 0,
        "success_rate":   successes  / episodes,
        "collision_rate": collisions / episodes,
        "timeout_rate":   timeouts   / episodes,
        "nfz_violation_rate": 0.0,
        "avg_steps_all":  float(np.mean(all_steps)),
        "avg_steps_succ": float(np.mean(succ_steps)) if succ_steps else None,
        "avg_final_dist": float(np.mean(goal_dists)),
        "min_obs_dist":   float(np.mean(min_obs_dists)) if min_obs_dists else None,
        "video_files": [],
    }


def _run_hybrid_nfz_capture(episodes, max_steps, camera, fps, record):
    """
    Re-run Hybrid NFZ and capture metrics into the common schema.
    (eval_nfz.run_nfz_eval doesn't return a dict — we run it ourselves
    using the same env/controller the existing script uses, unchanged.)
    """
    import numpy as np
    import random
    from envs.drone_nav_nfz import DroneNavNFZEnv
    from controllers.hybrid_gnn_llm_controller import HybridGNNLLMController
    from datetime import datetime
    import cv2, os
    from tqdm import tqdm

    ctrl = HybridGNNLLMController()

    base_path = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(base_path, "..", "videos", "testing_final_20eps")
    os.makedirs(video_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    successes = collisions = timeouts = nfz_violations = 0
    all_steps = []
    succ_steps = []
    goal_dists = []
    min_obs_dists = []

    for ep in tqdm(range(episodes), desc="Hybrid NFZ capture"):
        # Deterministic seed per episode
        ep_seed = ep * 1000 + 500
        random.seed(ep_seed)
        np.random.seed(ep_seed)
        env = DroneNavNFZEnv(gui=False)

        obs = env.reset()
        done = False
        steps = 0
        info = {}
        ep_nfz = False
        ep_min = float("inf")

        video_writer = None
        if record:  # Record all episodes
            vp = os.path.join(video_dir,
                              f"3d_hybrid_nfz_capture_ep{ep+1}_{timestamp}.mp4")
            video_writer = cv2.VideoWriter(
                vp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))

        while not done and steps < max_steps:
            action = ctrl.act(obs)
            obs, rew, done, info = env.step(action)
            steps += 1

            if info.get("nfz_violation") or info.get("nfz_blocked"):
                ep_nfz = True

            pos = obs.get("pos", np.zeros(3))
            for ob in obs.get("obstacles", []):
                d = np.linalg.norm(pos - ob["pos"]) - ob.get("radius", 0.1)
                ep_min = min(ep_min, d)

            if record and video_writer:
                frame = env.render(width=640, height=480, camera=camera)
                video_writer.write(frame)

        if video_writer:
            video_writer.release()

        all_steps.append(steps)
        goal_dists.append(info.get("dist_to_goal", 0))
        if ep_min < float("inf"):
            min_obs_dists.append(ep_min)
        if ep_nfz:
            nfz_violations += 1

        if info.get("success"):
            successes += 1
            succ_steps.append(steps)
            outcome = "SUCCESS"
        elif info.get("collision"):
            collisions += 1
            outcome = "COLLISION"
        else:
            timeouts += 1
            outcome = "TIMEOUT"

        tqdm.write(f"  Ep {ep+1:2d}: {outcome:9s} | {steps:4d} steps | "
                   f"dist={info.get('dist_to_goal', 0):.2f}m"
                   + (" | NFZ!" if ep_nfz else ""))

        env.close()

    return {
        "method": "Hybrid_NFZ",
        "episodes": episodes,
        "successes": successes,
        "collisions": collisions,
        "timeouts": timeouts,
        "nfz_violations": nfz_violations,
        "success_rate":  successes  / episodes,
        "collision_rate": collisions / episodes,
        "timeout_rate":  timeouts   / episodes,
        "nfz_violation_rate": nfz_violations / episodes,
        "avg_steps_all":  float(np.mean(all_steps)),
        "avg_steps_succ": float(np.mean(succ_steps)) if succ_steps else None,
        "avg_final_dist": float(np.mean(goal_dists)),
        "min_obs_dist":   float(np.mean(min_obs_dists)) if min_obs_dists else None,
        "video_files": [],
    }


if __name__ == "__main__":
    main()
