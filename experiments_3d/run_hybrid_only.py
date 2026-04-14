"""
Hybrid-Only Benchmark Runner (Deterministic)
==============================================
Seeds each episode for 100% reproducible results.
Runs ONLY Hybrid GNN-LLM controller — no VLM API costs.
"""

import argparse, os, sys, json, random
from datetime import datetime
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_hybrid(env_type, episodes, max_steps):
    """Run Hybrid controller with per-episode seeding for stability."""
    from controllers.hybrid_gnn_llm_controller import HybridGNNLLMController

    if env_type == "nfz":
        from envs.drone_nav_nfz import DroneNavNFZEnv
    else:
        from envs.drone_nav_3d import DroneNav3DEnv

    ctrl = HybridGNNLLMController()

    successes = collisions = timeouts = nfz_violations = 0
    all_steps, succ_steps, goal_dists = [], [], []

    for ep in tqdm(range(episodes), desc=f"Hybrid {env_type}"):
        # Deterministic seed per episode — same obstacle patterns every run
        ep_seed = ep * 1000 + (0 if env_type == "standard" else 500)
        random.seed(ep_seed)
        np.random.seed(ep_seed)

        # Create fresh env each episode to ensure clean seed state
        if env_type == "nfz":
            env = DroneNavNFZEnv(gui=False)
        else:
            env = DroneNav3DEnv(gui=False)

        obs = env.reset()
        done, steps, info = False, 0, {}
        ep_nfz = False

        while not done and steps < max_steps:
            action = ctrl.act(obs)
            obs, rew, done, info = env.step(action)
            steps += 1
            if info.get("nfz_violation") or info.get("nfz_blocked"):
                ep_nfz = True

        all_steps.append(steps)
        goal_dists.append(info.get("dist_to_goal", 0))
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
        "env": env_type,
        "episodes": episodes,
        "successes": successes,
        "collisions": collisions,
        "timeouts": timeouts,
        "nfz_violations": nfz_violations,
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "timeout_rate": timeouts / episodes,
        "nfz_violation_rate": nfz_violations / episodes if env_type == "nfz" else 0.0,
        "avg_steps_all": float(np.mean(all_steps)),
        "avg_steps_succ": float(np.mean(succ_steps)) if succ_steps else None,
        "avg_final_dist": float(np.mean(goal_dists)),
    }


def print_results(std, nfz):
    sep = "=" * 70
    mid = "-" * 70
    print(f"\n{sep}")
    print(f"  HYBRID GNN-LLM BENCHMARK RESULTS (DETERMINISTIC)")
    print(f"{sep}")
    print(f"\n{'STANDARD (no NFZ)':^70}")
    print(mid)
    print(f"  Success Rate:    {std['success_rate']*100:5.1f}%  "
          f"({std['successes']}/{std['episodes']})")
    print(f"  Collision Rate:  {std['collision_rate']*100:5.1f}%")
    print(f"  Timeout Rate:    {std['timeout_rate']*100:5.1f}%")
    print(f"  Avg Steps (all): {std['avg_steps_all']:.0f}")
    if std['avg_steps_succ']:
        print(f"  Avg Steps (suc): {std['avg_steps_succ']:.0f}")
    print(f"  Avg Final Dist:  {std['avg_final_dist']:.3f}m")

    print(f"\n{'NFZ ENVIRONMENT':^70}")
    print(mid)
    print(f"  Success Rate:    {nfz['success_rate']*100:5.1f}%  "
          f"({nfz['successes']}/{nfz['episodes']})")
    print(f"  Collision Rate:  {nfz['collision_rate']*100:5.1f}%")
    print(f"  Timeout Rate:    {nfz['timeout_rate']*100:5.1f}%")
    print(f"  NFZ Violations:  {nfz['nfz_violation_rate']*100:5.1f}%")
    print(f"  Avg Steps (all): {nfz['avg_steps_all']:.0f}")
    if nfz['avg_steps_succ']:
        print(f"  Avg Steps (suc): {nfz['avg_steps_succ']:.0f}")
    print(f"  Avg Final Dist:  {nfz['avg_final_dist']:.3f}m")

    std_ok = "PASS" if std['success_rate'] > 0.80 else "FAIL"
    nfz_ok = "PASS" if nfz['success_rate'] > 0.75 else "FAIL"
    print(f"\n{mid}")
    print(f"  Standard >80%: {std_ok}  |  NFZ >75%: {nfz_ok}")
    print(f"{sep}\n")


def main():
    parser = argparse.ArgumentParser(description="Hybrid-only benchmark (deterministic)")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=800)
    args = parser.parse_args()

    print("\n[1/2] Running Hybrid - Standard env...")
    std = run_hybrid("standard", args.episodes, args.max_steps)

    print("\n[2/2] Running Hybrid - NFZ env...")
    nfz = run_hybrid("nfz", args.episodes, args.max_steps)

    print_results(std, nfz)

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "hybrid_tuning")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"hybrid_deterministic_{ts}.json")
    with open(path, "w") as f:
        json.dump({"standard": std, "nfz": nfz}, f, indent=2, default=str)
    print(f"  Saved -> {path}")


if __name__ == "__main__":
    main()
