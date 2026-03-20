"""
Run All NFZ Experiments
========================
Runs all 3 experiment methods on the NFZ environment sequentially
and prints a summary comparison table.

Usage:
    python experiments/cat_expt/run_all_nfz.py
"""

import os
import sys

# Ensure experiments_2d dir is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nfz_experiment import run_nfz_experiment


def main():
    methods = ["APEX", "PULSE", "HYBRID"]
    method_labels = {
        "APEX": "Expt 1: APEX Original",
        "PULSE": "Expt 2: Pulse / 3-Tier",
        "HYBRID": "Expt 3: Hybrid GNN-LLM"
    }

    results = {}

    print("=" * 70)
    print("  NFZ EXPERIMENT SUITE - Running All 3 Methods")
    print("  Robot must reach the goal without collisions or NFZ violations")
    print("=" * 70)

    for method in methods:
        print(f"\n>>> Starting {method_labels[method]}...")
        try:
            result = run_nfz_experiment(method=method, obstacle_speed=1.0)
            results[method] = result
        except Exception as e:
            print(f"  ERROR running {method}: {e}")
            results[method] = {"collision": "ERROR", "nfz_violated": "ERROR",
                               "time": 0, "steps": 0}

    # Summary table
    print("\n")
    print("=" * 70)
    print("  NFZ EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"  {'Method':<25} {'Goal?':<10} {'Collision':<12} {'NFZ Safe':<12} {'Time':<10}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")

    for method in methods:
        r = results.get(method, {})
        if r is None:
            r = {}
        collision = r.get("collision", "N/A")
        nfz = r.get("nfz_violated", "N/A")
        t = r.get("time", 0)
        steps = r.get("steps", 0)

        goal_reached = "[YES]" if steps > 0 and steps < 3000 - 1 else "[NO]"
        collision_str = "[YES]" if collision is True else ("[NO]" if collision is False else str(collision))
        nfz_str = "[YES]" if nfz is False else ("[NO]" if nfz is True else str(nfz))

        print(f"  {method_labels[method]:<25} {goal_reached:<10} {collision_str:<12} {nfz_str:<12} {t:<10.1f}s")

    print(f"  {'-'*69}")
    print("=" * 70)


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("WARNING: OPENAI_API_KEY not found. APEX method requires it.")
        print("Set it with: $env:OPENAI_API_KEY = 'your-key-here'\n")

    main()
