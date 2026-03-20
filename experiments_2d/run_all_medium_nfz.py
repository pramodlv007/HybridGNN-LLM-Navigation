"""
Run All Medium NFZ Experiments
================================
Runs all 3 methods on the Medium NFZ environment (2 NFZ zones).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medium_nfz_experiment import run_medium_nfz_experiment


def main():
    methods = ["APEX", "PULSE", "HYBRID"]
    method_labels = {
        "APEX": "Expt 1: APEX Original",
        "PULSE": "Expt 2: Pulse / 3-Tier",
        "HYBRID": "Expt 3: Hybrid GNN-LLM"
    }

    results = {}

    print("=" * 70)
    print("  MEDIUM NFZ EXPERIMENT SUITE - 2 No-Fly Zones")
    print("  Zone 1: Center [-1.2, 1.2] x [-1.2, 1.2]")
    print("  Zone 2: Right Side [2.5, 3.8] x [-3.0, 3.0]")
    print("=" * 70)

    for method in methods:
        print(f"\n>>> Starting {method_labels[method]}...")
        try:
            result = run_medium_nfz_experiment(method=method, obstacle_speed=1.0)
            results[method] = result
        except Exception as e:
            print(f"  ERROR running {method}: {e}")
            results[method] = {"collision": "ERROR", "nfz_violated": "ERROR",
                               "time": 0, "steps": 0}

    print("\n")
    print("=" * 70)
    print("  MEDIUM NFZ EXPERIMENT SUMMARY (2 zones)")
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
