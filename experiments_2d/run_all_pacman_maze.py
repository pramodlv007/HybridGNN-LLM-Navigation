"""
Run All Pac-Man Maze Experiments
================================
Runs APEX, PULSE, and HYBRID on the MuJoCo Pac-Man maze
and prints a comparison table.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pacman_maze_experiment import run_pacman_maze_experiment

def main():
    methods = ["PULSE", "HYBRID", "APEX"]
    all_results = {}

    for method in methods:
        print(f"\n{'#'*60}")
        print(f" RUNNING: {method}")
        print(f"{'#'*60}")
        result = run_pacman_maze_experiment(method=method, obstacle_speed=1.0)
        all_results[method] = result

    # Print comparison
    print(f"\n\n{'='*70}")
    print(f" PAC-MAN MAZE BENCHMARK COMPARISON")
    print(f"{'='*70}")
    print(f"{'Method':<12} | {'Reached Goal':<14} | {'Collision':<12} | {'Steps':<8} | {'Time':<8}")
    print(f"{'-'*70}")

    for method in methods:
        r = all_results.get(method)
        if r:
            max_steps = 3000
            success = r.get("steps", max_steps) < max_steps - 1
            print(f"{method:<12} | {'YES' if success else 'NO':<14} | "
                  f"{'YES' if r['collision'] else 'NO':<12} | "
                  f"{r['steps']:<8} | {r['time']:.1f}s")
        else:
            print(f"{method:<12} | ERROR")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
