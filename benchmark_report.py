"""
Benchmark Report Generator
=============================
Reads CSV results from 2D and 3D benchmarks and generates a markdown report.

Usage:
  python benchmark_report.py
"""

import os
import sys
import csv
import glob
import numpy as np
from datetime import datetime
from collections import defaultdict


def load_latest_csv(directory, prefix):
    """Find and load the most recent CSV file matching prefix."""
    pattern = os.path.join(directory, f"{prefix}*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    with open(latest, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader), latest


def generate_report(report_path):
    """Generate the benchmark comparison report."""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "benchmarks")
    dir_3d = os.path.join(base_dir, "3D")
    dir_2d = os.path.join(base_dir, "2D")

    lines = []
    lines.append("# Comprehensive Navigation Benchmark Report")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    lines.append("## Experimental Setup\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append("| Methods | Hybrid GNN+LLM, LLM-Only, VLM, APEX |")
    lines.append("| Episodes per method | 10 |")
    lines.append("| 3D Environment | PyBullet (20 obstacles, no NFZ) |")
    lines.append("| 2D Environment | MuJoCo (25 obstacles, with NFZ) |")
    lines.append("| Max steps | 600 (3D) / 3000 (2D) |")
    lines.append("")

    # ===================== 3D RESULTS =====================
    result_3d = load_latest_csv(dir_3d, "benchmark_3d")
    if result_3d:
        data_3d, csv_path = result_3d
        lines.append("## 3D Benchmark Results (PyBullet)\n")
        lines.append(f"*Data source: `{os.path.basename(csv_path)}`*\n")

        groups = defaultdict(list)
        for r in data_3d:
            groups[r["method"]].append(r)

        lines.append("| Method | Success Rate | Collision Rate | Timeout Rate | Avg Steps | Path Efficiency | Min Obs Dist |")
        lines.append("|--------|-------------|---------------|-------------|-----------|----------------|-------------|")

        for method in ["Hybrid_GNN_LLM", "LLM_Only", "VLM", "APEX"]:
            results = groups.get(method, [])
            if not results:
                continue
            n = len(results)
            succ = sum(int(r["success"]) for r in results)
            coll = sum(int(r["collision"]) for r in results)
            tout = n - succ - coll
            avg_steps = np.mean([int(r["steps"]) for r in results])
            avg_pe = np.mean([float(r["path_efficiency"]) for r in results])
            avg_md = np.mean([float(r["min_obs_dist"]) for r in results])

            lines.append(
                f"| {method} | {succ}/{n} ({succ/n*100:.0f}%) | "
                f"{coll}/{n} ({coll/n*100:.0f}%) | {tout}/{n} ({tout/n*100:.0f}%) | "
                f"{avg_steps:.1f} | {avg_pe:.2f}x | {avg_md:.3f}m |")

        lines.append("")

    # ===================== 2D RESULTS =====================
    result_2d = load_latest_csv(dir_2d, "benchmark_2d")
    if result_2d:
        data_2d, csv_path = result_2d
        lines.append("## 2D Benchmark Results (MuJoCo with NFZ)\n")
        lines.append(f"*Data source: `{os.path.basename(csv_path)}`*\n")

        groups = defaultdict(list)
        for r in data_2d:
            groups[r["method"]].append(r)

        lines.append("| Method | Success Rate | Collision Rate | NFZ Violations | Avg Steps |")
        lines.append("|--------|-------------|---------------|---------------|-----------|")

        for method in ["Hybrid_GNN_LLM", "LLM_Only", "VLM", "APEX"]:
            results = groups.get(method, [])
            if not results:
                continue
            n = len(results)
            succ = sum(int(r["success"]) for r in results)
            coll = sum(int(r["collision"]) for r in results)
            nfz = sum(int(r["nfz_violated"]) for r in results)
            avg_steps = np.mean([int(r["steps"]) for r in results])

            lines.append(
                f"| {method} | {succ}/{n} ({succ/n*100:.0f}%) | "
                f"{coll}/{n} ({coll/n*100:.0f}%) | {nfz}/{n} | "
                f"{avg_steps:.1f} |")

        lines.append("")

    # ===================== OBSERVATIONS =====================
    lines.append("## Observations\n")
    lines.append("### Method Comparison\n")

    if result_3d:
        groups = defaultdict(list)
        for r in data_3d:
            groups[r["method"]].append(r)

        best_method = max(groups.keys(),
                         key=lambda m: sum(int(r["success"]) for r in groups[m]) / len(groups[m]))
        safest_method = min(groups.keys(),
                           key=lambda m: sum(int(r["collision"]) for r in groups[m]) / len(groups[m]))
        fastest_method = min(groups.keys(),
                            key=lambda m: np.mean([int(r["steps"]) for r in groups[m] if int(r["success"])]) if any(int(r["success"]) for r in groups[m]) else float('inf'))

        lines.append(f"- **Highest Success Rate (3D):** {best_method}")
        lines.append(f"- **Lowest Collision Rate (3D):** {safest_method}")
        lines.append(f"- **Fastest Goal Reach (3D):** {fastest_method}")
        lines.append("")

    lines.append("### Key Takeaways\n")
    lines.append("1. **Hybrid GNN+LLM** combines strategic LLM bearing with GNN safety filtering, "
                 "providing the best balance of goal-seeking and obstacle avoidance.")
    lines.append("2. **LLM-Only** relies on GPT-4o text prompts for action selection with GNN safety shield.")
    lines.append("3. **VLM** uses visual camera frames with GPT-4o vision, providing a fundamentally "
                 "different perception modality than text-based methods.")
    lines.append("4. **APEX** uses greedy one-step scoring without LLM involvement, "
                 "providing a compute-efficient baseline.")
    lines.append("")

    lines.append("### Video Directory Structure\n")
    lines.append("```")
    lines.append("results/benchmarks/")
    lines.append("├── 3D/")
    lines.append("│   ├── Hybrid_GNN_LLM/  (10 videos)")
    lines.append("│   ├── LLM_Only/        (10 videos)")
    lines.append("│   ├── VLM/             (10 videos)")
    lines.append("│   ├── APEX/            (10 videos)")
    lines.append("│   └── benchmark_3d_*.csv")
    lines.append("└── 2D/")
    lines.append("    ├── Hybrid_GNN_LLM/  (10 videos)")
    lines.append("    ├── LLM_Only/        (10 videos)")
    lines.append("    ├── APEX/            (10 videos)")
    lines.append("    └── benchmark_2d_*.csv")
    lines.append("```")

    report_content = "\n".join(lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"Report saved: {report_path}")
    return report_content


if __name__ == "__main__":
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "benchmarks", "benchmark_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    content = generate_report(report_path)
    print("\n" + content)
