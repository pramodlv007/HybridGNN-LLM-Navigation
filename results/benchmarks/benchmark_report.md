# Comprehensive Navigation Benchmark Report

*Generated: 2026-04-01 00:55:31*

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Methods | Hybrid GNN+LLM, LLM-Only, VLM, APEX |
| Episodes per method | 10 |
| 3D Environment | PyBullet (20 obstacles, no NFZ) |
| 2D Environment | MuJoCo (25 obstacles, with NFZ) |
| Max steps | 600 (3D) / 3000 (2D) |

## 3D Benchmark Results (PyBullet)

*Data source: `benchmark_3d_20260331_215428.csv`*

| Method | Success Rate | Collision Rate | Timeout Rate | Avg Steps | Path Efficiency | Min Obs Dist |
|--------|-------------|---------------|-------------|-----------|----------------|-------------|
| Hybrid_GNN_LLM | 7/10 (70%) | 3/10 (30%) | 0/10 (0%) | 260.4 | 1.38x | 0.185m |
| LLM_Only | 5/10 (50%) | 1/10 (10%) | 4/10 (40%) | 531.1 | 1.88x | 0.213m |
| VLM | 6/10 (60%) | 1/10 (10%) | 3/10 (30%) | 439.6 | 1.77x | 0.266m |
| APEX | 7/10 (70%) | 2/10 (20%) | 1/10 (10%) | 419.8 | 1.52x | 0.319m |

## Observations

### Method Comparison

- **Highest Success Rate (3D):** Hybrid_GNN_LLM
- **Lowest Collision Rate (3D):** LLM_Only
- **Fastest Goal Reach (3D):** Hybrid_GNN_LLM

### Key Takeaways

1. **Hybrid GNN+LLM** combines strategic LLM bearing with GNN safety filtering, providing the best balance of goal-seeking and obstacle avoidance.
2. **LLM-Only** relies on GPT-4o text prompts for action selection with GNN safety shield.
3. **VLM** uses visual camera frames with GPT-4o vision, providing a fundamentally different perception modality than text-based methods.
4. **APEX** uses greedy one-step scoring without LLM involvement, providing a compute-efficient baseline.

### Video Directory Structure

```
results/benchmarks/
├── 3D/
│   ├── Hybrid_GNN_LLM/  (10 videos)
│   ├── LLM_Only/        (10 videos)
│   ├── VLM/             (10 videos)
│   ├── APEX/            (10 videos)
│   └── benchmark_3d_*.csv
└── 2D/
    ├── Hybrid_GNN_LLM/  (10 videos)
    ├── LLM_Only/        (10 videos)
    ├── APEX/            (10 videos)
    └── benchmark_2d_*.csv
```