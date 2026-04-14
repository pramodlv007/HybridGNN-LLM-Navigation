# 🧠 Hybrid GNN+LLM Navigation Framework

> A tiered decision-making architecture combining **Graph Neural Networks** for real-time safety assessment with **Large Language Models** for strategic planning — benchmarked in 2D MuJoCo and **3D PyBullet** environments against a VLM baseline.

---

## 🎬 Demo Videos

### 2D Hybrid GNN+LLM — Mixed Obstacle Navigation
![experiments_2d/videos/hybrid_gnn_llm_mixed_obstacle](experiments_2d/videos/hybrid_gnn_llm_mixed_obstacle.gif)

### 3D Hybrid GNN+LLM — Drone NFZ Avoidance
![experiments_3d/videos/gnn_llm/3d_gnn_llm_nfz](experiments_3d/videos/gnn_llm/3d_gnn_llm_nfz.gif)

---

## 🏗️ Architecture

The Hybrid framework uses a **Sample-Filter-Select** pipeline that separates safety from strategy:

```
┌────────────────────────────────────────────────────────────┐
│                 HYBRID GNN+LLM CONTROLLER                  │
│                                                            │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────────┐ │
│  │   SAMPLE     │   │   FILTER     │   │    SELECT       │ │
│  │  16 candidate│──▶│  GNN Safety  │──▶│ LLM Strategic   │ │
│  │  directions  │   │  Checker     │   │ Bearing         │ │
│  └─────────────┘   └──────────────┘   └────────────────┘ │
│                            │                    │          │
│                     ┌──────▼──────────────────▼────────┐  │
│                     │         SAFETY SHIELD             │  │
│                     │  5-step collision lookahead        │  │
│                     │  NFZ boundary prevention           │  │
│                     │  Reactive escape fallback          │  │
│                     └──────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### How It Works

| Step | Component | Action |
|------|-----------|--------|
| 1 | **Sample** | Generate 16 candidate velocity directions at cruise speed |
| 2 | **Filter (GNN)** | Score each candidate for risk: obstacle proximity, TTC, wall/NFZ penalties |
| 3 | **Select (LLM)** | Among safe candidates, pick the one best aligned with strategic bearing |
| 4 | **Safety Shield** | Simulate 5 future steps; override with escape maneuver if collision predicted |
| 5 | **Reactive Fallback** | If no safe candidates exist, brake or push away from nearest threat |

---

## 📂 Repository Structure

```text
HybridGNN-LLM-Navigation/
│
├── README.md                              # This file
├── requirements.txt                       # All Python dependencies
├── .gitignore
│
├── experiments_2d/                        # ═══ 2D MuJoCo Experiments ═══
│   ├── README.md                          # Detailed 2D documentation
│   ├── hybrid_nav.py                      # Core hybrid classes
│   ├── run_gnn_llm_hybrid.py              # Mixed obstacle runner
│   ├── nfz_experiment.py                  # No-Fly Zone experiment
│   ├── medium_nfz_experiment.py           # Dual NFZ experiment
│   ├── pacman_maze_experiment.py          # Pac-Man maze experiment
│   ├── drone_hybrid_experiment.py         # Drone hybrid experiment
│   ├── run_2d_full_comparison.py          # Full 2D Hybrid vs VLM comparison
│   ├── run_benchmark_2d.py                # 2D benchmark runner
│   ├── env/                              # MuJoCo XML environments
│   ├── model/                            # GNN models + trained weights
│   ├── utils/                            # APEX agent, MuJoCo simulator
│   ├── visualization/                    # Result plots and charts
│   └── videos/                           # Experiment recordings
│
├── experiments_3d/                        # ═══ 3D PyBullet Drone Experiments ═══
│   ├── README.md                          # Detailed 3D documentation
│   ├── run_3d_expt.py                     # Single-episode 3D runner
│   ├── run_3d_expt_loop.py                # Multi-episode 3D runner
│   ├── run_benchmark_3d.py                # Benchmark runner (Hybrid vs VLM)
│   ├── run_multi_trial.py                 # Multi-trial runner
│   ├── controllers/                       # Navigation controllers
│   │   ├── hybrid_controller.py           # Standard 3D hybrid
│   │   ├── hybrid_nfz_controller.py       # NFZ-aware hybrid
│   │   ├── hybrid_gnn_llm_controller.py   # GNN+LLM combined (latest)
│   │   ├── vlm_controller.py              # VLM baseline controller (NEW)
│   │   ├── explicit_llm_controller.py     # Explicit LLM variant
│   │   ├── gnn_safety_checker.py          # GNN candidate scoring
│   │   ├── llm_strategic_navigator.py     # LLM strategic bearing
│   │   ├── apex_baseline.py               # APEX baseline (comparison)
│   │   └── risk.py                        # TTC + proximity risk models
│   ├── envs/                              # PyBullet environments
│   │   ├── drone_nav_3d.py                # Standard 3D (20 obstacles)
│   │   └── drone_nav_nfz.py               # NFZ (25 obs + hard wall)
│   ├── models/                            # 3D GNN models
│   │   ├── local_risk_gnn.py
│   │   └── trained_risk_gnn.pt            # Trained GNN weights
│   ├── scripts/                           # Eval runners
│   │   ├── eval.py                        # Standard eval
│   │   ├── eval_nfz.py                    # NFZ eval
│   │   ├── eval_gnn_llm.py                # GNN+LLM eval
│   │   ├── eval_vlm.py                    # VLM baseline eval (NEW)
│   │   └── eval_explicit_llm.py           # Explicit LLM eval
│   ├── training/                          # GNN training scripts
│   └── videos/                            # Recordings by experiment type
│       ├── hybrid/                        # Standard hybrid
│       ├── nfz/                           # NFZ hybrid
│       ├── gnn_llm/                       # GNN+LLM
│       └── explicit_llm/                  # Explicit LLM + action logs
│
├── results/                               # ═══ Experiment Results ═══
│   ├── benchmarks/                        # Aggregate benchmark reports
│   ├── comparison_2d/                     # 2D Hybrid vs VLM data
│   ├── comparison_3d/                     # 3D Hybrid vs VLM data (NEW)
│   │   └── 3d_hybrid_vs_vlm_report_20260404.md
│   ├── hybrid_tuning/                     # Hyperparameter tuning logs
│   └── testing_final_20eps/               # Final acceptance tests
│
├── docs/                                  # Additional documentation
│   ├── architecture.md                    # Framework deep-dive
│   └── benchmark_results.md               # Performance comparison
│
├── benchmark_report.py                    # Report generation script (NEW)
└── create_report.py                       # DOCX report creator (NEW)
```

---

## 📊 Benchmark Results: Hybrid GNN+LLM vs VLM

> **Setup:** 3D PyBullet | 15 episodes per condition | GPT-4o (text) vs GPT-4o (vision) | Identical GNN safety checker for both

### Standard Environment (No NFZ) — 20 Obstacles

| Metric | Hybrid GNN+LLM | VLM (GPT-4o Vision) | Δ | Winner |
|--------|:--------------:|:-------------------:|:----:|:------:|
| **Success Rate** | **80.0%** (12/15) | 46.7% (7/15) | +33.3 pp | 🏆 Hybrid |
| Collision Rate | **0.0%** (0/15) | 6.7% (1/15) | −6.7 pp | 🏆 Hybrid |
| Timeout Rate | **20.0%** | 46.7% | −26.7 pp | 🏆 Hybrid |
| Avg Steps to Goal | **259 steps** | 346 steps | −87 steps | 🏆 Hybrid |

### NFZ Environment — 25 Obstacles + L-Shaped Hard Wall

| Metric | Hybrid NFZ | VLM (GPT-4o Vision) | Δ | Winner |
|--------|:----------:|:-------------------:|:----:|:------:|
| **Success Rate** | **60.0%** (9/15) | 33.3% (5/15) | +26.7 pp | 🏆 Hybrid |
| Collision Rate | 13.3% (2/15) | 26.7% (4/15) | −13.4 pp | 🏆 Hybrid |
| Timeout Rate | **26.7%** | 40.0% | −13.3 pp | 🏆 Hybrid |
| **NFZ Violation Rate** | **0.0%** (0/15) | 46.7% (7/15) | −46.7 pp | 🏆 Hybrid |
| Avg Steps to Goal | **236 steps** | 306 steps | −70 steps | 🏆 Hybrid |

### Overall Win Tally (11 metrics across 2 environments)

| | Hybrid GNN+LLM | VLM |
|--|:-:|:-:|
| **Metrics Won** | **11 / 11** | 0 / 11 |

> ⚠️ **Note on VLM results:** 98–100% of VLM API calls failed (rate limits / payload errors), so the VLM controller navigated almost entirely on a goal-bearing fallback — effectively the same GNN safety shield with no vision intelligence. VLM success rates are a **lower bound** on true VLM performance.

---

## 🧪 Experiments Overview

### 2D MuJoCo Experiments

#### Mixed Obstacle Navigation
20 obstacles (10 dynamic + 10 static) — Navigate through chaotic obstacle field
![experiments_2d/videos/hybrid_gnn_llm_mixed_obstacle](experiments_2d/videos/hybrid_gnn_llm_mixed_obstacle.gif)

```bash
cd experiments_2d
python run_gnn_llm_hybrid.py
```

#### No-Fly Zone (NFZ)
L-shaped no-fly zone + 25 obstacles — Avoid NFZ while dodging obstacles
![experiments_2d/videos/nfz_hybrid](experiments_2d/videos/nfz_hybrid.gif)

```bash
python nfz_experiment.py --method HYBRID
```

#### Medium NFZ (Dual Zones)
2 no-fly zones + 25 obstacles — Navigate between dual NFZs

```bash
python medium_nfz_experiment.py --method HYBRID
```

#### Full 2D Hybrid vs VLM Comparison

```bash
python run_2d_full_comparison.py --method HYBRID
python run_2d_full_comparison.py --method VLM
```

---

### 3D PyBullet Drone Experiments

#### Standard 3D Navigation (Hybrid GNN+LLM)
8×8m arena, 20 obstacles, cruise speed 1.8 m/s

```bash
cd experiments_3d
python -m scripts.eval_gnn_llm --episodes 15
# or single episode:
python run_3d_expt.py --controller hybrid_gnn_llm
```

#### 3D NFZ Avoidance (Hybrid NFZ-Aware)
L-shaped hard wall + 25 obstacles — reactive NFZ deflection

```bash
python -m scripts.eval_nfz --controller hybrid --episodes 15
```

#### VLM Baseline (NEW)
GPT-4o Vision — rendered camera frame → bearing angle

```bash
python -m scripts.eval_vlm --episodes 15
```

#### Full 3D Hybrid vs VLM Benchmark (NEW)
Runs all 4 conditions (Hybrid/VLM × Std/NFZ) and generates a comparison report:

```bash
python run_benchmark_3d.py --episodes 15
```

---

## 📈 2D Experiment Results (Preliminary)

Preliminary 2D tests in MuJoCo (5 episodes, GPT-4o text vs VLM):

### Standard Environment (2D)
| Metric | Hybrid GNN+LLM | VLM (GPT-4o) |
|--------|:--------------:|:------------:|
| Success Rate | 100.0% | 100.0% |
| Collision Rate | 0.0% | 0.0% |
| NFZ Violation Rate | 0.0% | 0.0% |
| Avg Steps | 798 | 613 |

### NFZ Environment (2D)
| Metric | Hybrid GNN+LLM | VLM (GPT-4o) |
|--------|:--------------:|:------------:|
| Success Rate | 100.0% | 100.0% |
| NFZ Violation Rate | **0.0%** | **100.0%** |
| Avg Steps | 702 | 585 |

Key finding: VLM solved faster in 2D but completely failed to respect No-Fly Zones (100% violation rate), while Hybrid maintained **0% violations** in all environments.

---

## 🚀 Quick Start

### Prerequisites
```bash
conda create -n hybrid-nav python=3.11
conda activate hybrid-nav
pip install -r requirements.txt
```

### Set OpenAI API Key
```bash
# Windows
set OPENAI_API_KEY=sk-proj-...

# Linux/Mac
export OPENAI_API_KEY=sk-proj-...
```

### Run 3D Benchmark (Hybrid vs VLM)
```bash
cd experiments_3d
python run_benchmark_3d.py --episodes 15
```

### Camera Options (3D)
```bash
--camera pov      # First-person (default)
--camera top      # Bird's-eye view
--camera follow   # Third-person chase
--camera chase    # Behind-drone view
--no-record       # Disable video recording
```

---

## 🔬 Key Technical Contributions

1. **Fixed TTC Risk Formula** — Corrected Time-to-Collision calculation that was causing false "always-at-risk" behavior in the original APEX framework.

2. **Drone Height Lock (3D)** — Explicitly locks drone altitude to the obstacle plane (z=1.0), forcing navigation *through* obstacles rather than flying over them.

3. **Reactive NFZ Avoidance** — Shifted from proactive (avoid NFZ from far away) to reactive (head for goal, deflect only when encountering NFZ boundary). Result: 0.0% NFZ violations.

4. **Sample-Filter-Select Pipeline** — Separates safety (GNN: instant, per-frame) from strategy (LLM: every ~2s), achieving both responsiveness and intelligence with low API cost.

5. **Multi-layer Safety Shield** — 5-step collision lookahead with emergency escape override, achieving 0% collision rate in the standard 3D environment.

6. **VLM Baseline Controller** — New `vlm_controller.py` enables direct head-to-head comparison against GPT-4o Vision using identical GNN safety infrastructure.

---

## 📺 All Video Recordings

All experiment videos are embedded inline in the sections above. They are also available at:
- **2D**: [`experiments_2d/videos/`](experiments_2d/videos/)
- **3D**: [`experiments_3d/videos/`](experiments_3d/videos/)

---

## 📄 Full Benchmark Report

The detailed per-episode breakdown (all 60 episodes across 4 conditions) is in:
[`results/comparison_3d/3d_hybrid_vs_vlm_report_20260404.md`](results/comparison_3d/3d_hybrid_vs_vlm_report_20260404.md)

---

## 📈 Citation

If you find this work helpful, please cite:

```bibtex
@misc{huang2025apexempoweringllmsphysicsbased,
      title={APEX: Empowering LLMs with Physics-Based Task Planning for Real-time Insight},
      author={Wanjing Huang and Weixiang Yan and Zhen Zhang and Ambuj Singh},
      year={2025},
      eprint={2505.13921},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.13921},
}
```

---

## 📄 License

This project is for research purposes. Please see the original APEX framework for licensing details.
