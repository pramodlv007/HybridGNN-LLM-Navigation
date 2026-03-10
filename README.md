# 🧠 Hybrid GNN+LLM Navigation Framework

> A tiered decision-making architecture combining **Graph Neural Networks** for real-time safety assessment with **Large Language Models** for strategic planning — tested across 2D MuJoCo and 3D PyBullet environments.

## 🎬 Demo Videos

### 2D Hybrid GNN+LLM — Mixed Obstacle Navigation
<video src="experiments_2d/videos/hybrid_gnn_llm_mixed_obstacle.mp4" controls width="600"></video>

### 3D Hybrid GNN+LLM — Drone NFZ Avoidance
<video src="experiments_3d/videos/gnn_llm/3d_gnn_llm_nfz.mp4" controls width="600"></video>

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
│   ├── run_all_*.py                       # Batch runners
│   ├── env/                               # MuJoCo XML environments
│   ├── model/                             # GNN models + trained weights
│   ├── utils/                             # APEX agent, MuJoCo simulator
│   └── videos/                            # Experiment recordings
│
├── experiments_3d/                        # ═══ 3D PyBullet Drone Experiments ═══
│   ├── README.md                          # Detailed 3D documentation
│   ├── controllers/                       # Navigation controllers
│   │   ├── hybrid_controller.py           # Standard 3D hybrid
│   │   ├── hybrid_nfz_controller.py       # NFZ-aware hybrid
│   │   ├── hybrid_gnn_llm_controller.py   # GNN+LLM combined
│   │   ├── explicit_llm_controller.py     # Explicit LLM variant
│   │   ├── gnn_safety_checker.py          # GNN candidate scoring
│   │   ├── llm_strategic_navigator.py     # LLM strategic bearing
│   │   ├── apex_baseline.py               # APEX baseline (comparison)
│   │   └── risk.py                        # TTC + proximity risk models
│   ├── envs/                              # PyBullet environments
│   │   ├── drone_nav_3d.py                # Standard 3D (20 obstacles)
│   │   └── drone_nav_nfz.py               # NFZ (25 obs + hard wall)
│   ├── models/                            # 3D GNN models
│   │   └── local_risk_gnn.py
│   ├── scripts/                           # Eval runners
│   │   ├── eval.py                        # Standard eval
│   │   ├── eval_nfz.py                    # NFZ eval
│   │   ├── eval_gnn_llm.py                # GNN+LLM eval
│   │   └── eval_explicit_llm.py           # Explicit LLM eval
│   └── videos/                            # Recordings by experiment type
│       ├── hybrid/                        # Standard hybrid
│       ├── nfz/                           # NFZ hybrid
│       ├── gnn_llm/                       # GNN+LLM
│       └── explicit_llm/                  # Explicit LLM + action logs
│
└── docs/                                  # Additional documentation
    ├── architecture.md                    # Framework deep-dive
    └── benchmark_results.md               # Performance comparison
```

---

## 🧪 Experiments Overview

### 2D MuJoCo Experiments

#### Mixed Obstacle Navigation
20 obstacles (10 dynamic + 10 static) — Navigate through chaotic obstacle field
<video src="experiments_2d/videos/hybrid_gnn_llm_mixed_obstacle.mp4" controls width="600"></video>

```bash
python run_gnn_llm_hybrid.py
```

#### No-Fly Zone (NFZ)
L-shaped no-fly zone + 25 obstacles — Avoid NFZ while dodging obstacles
<video src="experiments_2d/videos/nfz_hybrid.mp4" controls width="600"></video>

```bash
python nfz_experiment.py --method HYBRID
```

#### Medium NFZ (Dual Zones)
2 no-fly zones + 25 obstacles — Navigate between dual NFZs
<video src="experiments_2d/videos/medium_nfz_hybrid.mp4" controls width="600"></video>

```bash
python medium_nfz_experiment.py --method HYBRID
```

#### Pac-Man Maze
Walled maze corridors — Navigate tight corridors without wall collision
<video src="experiments_2d/videos/pacman_maze_hybrid.mp4" controls width="600"></video>

```bash
python pacman_maze_experiment.py --method HYBRID
```

#### Drone Hybrid
Drone-like physics — Navigate with drone movement model
<video src="experiments_2d/videos/drone_hybrid.mp4" controls width="600"></video>

```bash
python drone_hybrid_experiment.py
```

### 3D PyBullet Drone Experiments

#### Standard 3D Navigation
8×8m arena, 20 obstacles — 3D drone navigation
<video src="experiments_3d/videos/hybrid/3d_hybrid_standard.mp4" controls width="600"></video>

```bash
python -m scripts.eval --controller hybrid --episodes 5
```

#### 3D NFZ Avoidance
L-shaped hard wall + 25 obstacles — 3D NFZ avoidance
<video src="experiments_3d/videos/nfz/3d_nfz_hybrid.mp4" controls width="600"></video>

```bash
python -m scripts.eval_nfz --controller hybrid --episodes 5
```

#### GNN+LLM Combined
Combined GNN safety + LLM strategic bearing
<video src="experiments_3d/videos/gnn_llm/3d_gnn_llm.mp4" controls width="600"></video>

```bash
python -m scripts.eval_gnn_llm --episodes 3
```

#### Explicit LLM (JSON Commands)
LLM issues direct JSON action commands — fully interpretable
<video src="experiments_3d/videos/explicit_llm/3d_explicit_llm.mp4" controls width="600"></video>

```bash
python -m scripts.eval_explicit_llm --episodes 3
```

---

## 📊 Performance Summary

### 2D Experiments (Cat Avoidance)

| Framework | Avg Time | Collisions/Run | Status |
|-----------|----------|----------------|--------|
| APEX Original | 30s+ | Multiple | ❌ Failed (frozen) |
| Pulse Strategy | 9.8s | 1.0 | ✅ Works |
| **Hybrid GNN+LLM** | **4.5s** | **0.75** | ✅ **Best** |

### 3D Experiments (Drone Navigation)

| Framework | Avg Steps | Success Rate | Collision Rate |
|-----------|-----------|-------------|----------------|
| APEX Baseline | Timeout | ~40% | High |
| **Hybrid GNN+LLM** | ~150–250 | **~100%** | **~0%** |
| Explicit LLM | ~400–750 | ~67–100% | ~0% |

### Hybrid vs Explicit LLM (3D)

| Metric | Hybrid GNN+LLM | Explicit LLM |
|--------|----------------|--------------|
| Avg Steps to Goal | ~150–250 | ~400–750 |
| LLM Calls / Episode | 1 every 2s (strategic) | 1 every 0.5s (tactical) |
| API Cost | Low | **4× higher** |

---

## 🚀 Quick Start

### Prerequisites
```bash
# Create conda environment
conda create -n hybrid-nav python=3.11
conda activate hybrid-nav

# Install dependencies
pip install -r requirements.txt
```

### Set OpenAI API Key
```bash
# Linux/Mac
export OPENAI_API_KEY=sk-proj-...

# Windows
set OPENAI_API_KEY=sk-proj-...
```

### Run 2D Experiments
```bash
cd experiments_2d

# Mixed obstacle environment
python run_gnn_llm_hybrid.py

# NFZ experiment
python nfz_experiment.py --method HYBRID

# Medium NFZ (two zones)
python medium_nfz_experiment.py --method HYBRID

# Pac-Man maze
python pacman_maze_experiment.py --method HYBRID
```

### Run 3D Experiments
```bash
cd experiments_3d

# Standard mixed obstacle (3D)
python -m scripts.eval --controller hybrid --episodes 5

# NFZ (3D)
python -m scripts.eval_nfz --controller hybrid --episodes 5

# GNN+LLM combined
python -m scripts.eval_gnn_llm --episodes 3

# Explicit LLM (JSON commands)
python -m scripts.eval_explicit_llm --episodes 3
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

3. **Reactive NFZ Avoidance** — Shifted from proactive (avoid NFZ from far away, wasting time) to reactive (head for goal, deflect only when encountering NFZ boundary).

4. **Sample-Filter-Select Pipeline** — Separates safety (GNN: instant, per-frame) from strategy (LLM: occasional, every ~2s), achieving both responsiveness and intelligence.

5. **Multi-layer Safety Shield** — 5-step collision lookahead with emergency escape override, achieving 0% collision rate in 3D experiments.

---

## 📺 All Video Recordings

All experiment videos are embedded inline in the sections above. They are also available at:
- **2D**: [`experiments_2d/videos/`](experiments_2d/videos/)
- **3D**: [`experiments_3d/videos/`](experiments_3d/videos/)

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
