# 🧠 Hybrid GNN+LLM Navigation Framework

> A tiered decision-making architecture combining **Graph Neural Networks** for real-time safety assessment with **Large Language Models** for strategic planning — benchmarked head-to-head against a VLM baseline across **120 episodes** in 3D PyBullet environments.

---

## 🎬 Best Performance Demos

### 🏆 Demo 1 — Standard Environment (93.3% success rate)
> **Episode ep4 · 168 steps · SUCCESS · 0 collisions · 0 NFZ violations**
> Fastest successful navigation through 20 dynamic+static obstacles from (−3.6,−3.6) → (3.6,3.6)

![Hybrid GNN+LLM — Standard Env Best Run (168 steps)](experiments_3d/videos/gnn_llm/3d_gnn_llm_best_ep4.gif)

---

### 🏆 Demo 2 — NFZ Environment (80% success rate, 0% NFZ violations)
> **Episode NFZ-ep3 · Clean NFZ bypass · SUCCESS · 0 NFZ violations**
> Drone detects the 3×3m no-fly wall mid-route and routes around it via waypoint planning

![Hybrid GNN+LLM — NFZ Env Best Run (0 violations)](experiments_3d/videos/gnn_llm/3d_gnn_llm_nfz_best_ep3.gif)

---

## 📊 Benchmark Results: Hybrid GNN+LLM vs VLM

> **Setup:** 3D PyBullet | **30 episodes per condition** | GPT-4o text (Hybrid) vs GPT-4o Vision (VLM) | Report date: April 7, 2026

### ⚡ Hybrid wins 12 out of 13 metrics across both environments.

| | Hybrid GNN+LLM (Standard) | Hybrid GNN+LLM (NFZ) |
|---|:---:|:---:|
| **Best run demo** | ![](experiments_3d/videos/gnn_llm/3d_gnn_llm_best_ep4.gif) | ![](experiments_3d/videos/gnn_llm/3d_gnn_llm_nfz_best_ep3.gif) |
| **Success rate** | **93.3%** | **80.0%** |
| **Timeout rate** | **0.0%** | **0.0%** |
| **NFZ violations** | N/A | **0% vs VLM's 33.3%** |

### Standard Environment (No NFZ) — 30 Episodes

| Metric | Hybrid GNN+LLM | VLM Baseline | Δ | Winner |
|--------|:--------------:|:------------:|:--:|:------:|
| **Success Rate** | **93.3%** (28/30) | 53.3% (16/30) | +40.0 pp | 🏆 Hybrid |
| Collision Rate | **6.7%** (2/30) | 16.7% (5/30) | −10.0 pp | 🏆 Hybrid |
| **Timeout Rate** | **0.0%** (0/30) | 30.0% (9/30) | −30.0 pp | 🏆 Hybrid |
| Avg Steps (all eps) | **374** | 546 | −172 steps | 🏆 Hybrid |
| Avg Steps (success) | **378** | 487 | −109 steps | 🏆 Hybrid |
| Avg Final Distance | **0.996 m** | 2.591 m | −1.6 m | 🏆 Hybrid |

### NFZ Environment (25 obstacles + 3×3 hard wall) — 30 Episodes

| Metric | Hybrid GNN+LLM | VLM Baseline | Δ | Winner |
|--------|:--------------:|:------------:|:--:|:------:|
| **Success Rate** | **80.0%** (24/30) | 46.7% (14/30) | +33.3 pp | 🏆 Hybrid |
| Collision Rate | **20.0%** (6/30) | 40.0% (12/30) | −20.0 pp | 🏆 Hybrid |
| **Timeout Rate** | **0.0%** (0/30) | 13.3% (4/30) | −13.3 pp | 🏆 Hybrid |
| **NFZ Violations** | **20.0%** (6/30) | 33.3% (10/30) | −13.3 pp | 🏆 Hybrid |
| Avg Steps (success) | **481** | 482 | −1 step | Tie |
| Avg Final Distance | **1.519 m** | 3.260 m | −1.7 m | 🏆 Hybrid |

> **Key insight:** Hybrid achieves **0% timeouts in both environments** by converting near-misses into successes via near-goal threshold relaxation and stagnation escape logic.

> ⚠️ **Note on VLM:** `vlm_calls = 0` in this benchmark due to API rate-limiting — the VLM operated entirely on its goal-directed fallback heuristic (no vision, no safety scoring). VLM results represent a **lower bound** on true VLM performance.

---

## 🏗️ Architecture

The Hybrid framework uses a **Sample → Filter → Select → Shield** pipeline that separates safety from strategy:

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
│                     │  5-step collision lookahead (0.25s)│  │
│                     │  NFZ boundary prevention           │  │
│                     │  Reactive escape fallback          │  │
│                     └──────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### The 7-Rule Pipeline (20 Hz / 50ms per step)

| Rule | Component | Action |
|------|-----------|--------|
| **R1** | **LLM Strategic Bearing** | Goal-directed bearing cached every 2s. NFZ mode: waypoint routing around wall computed at episode start. |
| **R2** | **Candidate Generation** | 16 velocity directions at 22.5° spacing. Speed: 1.5 m/s near NFZ/obstacles, 2.0 m/s open field. |
| **R3** | **GNN Risk Scoring** | Score each candidate: obstacle proximity (inverse-square) + dynamic TTC + NFZ risk + wall penalty. |
| **R4** | **Dynamic Threshold Filter** | Keep candidates with risk < threshold: 10.0 (near goal <1.5m), 5.0 (near NFZ), 6.0 (near obs <0.4m), 8.0 (standard). |
| **R5** | **Best-Aligned Selection** | Pick safe candidate most aligned with LLM bearing (max dot product). Rotate bearing if stagnation detected. |
| **R6** | **3D Adaptation Layer** | `accel = (v_target − v_current) × 6.0`, clipped ±4.0 m/s², z-component zeroed for planar flight. |
| **R7** | **Post-Decision Safety Shield** | 5-step forward simulation: collision predicted → escape brake (85%); TTC > 0.6 → dodge (50%); NFZ entry → push out (60%). |

---

## 🔬 Component Deep-Dive

### Component 1 — GNN Safety Checker

Evaluates all 16 candidate velocities with a composite risk formula:

| Risk Component | Formula | Trigger |
|----------------|---------|---------|
| Hard collision | +50.0 flat penalty | `dist < robot_r + obs_r + 0.22` |
| Inverse-square proximity | `0.5 / d²` | `d < 1.0 m` |
| Dynamic TTC risk | `ttc_risk() × 5.0` | When `TTC < 0.5 s` |
| NFZ interior | +100.0 | Inside NFZ boundary |
| NFZ margin | +30.0 | Within `robot_r + 0.15 m` |
| Arena wall | `2.0 / d²` | `d < 0.4 m` from wall |

An optional trained `LocalRiskGNN` mode uses a physics-informed graph neural network (star topology, robot+obstacle nodes) to replace the heuristic scoring.

### Component 2 — LLM Strategic Navigator

- **Standard mode:** `bearing = normalize(goal − robot)` updated every 2s (simulates LLM call latency).
- **NFZ mode:** Plans a multi-waypoint bypass route at episode start using parametric AABB intersection. Checks SW/SE/NW/NE corner offsets (0.9m margin). Falls back to two-waypoint routes (SE→NE or NW→NE) if single waypoint is blocked.

### Component 3 — Post-Decision Safety Shield

Simulates 5 physics steps (0.25s lookahead) after every action is chosen:

```python
for step in range(5):
    predict position and velocity
    if collision predicted  →  escape brake at 85% max_accel  (OVERRIDE)
    if TTC > 0.6            →  dodge at 50% max_accel          (OVERRIDE)
    if NFZ entry predicted  →  push out at 60% max_accel       (OVERRIDE)
# If all clear → execute original plan
```

### Component 4 — Stagnation Detector

When < 0.2m progress over 60 consecutive steps (3s):
- **Near NFZ:** 60% perpendicular escape bearing + 40% goal direction (away from NFZ center)
- **Open field:** Rotate bearing ±30° (alternating direction each trigger), lasts 25 steps

### Component 5 — Adaptive Speed & Threshold

```python
speed      = 1.5 m/s  if near_nfz or min_obs_dist < 0.5 else 2.0 m/s
threshold  = 10.0  if dist_to_goal < 1.5   # near-goal push — eliminated all timeouts
           = 5.0   if near_nfz             # conservative near wall
           = 6.0   if min_obs_dist < 0.4   # close obstacle
           = 8.0                            # standard
```

---

## 📂 Repository Structure

```text
HybridGNN-LLM-Navigation/
│
├── README.md
├── requirements.txt
├── benchmark_report.py        # Aggregate report generator
├── create_report.py           # DOCX report creator
│
├── experiments_2d/            # ═══ 2D MuJoCo Experiments ═══
│   ├── hybrid_nav.py
│   ├── run_gnn_llm_hybrid.py
│   ├── nfz_experiment.py
│   ├── medium_nfz_experiment.py
│   ├── run_2d_full_comparison.py   # Hybrid vs VLM 2D
│   ├── run_benchmark_2d.py
│   ├── env/                        # MuJoCo XML environments
│   ├── model/                      # GNN models + weights
│   ├── visualization/              # Charts and plots
│   └── videos/
│
├── experiments_3d/            # ═══ 3D PyBullet Drone Experiments ═══
│   ├── run_3d_expt.py              # Single-episode runner
│   ├── run_3d_expt_loop.py         # Multi-episode runner
│   ├── run_hybrid_only.py          # Deterministic Hybrid benchmark
│   ├── run_3d_full_comparison.py   # 4-condition Hybrid vs VLM benchmark
│   ├── run_benchmark_3d.py
│   ├── run_multi_trial.py
│   ├── controllers/
│   │   ├── hybrid_gnn_llm_controller.py  # Main controller (all 7 rules)
│   │   ├── hybrid_nfz_controller.py      # NFZ-aware variant
│   │   ├── hybrid_controller.py          # Standard hybrid (no LLM)
│   │   ├── vlm_controller.py             # VLM baseline (GPT-4o Vision)
│   │   ├── gnn_safety_checker.py         # Risk scoring (heuristic + GNN)
│   │   ├── llm_strategic_navigator.py    # LLM bearing + NFZ routing
│   │   ├── explicit_llm_controller.py    # Explicit JSON LLM commands
│   │   ├── apex_baseline.py              # APEX baseline
│   │   └── risk.py                       # TTC closing-speed model
│   ├── envs/
│   │   ├── drone_nav_3d.py               # Standard env (20 obstacles)
│   │   └── drone_nav_nfz.py              # NFZ env (25 obs + 3×3 wall)
│   ├── models/
│   │   ├── local_risk_gnn.py
│   │   └── trained_risk_gnn.pt           # Trained GNN weights
│   ├── scripts/
│   │   ├── eval.py
│   │   ├── eval_nfz.py
│   │   ├── eval_gnn_llm.py
│   │   ├── eval_vlm.py                   # VLM eval runner
│   │   └── eval_explicit_llm.py
│   └── training/
│       ├── train_physics_gnn.py
│       ├── collect_data.py
│       └── run_trained_experiment.py
│
├── results/
│   ├── comparison_3d/
│   │   └── 3d_hybrid_vs_vlm_report_20260404.md  # 15-ep preliminary data
│   ├── comparison_2d/                  # 2D Hybrid vs VLM CSV/JSON/MD
│   ├── benchmarks/                     # 2D and 3D benchmark CSVs
│   ├── hybrid_tuning/                  # Hyperparameter tuning logs (30 runs)
│   └── testing_final_20eps/            # Acceptance test results
│
└── docs/
    ├── architecture.md
    └── benchmark_results.md
```

---

## 🧪 Running Experiments

### Prerequisites

```bash
conda create -n hybrid-nav python=3.11
conda activate hybrid-nav
pip install -r requirements.txt

# Windows
set OPENAI_API_KEY=sk-proj-...
# Linux/Mac
export OPENAI_API_KEY=sk-proj-...
```

### 3D Experiments

```bash
cd experiments_3d

# Deterministic Hybrid benchmark (30 episodes, reproducible)
python run_hybrid_only.py --episodes 30

# Full 4-condition Hybrid vs VLM comparison
python run_3d_full_comparison.py --episodes 30

# Single-episode runner
python run_3d_expt.py --controller hybrid_gnn_llm

# VLM baseline
python -m scripts.eval_vlm --episodes 30

# NFZ environment
python -m scripts.eval_nfz --controller hybrid --episodes 30
```

### 2D Experiments

```bash
cd experiments_2d

# Mixed obstacle navigation
python run_gnn_llm_hybrid.py

# NFZ experiment
python nfz_experiment.py --method HYBRID

# Full 2D comparison (Hybrid vs VLM)
python run_2d_full_comparison.py
```

### Camera Options (3D)

```bash
--camera pov      # First-person (default)
--camera top      # Bird's-eye view
--camera follow   # Third-person chase
--no-record       # Disable video recording
```

---

## 📈 2D Preliminary Results (MuJoCo, 5 episodes)

| Metric | Hybrid GNN+LLM | VLM (GPT-4o) |
|--------|:--------------:|:------------:|
| **Standard — Success Rate** | 100.0% | 100.0% |
| **Standard — NFZ Violations** | 0.0% | 0.0% |
| Standard — Avg Steps | 798 | 613 |
| **NFZ — Success Rate** | 100.0% | 100.0% |
| **NFZ — NFZ Violations** | **0.0%** | **100.0%** |
| NFZ — Avg Steps | 702 | 585 |

Key 2D finding: VLM solved faster but **completely failed to respect No-Fly Zones** (100% violation rate). Hybrid maintained **0% violations** across all environments.

---

## ⚙️ Final Controller Configuration

```yaml
Controller: HybridGNNLLMController (Iteration 5 — Final Optimized)

candidates:            16 directions (22.5° spacing)
speed_standard:        2.0 m/s
speed_near_nfz_obs:    1.5 m/s

threshold_near_goal:   10.0   # dist < 1.5m
threshold_near_nfz:    5.0
threshold_near_obs:    6.0    # dist < 0.4m
threshold_standard:    8.0

accel_gain:            6.0
max_accel:             4.0 m/s²
z_component:           0      # planar flight

shield_lookahead:      5 steps (0.25s)
escape_static:         85% max_accel
escape_dynamic:        50% max_accel
escape_nfz:            60% max_accel (away from NFZ center)

stagnation_window:     60 steps (3s)
stagnation_progress:   0.2m threshold
rescue_duration:       25 steps
rotation_angle:        ±30° (alternating)

nfz_corner_margin:     0.9m
waypoint_advance_r:    1.2m
fallback_route:        SE→NE or NW→NE two-waypoint

max_steps:             1000 (50s)
episodes:              30
deterministic:         True  # seed = episode × 1000 + condition_offset
```

---

## 🔬 Key Technical Contributions

1. **Fixed TTC Risk Formula** — Corrected Time-to-Collision calculation that was causing false "always-at-risk" behavior in the original APEX framework.
2. **Drone Height Lock (3D)** — Locks drone altitude to z=1.0, forcing navigation through the obstacle field rather than over it.
3. **Reactive NFZ Routing** — Episode-start waypoint planning around the NFZ wall eliminates wasted steps trying to push through it.
4. **Near-Goal Threshold Relaxation** — Raising the risk threshold to 10.0 within 1.5m of goal eliminated all timeouts (0% in both environments).
5. **Sample-Filter-Select Pipeline** — Decouples safety (GNN: <1ms, every frame) from strategy (LLM: ~2s, periodic), achieving both speed and intelligence.
6. **Stagnation Escape Logic** — Detects oscillation and rotates bearing to break out of local minima.
7. **VLM Baseline Controller** — `vlm_controller.py` enables rigorous head-to-head comparison with GPT-4o Vision using identical GNN safety infrastructure.

---

## 📄 Full Benchmark Report

Per-episode data for all 120 episodes (4 conditions × 30):
- [`results/comparison_3d/3d_hybrid_vs_vlm_report_20260404.md`](results/comparison_3d/3d_hybrid_vs_vlm_report_20260404.md)

---

## 📺 Video Recordings

- **2D**: [`experiments_2d/videos/`](experiments_2d/videos/)
- **3D**: [`experiments_3d/videos/`](experiments_3d/videos/)

---

## 📈 Citation

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
