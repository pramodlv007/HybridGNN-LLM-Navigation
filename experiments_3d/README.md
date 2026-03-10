# 3D PyBullet Drone Experiments — Hybrid GNN+LLM Navigation

## Overview

3D autonomous drone navigation experiments in PyBullet. A physical drone model navigates through dense obstacle fields using the **Hybrid GNN+LLM** framework with multiple controller variants.

---

## 🚀 Controller Variants

### 1. Hybrid Controller (`hybrid_controller.py`)
Standard 3-tier decision architecture:
- **Safety Shield (Tier 1)** — Filters N candidate accelerations via short-term simulation; rejects collisions and NFZ violations using TTC and proximity risk
- **Strategic Rollout (Tier 2)** — Re-scores safe candidates via longer rollouts; evaluates goal progress and obstacle penalties
- **Reactive Fallback (Tier 3)** — Emergency braking/escape when no safe candidates exist

### 2. Hybrid NFZ Controller (`hybrid_nfz_controller.py`)
Enhanced for NFZ environments:
- All Tier 1–3 logic from above
- NFZ boundary proximity risk added to GNN scoring
- Multi-step NFZ sweep prediction (simulates future path for NFZ entry)
- Reactive NFZ deflection (heads straight, deflects only at boundary)

### 3. GNN+LLM Controller (`hybrid_gnn_llm_controller.py`)
Combined pipeline where LLM provides strategic bearing and GNN handles micro-avoidance:
- LLM called every ~2s for strategic direction
- GNN scores 16 candidates every frame
- 5-step safety shield with collision lookahead

### 4. Explicit LLM Controller (`explicit_llm_controller.py`)
LLM issues direct JSON action commands (turn, fly, hover):
- GPT-4o receives obstacle positions and returns explicit actions
- Actions are visual only; actual navigation still uses goal bearing
- Higher API cost but fully interpretable and auditable
- See action logs in `videos/explicit_llm/`

---

## 📁 File Reference

### Controllers (`controllers/`)
| File | Role |
|------|------|
| `hybrid_controller.py` | Standard 3D hybrid (Safety Shield + Rollout + Fallback) |
| `hybrid_nfz_controller.py` | NFZ-aware hybrid with boundary detection |
| `hybrid_gnn_llm_controller.py` | GNN safety + LLM strategic bearing |
| `explicit_llm_controller.py` | LLM issues explicit JSON actions |
| `gnn_safety_checker.py` | GNN candidate ranking (obstacle/wall/NFZ/TTC risk) |
| `llm_strategic_navigator.py` | LLM strategic bearing provider |
| `apex_baseline.py` | APEX baseline controller (for comparison) |
| `risk.py` | Optimized TTC and proximity risk calculations |

### Environments (`envs/`)
| File | Description |
|------|------------|
| `drone_nav_3d.py` | Standard 3D: 8×8m arena, 20 obstacles (10 dynamic + 10 static) |
| `drone_nav_nfz.py` | NFZ variant: 25 obstacles + cross-shaped no-fly zone + physical drone model |
| `drone_available_move.json` | Drone action space for LLM integration |

### Models (`models/`)
| File | Description |
|------|------------|
| `local_risk_gnn.py` | LocalRiskGNN for 3D obstacle risk scoring |

### Scripts (`scripts/`)
| File | Description |
|------|------------|
| `eval.py` | Standard experiment runner |
| `eval_nfz.py` | NFZ experiment runner |
| `eval_gnn_llm.py` | GNN+LLM experiment runner |
| `eval_explicit_llm.py` | Explicit LLM experiment runner |

---

## 🧪 Experiments

### 1. Standard Mixed-Obstacle
- **Arena**: 8×8m, 20 obstacles (10 dynamic, 10 static)
- **Start**: `(-3.6, -3.6, 1.0)` → **Goal**: `(3.6, 3.6, 1.0)`
- **Height-locked** at z=1.0 to force navigation through obstacles

<video src="videos/hybrid/3d_hybrid_standard.mp4" controls width="600"></video>

```bash
python -m scripts.eval --controller hybrid --episodes 10
```

### 2. No-Fly Zone (NFZ)
- **Arena**: 25 obstacles + L-shaped / cross-shaped hard wall NFZ
- **Challenge**: Detect NFZ reactively and reroute

<video src="videos/nfz/3d_nfz_hybrid.mp4" controls width="600"></video>

```bash
python -m scripts.eval_nfz --controller hybrid --episodes 5
```

### 3. GNN+LLM Combined
- **Same environments** as above but with LLM strategic bearing

<video src="videos/gnn_llm/3d_gnn_llm.mp4" controls width="600"></video>

**With NFZ:**

<video src="videos/gnn_llm/3d_gnn_llm_nfz.mp4" controls width="600"></video>

```bash
python -m scripts.eval_gnn_llm --episodes 3
```

### 4. Explicit LLM
- **Same environments** but LLM issues explicit JSON commands
- Produces both video and action logs

<video src="videos/explicit_llm/3d_explicit_llm.mp4" controls width="600"></video>

**With NFZ:**

<video src="videos/explicit_llm/3d_explicit_llm_nfz.mp4" controls width="600"></video>

```bash
python -m scripts.eval_explicit_llm --episodes 3
# Without NFZ:
python -m scripts.eval_explicit_llm --no-nfz --episodes 3
```

---

## 📺 Camera & Recording Options

```bash
--camera pov       # First-person from drone (default)
--camera top       # Bird's-eye view (best for seeing NFZ)
--camera follow    # Third-person chase
--camera chase     # Behind-drone smoothed camera
--camera front     # Front-facing view
--no-record        # Disable video for faster benchmarking
```

---

## 🏗️ Physical Drone Model

The drone is a compound PyBullet model (not a simple sphere):

| Part | Visual | Purpose |
|------|--------|---------|
| Central fuselage | Dark gray flat box | Main body |
| Front arm | **RED** cylinder | Shows heading |
| Back arm | Gray cylinder | Orientation |
| Left/Right arms | **GREEN** cylinders | Orientation |
| 4 Rotor discs | Dark flat cylinders | Visual realism |

**Physical behaviors**: Pitches forward when flying, rolls during turns, levels flat when hovering. Chase camera uses exponential moving average (factor 0.05) for smooth following.

---

## 📊 Performance

| Controller | Avg Steps | Success Rate | Collision Rate |
|-----------|-----------|-------------|----------------|
| APEX Baseline | Timeout | ~40% | High |
| **Hybrid** | ~150–250 | ~100% | ~0% |
| **GNN+LLM** | ~150–250 | ~90–100% | ~0% |
| Explicit LLM | ~400–750 | ~67–100% | ~0% |
