# 3D PyBullet Drone Experiments — Hybrid GNN+LLM Navigation

> **Final Benchmark Results** | April 2026 | 30 episodes (Hybrid) • 20 episodes (VLM) | Deterministic seeding

---

## 🏆 Benchmark Results — Hybrid GNN-LLM vs VLM

### Standard Environment (No NFZ)

| Metric | **Hybrid GNN-LLM** | VLM Baseline | Δ |
|--------|--------------------|--------------|---|
| **Success Rate** | **93.3%** (28/30) | 53.3% (16/30) | **+40.0%** ✅ |
| Collision Rate | 6.7% | 16.7% | -10.0% |
| **Timeout Rate** | **0.0%** | 30.0% | **-30.0%** ✅ |
| Avg Steps (success) | 378 | 487 | -109 steps |
| Avg Final Distance | 0.49m | 2.59m | -2.1m |

### NFZ Environment (25 obstacles + 3×3 hard wall)

| Metric | **Hybrid GNN-LLM** | VLM Baseline | Δ |
|--------|--------------------|--------------|---|
| **Success Rate** | **80.0%** (8/10) | 46.7% | **+33.3%** ✅ |
| Collision Rate | 20.0% | 40.0% | -20.0% |
| **Timeout Rate** | **0.0%** | 13.3% | **-13.3%** ✅ |
| NFZ Violations | 10.0% | 33.3% | -23.3% |
| Avg Steps (success) | 402 | 482 | -80 steps |

---

## 🎬 Demo Videos

### ✅ Hybrid GNN-LLM — Standard (Best: Ep 9, **137 steps**)

https://github.com/pramodlv007/HybridGNN-LLM-Navigation/raw/3d-benchmark-results/experiments_3d/demos/hybrid_standard_best_ep9_137steps.mp4

### ✅ Hybrid GNN-LLM — NFZ (Best: Ep 4, **274 steps**)

https://github.com/pramodlv007/HybridGNN-LLM-Navigation/raw/3d-benchmark-results/experiments_3d/demos/hybrid_nfz_best_ep4_274steps.mp4

### 🔵 VLM Baseline — Standard

https://github.com/pramodlv007/HybridGNN-LLM-Navigation/raw/3d-benchmark-results/experiments_3d/demos/vlm_standard_best_ep5.mp4

### 🔵 VLM Baseline — NFZ

https://github.com/pramodlv007/HybridGNN-LLM-Navigation/raw/3d-benchmark-results/experiments_3d/demos/vlm_nfz_best_ep15.mp4

---

## ⚙️ Key Optimizations Made (April 2026)

### 1. Near-Goal Threshold Relaxation
```python
# Before: Fixed threshold = 8.0 always
# After:  Dynamic threshold based on context
if dist_to_goal < 1.5:
    threshold = 10.0   # Push through final approach → eliminates 100% of timeouts
elif near_nfz:
    threshold = 5.0    # Conservative near wall
elif min_obs_dist < 0.4:
    threshold = 6.0    # Cautious near obstacles
else:
    threshold = 8.0    # Standard
```
**Impact:** Timeout rate dropped from **30% → 0%** in standard, **13% → 0%** in NFZ.

### 2. Universal Stagnation Escape
```python
# Before: Stagnation only detected in NFZ mode
# After:  Active in ALL environments
if progress_last_60_steps < 0.2m:
    stagnation_mode = True   # Rotate bearing ±30° to escape local minima
```
**Impact:** Eliminated oscillation loops that previously caused indefinite stagnation near clustered obstacles.

### 3. Velocity-Driven Yaw + Acceleration Banking (NFZ env)
```python
# Before: target_yaw = action_acc[3]  →  static, caused no banking
# After:  velocity-derived yaw + acceleration decomposition (same as std env)
yaw = atan2(drone_vel_y, drone_vel_x)            # from velocity
pitch = -clip(forward_accel * 0.06, ±20°)        # lean forward
roll  =  clip(-lateral_accel * 0.08, ±25°)       # bank into turns
```
**Impact:** NFZ drone now has identical realistic banking/tilting dynamics as standard env.

### 4. NFZ Waypoint Pre-Planning
```python
# LLM navigator plans bypass route at episode start
# Tries SE/NE/NW corners with 0.9m margin
# Falls back to 2-waypoint route if needed
waypoints = plan_nfz_bypass(start, goal, nfz_blocks)
```
**Impact:** Drone routes *around* the NFZ wall instead of trying to push through it.

### 5. Deterministic Per-Episode Seeding
```python
seed = episode_index * 1000 + condition_offset
random.seed(seed); np.random.seed(seed)
```
**Impact:** 100% reproducible results — same episode always produces identical trajectory.

### 6. Invisible Arena Walls (Visual)
```python
wall_color = [0.6, 0.6, 0.6, 0.0]   # alpha=0 → fully transparent
```
**Impact:** Cleaner camera view without grey walls blocking the drone during NFZ recording.

---

## 🏗️ Architecture — Hybrid GNN-LLM Pipeline

```
Observation (pos, vel, goal, obstacles, nfz_blocks)
          │
          ▼
   ┌─────────────────┐
   │  LLM Navigator   │  → Strategic bearing (cached 2s)
   │  (R1 Bearing)    │    NFZ mode: waypoint routing
   └─────────────────┘
          │
          ▼
   ┌─────────────────┐
   │ Candidate Gen   │  → 16 velocity vectors, adaptive speed
   │  (R2 Candidates)│    1.5 m/s near NFZ, 2.0 m/s open field
   └─────────────────┘
          │
          ▼
   ┌─────────────────┐
   │ GNN Safety Check│  → Risk score per candidate
   │  (R3 Scoring)   │    obstacle + TTC + NFZ + wall risk
   └─────────────────┘
          │
          ▼
   ┌─────────────────┐
   │ Dynamic Filter  │  → Keep risk < dynamic threshold
   │  (R4 Filter)    │    10.0 near goal / 5.0 near NFZ / 8.0 std
   └─────────────────┘
          │
          ▼
   ┌─────────────────┐
   │ Bearing Align   │  → argmax(dot(candidate, LLM_bearing))
   │  (R5 Select)    │    + stagnation escape if stuck
   └─────────────────┘
          │
          ▼
   ┌─────────────────┐
   │ 3D Adaptation   │  → accel = (v_target - v_current) × 6.0
   │  (R6 Convert)   │    clipped ±4.0 m/s², z=0
   └─────────────────┘
          │
          ▼
   ┌─────────────────┐
   │ Safety Shield   │  → 5-step forward simulation
   │  (R7 Override)  │    collision/NFZ predicted → escape
   └─────────────────┘
          │
          ▼
      Execute Action
```

---

## 🚀 Quick Start

```bash
# Run Hybrid GNN-LLM benchmark (30 episodes, standard + NFZ, with video)
cd experiments_3d
python run_3d_full_comparison.py --episodes 30 --max-steps 1000

# Run Hybrid NFZ only with HUD overlay showing LLM bearing
python run_hybrid_nfz_video.py

# Run standard hybrid benchmark
python run_hybrid_only.py --episodes 30 --max-steps 1000
```

---

## 📁 Key Files

### Controllers (`controllers/`)
| File | Role |
|------|------|
| `hybrid_gnn_llm_controller.py` | **Main controller** — 7-rule pipeline, adaptive threshold, stagnation escape, safety shield |
| `gnn_safety_checker.py` | GNN risk scoring (obstacle proximity + TTC + NFZ + wall) |
| `llm_strategic_navigator.py` | LLM bearing provider (direct + NFZ waypoint routing) |
| `risk.py` | TTC and proximity risk computations |

### Environments (`envs/`)
| File | Description |
|------|------------|
| `drone_nav_3d.py` | Standard: 8×8m arena, 20 obstacles (10 dynamic + 10 static) |
| `drone_nav_nfz.py` | NFZ: 25 obstacles + 3×3m hard wall, invisible border walls, velocity-derived banking |

### Runners
| File | Description |
|------|------------|
| `run_3d_full_comparison.py` | Full 4-condition benchmark (Hybrid + VLM × Standard + NFZ) |
| `run_hybrid_nfz_video.py` | Hybrid NFZ with LLM bearing HUD overlay |
| `run_hybrid_only.py` | Hybrid standard only |

---

## 🏗️ Physical Drone Model

Compound PyBullet model with realistic dynamics:

| Part | Visual | Purpose |
|------|--------|---------|
| Central fuselage | Dark gray flat box | Main body |
| Front arm | **RED** cylinder | Shows heading direction |
| Back/Side arms | Gray / **GREEN** | Orientation reference |
| 4 Rotor discs | Dark flat cylinders | Visual realism |

**Dynamics**: Pitches forward on acceleration, banks (rolls) into turns, levels flat when hovering.  
Banking is computed via acceleration decomposition: forward component → pitch, lateral component → roll.

---

## 📺 Camera Options

```bash
--camera follow    # Third-person chase (default, best for standard env)
--camera top       # Bird's-eye view (best for NFZ overview)
--camera chase     # Behind-drone smoothed camera (exponential moving avg 0.05)
--camera pov       # First-person from drone nose
--camera front     # Front-facing fixed view
```

---

## 📊 Environment Specifications

| Parameter | Standard | NFZ |
|-----------|----------|-----|
| Arena | 8×8×4m | 8×8×4m |
| Static obstacles | 10 | 15 |
| Dynamic obstacles | 10 | 10 |
| NFZ region | — | 3×3m center rectangle |
| NFZ enforcement | — | Hard wall (velocity zeroed) |
| Start → Goal | (-3.6,-3.6) → (3.6,3.6) | Same |
| Diagonal distance | 10.18m | 10.18m |
| Max steps | 1000 | 1000 |
| Timestep | 0.05s | 0.05s |
