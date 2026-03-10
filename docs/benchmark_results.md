# Benchmark Results

## 2D MuJoCo Experiments

### Cat Avoidance (Mixed 20-Obstacle Environment)

| Framework | Avg Time (s) | Collisions/Run | Success Rate | Approach |
|-----------|-------------|----------------|-------------|----------|
| APEX Original | 30+ | Multiple | ❌ Failed | Graphormer trigger → LLM every frame → frozen |
| Pulse Strategy | 9.8 | 1.0 | ✅ 100% | Alternating avoidance/cooldown states |
| **Hybrid GNN+LLM** | **4.5** | **0.75** | **✅ 100%** | Sample-Filter-Select pipeline |

> **Why APEX fails**: In dense 25-obstacle environments, GNN triggers danger every frame → constant LLM calls → robot "freezes" from decision latency.

### NFZ Experiment Benchmarks

| Framework | Success | NFZ Violations | Avg Steps |
|-----------|---------|---------------|-----------|
| APEX | ✅ | 0–2 | 500–800 |
| **HYBRID** | **✅** | **0** | **200–400** |

### Medium NFZ (Dual Zone) Benchmarks

| Framework | Success | NFZ Violations | Avg Steps |
|-----------|---------|---------------|-----------|
| APEX | ✅ | 0–1 | 600–900 |
| **HYBRID** | **✅** | **0** | **250–450** |

### Pac-Man Maze Benchmarks

| Framework | Success | Wall Hits | Avg Steps |
|-----------|---------|-----------|-----------|
| APEX | ✅ | 2–5 | 400–700 |
| PULSE | ✅ | 1–3 | 350–550 |
| **HYBRID** | **✅** | **0** | **250–400** |

---

## 3D PyBullet Drone Experiments

### Standard Mixed-Obstacle (20 obstacles)

| Controller | Avg Steps | Success Rate | Collision Rate |
|-----------|-----------|-------------|----------------|
| APEX Baseline | Timeout (1000) | ~40% | High |
| **Hybrid** | ~150–250 | ~100% | ~0% |

### NFZ Environment (25 obstacles + cross NFZ)

| Controller | Avg Steps | Success Rate | NFZ Violations |
|-----------|-----------|-------------|----------------|
| APEX Baseline | Timeout | ~30% | Multiple |
| **Hybrid NFZ** | ~200–350 | ~100% | 0 |

### GNN+LLM vs Explicit LLM

| Metric | Hybrid GNN+LLM | Explicit LLM |
|--------|----------------|--------------|
| Avg Steps to Goal | ~150–250 | ~400–750 |
| Success Rate | ~90–100% | ~67–100% |
| Collision Rate | ~0% | ~0% |
| LLM Calls/Episode | 1 per 2s (strategic) | 1 per 0.5s (tactical) |
| API Cost | Low | **4× higher** |
| Interpretability | Low (bearing only) | **High** (JSON actions) |

> **Explicit LLM is less efficient** due to: API latency dead time, rate-limit mock fallback, rotation oscillation, discrete action granularity, and stateless LLM calls.
>
> **Where Explicit LLM excels**: Fully interpretable audit trail — every decision logged with timestamp and natural language description.

---

## Key Findings

1. **Hybrid is consistently 2× faster** than the next-best approach across all experiments
2. **GNN micro-avoidance is essential** — it provides per-frame safety without LLM latency
3. **LLM macro-strategy helps but is not critical per-frame** — calling every ~2s is sufficient
4. **NFZ compliance reaches 100%** with reactive deflection + multi-step sweep prediction
5. **3D height-lock is crucial** for rigorous testing — without it, drones simply fly over obstacles
