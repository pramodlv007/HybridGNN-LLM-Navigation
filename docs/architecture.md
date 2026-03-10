# Framework Architecture

## Hybrid GNN+LLM Decision Pipeline

The core innovation is separating **safety** (fast, per-frame GNN) from **strategy** (slow, periodic LLM):

```
                    ┌──────────────────────┐
                    │    Environment       │
                    │  (MuJoCo / PyBullet) │
                    └──────────┬───────────┘
                               │ observations
                               ▼
                    ┌──────────────────────┐
                    │   SAMPLE GENERATOR   │
                    │  16–25 candidate     │
                    │  velocity vectors    │
                    └──────────┬───────────┘
                               │ candidates[]
                               ▼
            ┌──────────────────────────────────────┐
            │         GNN SAFETY CHECKER           │
            │                                      │
            │  For each candidate:                 │
            │  ├─ Obstacle proximity risk           │
            │  ├─ Time-to-Collision (TTC)           │
            │  ├─ Wall / boundary penalty           │
            │  ├─ NFZ proximity risk (if applicable)│
            │  └─ Combined risk score → safe/unsafe │
            │                                      │
            │  Output: safe_candidates[]            │
            └──────────────────┬───────────────────┘
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │       LLM STRATEGIC NAVIGATOR        │
            │                                      │
            │  Called every ~2 seconds:             │
            │  ├─ Receives: robot pos, goal pos,   │
            │  │   obstacle summary                 │
            │  ├─ Returns: strategic bearing vector │
            │  └─ Cached between calls              │
            │                                      │
            │  Selection: safe candidate most       │
            │  aligned with strategic bearing       │
            └──────────────────┬───────────────────┘
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │          SAFETY SHIELD               │
            │                                      │
            │  5-step future simulation:            │
            │  ├─ If collision predicted → OVERRIDE │
            │  │   with escape acceleration         │
            │  └─ Otherwise → EXECUTE selected move │
            └──────────────────┬───────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   EXECUTE ACTION     │
                    │  [ax, ay, az, yaw]   │
                    └──────────────────────┘
```

---

## 2D vs 3D Implementation Differences

| Aspect | 2D (MuJoCo) | 3D (PyBullet) |
|--------|-------------|---------------|
| **Physics** | MuJoCo top-down 2D | PyBullet full 3D |
| **Robot** | Circle/sphere | Compound drone (arms, rotors) |
| **Height** | Fixed (ground plane) | Locked at z=1.0 |
| **GNN** | LocalRiskGNN (2D features) | LocalRiskGNN (adapted 3D) |
| **Candidates** | 16 directions × 1 speed | 16 dir × 2 speeds + stop |
| **NFZ** | Rectangle regions | Cross-shaped hard walls |
| **Camera** | Top-down MuJoCo viewer | POV / chase / bird's-eye |

---

## GNN Risk Scoring Details

The `GNNSafetyChecker` computes a composite risk score for each candidate velocity:

```
risk(candidate) = w₁ · obstacle_proximity
                + w₂ · inverse_TTC
                + w₃ · wall_boundary_penalty
                + w₄ · nfz_proximity_risk
```

Where:
- **obstacle_proximity** = Σ (1/d²) for obstacles within influence radius
- **inverse_TTC** = 1/TTC for dynamic obstacles on collision course
- **wall_boundary** = exponential penalty as robot nears arena walls
- **nfz_proximity** = penalty for candidates that move toward NFZ boundaries

A candidate is **safe** if `risk < threshold` (default: 8.0).
