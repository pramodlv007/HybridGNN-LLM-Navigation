# 3D Drone Experiment Videos

## `hybrid/` — Standard Hybrid Controller

![Standard Hybrid](hybrid/3d_hybrid_standard.gif)

**Best Runs (fastest episodes):**

![Best Ep4](hybrid/3d_hybrid_best_ep4.gif)

![Best Ep1](hybrid/3d_hybrid_best_ep1.gif)

---

## `nfz/` — NFZ-Aware Hybrid Controller

![NFZ Hybrid](nfz/3d_nfz_hybrid.gif)

**Best Runs:**

![NFZ Best Ep4](nfz/3d_nfz_hybrid_best_ep4.gif)

![NFZ Best Ep1](nfz/3d_nfz_hybrid_best_ep1.gif)

---

## `gnn_llm/` — GNN+LLM Combined Controller

### Standard Environment
![GNN+LLM Standard](gnn_llm/3d_gnn_llm.gif)

**Best Runs:**

![GNN+LLM Best Ep4](gnn_llm/3d_gnn_llm_best_ep4.gif)

![GNN+LLM Best Ep1](gnn_llm/3d_gnn_llm_best_ep1.gif)

### NFZ Environment
![GNN+LLM NFZ](gnn_llm/3d_gnn_llm_nfz.gif)

**Best Runs:**

![GNN+LLM NFZ Best Ep3](gnn_llm/3d_gnn_llm_nfz_best_ep3.gif)

![GNN+LLM NFZ Best Ep1](gnn_llm/3d_gnn_llm_nfz_best_ep1.gif)

---

## `explicit_llm/` — Explicit LLM Controller

### Standard Environment
![Explicit LLM](explicit_llm/3d_explicit_llm.gif)

**Best Runs:**

![Explicit LLM Best Ep1](explicit_llm/3d_explicit_llm_best_ep1.gif)

![Explicit LLM Best Ep2](explicit_llm/3d_explicit_llm_best_ep2.gif)

### NFZ Environment
![Explicit LLM NFZ](explicit_llm/3d_explicit_llm_nfz.gif)

**Best Run:**

![Explicit LLM NFZ Best](explicit_llm/3d_explicit_llm_nfz_best_ep2.gif)

---

## Action Logs (Explicit LLM)

| File | Description |
|------|-------------|
| `3d_explicit_llm_action_log.txt` | Standard run action log |
| `3d_explicit_llm_best_ep1_action_log.txt` | Best ep1 action log |
| `3d_explicit_llm_best_ep2_action_log.txt` | Best ep2 action log |
| `3d_explicit_llm_nfz_action_log.txt` | NFZ run action log |
| `3d_explicit_llm_nfz_best_ep2_action_log.txt` | NFZ best run action log |

> Action logs contain timestamped LLM decisions:
> ```
> [00.05s] LLM ACTION SELECTED: turn heading right by 30 degrees (Type: rotate)
> [01.05s] LLM ACTION SELECTED: fly forward along current heading for 1s (Type: translate)
> ```
