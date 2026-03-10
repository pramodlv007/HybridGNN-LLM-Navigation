# 3D Drone Experiment Videos

Curated recordings organized by controller variant:

## `hybrid/` — Standard Hybrid Controller
| File | Description |
|------|-------------|
| `3d_hybrid_standard.mp4` | Standard 3D drone navigating 20-obstacle field |

## `nfz/` — NFZ-Aware Hybrid Controller
| File | Description |
|------|-------------|
| `3d_nfz_hybrid.mp4` | Drone avoiding cross-shaped NFZ + 25 obstacles |

## `gnn_llm/` — GNN+LLM Combined Controller
| File | Description |
|------|-------------|
| `3d_gnn_llm.mp4` | Combined GNN safety + LLM strategy (standard env) |
| `3d_gnn_llm_nfz.mp4` | Combined GNN+LLM in NFZ environment |

## `explicit_llm/` — Explicit LLM Controller
| File | Description |
|------|-------------|
| `3d_explicit_llm.mp4` | LLM issuing JSON commands (standard env) |
| `3d_explicit_llm_action_log.txt` | Timestamped action decisions |
| `3d_explicit_llm_nfz.mp4` | LLM with NFZ environment |
| `3d_explicit_llm_nfz_action_log.txt` | NFZ action log |

> **Action logs** contain timestamped LLM decisions like:
> ```
> [00.05s] LLM ACTION SELECTED: turn heading right by 30 degrees (Type: rotate)
> [01.05s] LLM ACTION SELECTED: fly forward along current heading for 1s (Type: translate)
> ```
