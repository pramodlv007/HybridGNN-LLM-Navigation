"""Multi-trial benchmark for averaging out stochastic variance."""
import sys, os, random
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_hybrid_only import run_hybrid

total_std, total_nfz = 0, 0
N_TRIALS = 3

for trial in range(N_TRIALS):
    random.seed(trial * 100 + 7)
    np.random.seed(trial * 100 + 7)
    std = run_hybrid("standard", 20, 750)

    random.seed(trial * 100 + 13)
    np.random.seed(trial * 100 + 13)
    nfz = run_hybrid("nfz", 20, 750)

    s_pct = std["success_rate"] * 100
    n_pct = nfz["success_rate"] * 100
    print(f"Trial {trial+1}: Std={s_pct:.0f}% | NFZ={n_pct:.0f}%")
    total_std += std["success_rate"]
    total_nfz += nfz["success_rate"]

avg_s = total_std / N_TRIALS * 100
avg_n = total_nfz / N_TRIALS * 100
print(f"\nAVERAGE over {N_TRIALS} trials: Std={avg_s:.0f}% | NFZ={avg_n:.0f}%")
s_ok = "PASS" if avg_s > 80 else "FAIL"
n_ok = "PASS" if avg_n > 75 else "FAIL"
print(f"  Standard >80%: {s_ok}  |  NFZ >75%: {n_ok}")
