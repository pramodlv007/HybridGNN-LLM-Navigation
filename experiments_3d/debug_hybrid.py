
import numpy as np
from controllers.gnn_safety_checker import GNNSafetyChecker
from controllers.llm_strategic_navigator import StrategicNavigator
from controllers.hybrid_gnn_llm_controller import HybridGNNLLMController

def debug_ctrl():
    ctrl = HybridGNNLLMController()
    obs = {
        "pos": np.array([0, 0, 1], dtype=np.float64),
        "vel": np.array([0, 0, 0], dtype=np.float64),
        "goal": np.array([8, 0, 1], dtype=np.float64),
        "obstacles": [{"pos": np.array([1, 0, 1]), "vel": np.array([0, 0, 0]), "radius": 0.3, "is_dynamic": False}],
        "nfz_blocks": [{"min": [3, -1], "max": [5, 1]}],
        "steps": 0
    }
    
    action = ctrl.act(obs)
    print(f"Debug Action: {action}")
    
    # Check candidates
    bearing = ctrl.llm.get_strategic_bearing(obs["pos"], obs["goal"], obs["obstacles"], obs["nfz_blocks"], 0)
    print(f"Bearing: {bearing}")
    
    candidates = ctrl.gnn.generate_velocity_candidates(speed=2.5)
    ranked = ctrl.gnn.rank_candidates(candidates, obs["pos"], obs["vel"], obs["obstacles"], obs["nfz_blocks"])
    
    for v, r in ranked:
        sim = np.dot(v, bearing)
        print(f"Vel: {v[:2]}, Risk: {r:.2f}, Alignment: {sim:.2f}")

if __name__ == "__main__":
    debug_ctrl()
