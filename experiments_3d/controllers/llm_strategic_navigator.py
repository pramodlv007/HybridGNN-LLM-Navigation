"""
Strategic Navigator (LLM Proxy) for 3D Navigation
===================================================
Combines the original simple direct-to-goal strategy (proven 80% in standard env)
with NFZ-aware waypoint routing (only activates when NFZ blocks are present).
"""

import numpy as np


class StrategicNavigator:
    """
    Simulates LLM strategic reasoning.
    - No NFZ: direct to goal (original behavior)
    - With NFZ: routes around NFZ via waypoints
    """

    START_POS = np.array([-3.6, -3.6])

    def __init__(self):
        self.last_update_time = -10.0
        self.current_bearing = np.array([1.0, 0.0, 0.0])

        # NFZ waypoint state
        self._waypoints = None
        self._wp_idx = 0
        self._nfz_planned = False

    def reset_episode(self):
        self._wp_idx = 0
        self._nfz_planned = False
        self._waypoints = None
        self.last_update_time = -10.0

    def get_bearing(self, robot_pos, goal_pos, time_now=0.0, nfz_blocks=None):
        """
        Compute bearing toward goal.
        - Without NFZ: simple direct-to-goal (updated every 2s)
        - With NFZ: waypoint-based routing around NFZ blocks
        """
        if not nfz_blocks:
            # === ORIGINAL SIMPLE NAVIGATOR ===
            if time_now - self.last_update_time > 2.0:
                vec = goal_pos[:2] - robot_pos[:2]
                norm = np.linalg.norm(vec)
                if norm > 0.1:
                    self.current_bearing = np.array([vec[0]/norm, vec[1]/norm, 0.0])
                else:
                    self.current_bearing = np.array([0.0, 0.0, 0.0])
                self.last_update_time = time_now
            return self.current_bearing

        # === NFZ-AWARE WAYPOINT NAVIGATOR ===
        # Plan waypoints once per episode
        if not self._nfz_planned:
            self._waypoints = self._plan_waypoints(goal_pos, nfz_blocks)
            self._wp_idx = 0
            self._nfz_planned = True

        # Detect episode reset
        if np.linalg.norm(robot_pos[:2] - self.START_POS) < 0.5:
            self._wp_idx = 0

        wpts = self._waypoints or [goal_pos]

        # Advance waypoint if reached
        while self._wp_idx < len(wpts) - 1:
            dist_to_wpt = np.linalg.norm(robot_pos[:2] - wpts[self._wp_idx][:2])
            if dist_to_wpt < 1.2:
                self._wp_idx += 1
            else:
                break

        target = wpts[self._wp_idx]

        # Compute bearing toward current waypoint
        vec = target[:2] - robot_pos[:2]
        norm = np.linalg.norm(vec)
        if norm > 0.1:
            self.current_bearing = np.array([vec[0]/norm, vec[1]/norm, 0.0])
        else:
            vec2 = goal_pos[:2] - robot_pos[:2]
            n2 = np.linalg.norm(vec2)
            if n2 > 0.01:
                self.current_bearing = np.array([vec2[0]/n2, vec2[1]/n2, 0.0])

        return self.current_bearing

    # ------------------------------------------------------------------
    # Waypoint Planning
    # ------------------------------------------------------------------

    def _plan_waypoints(self, goal, nfz_blocks):
        """Plan route around NFZ. Returns list of 3D waypoints."""
        if not nfz_blocks:
            return [goal]

        z = float(goal[2])
        start2d = self.START_POS.copy()
        goal2d = goal[:2].copy()

        # Check if direct path is blocked
        if not self._segment_hits_nfz(start2d, goal2d, nfz_blocks, margin=0.3):
            return [goal]

        # Try bypassing NFZ corners (single waypoint)
        best_path = None
        best_cost = float('inf')
        MARGIN = 0.9

        for block in nfz_blocks:
            bmin = np.array(block['min'][:2])
            bmax = np.array(block['max'][:2])

            corners = [
                np.array([bmin[0] - MARGIN, bmin[1] - MARGIN]),  # SW
                np.array([bmax[0] + MARGIN, bmin[1] - MARGIN]),  # SE
                np.array([bmin[0] - MARGIN, bmax[1] + MARGIN]),  # NW
                np.array([bmax[0] + MARGIN, bmax[1] + MARGIN]),  # NE
            ]

            for corner in corners:
                corner = np.clip(corner, -3.4, 3.4)
                if self._point_in_nfz(corner, nfz_blocks, margin=0.1):
                    continue
                # Check BOTH segments: start→corner AND corner→goal
                if self._segment_hits_nfz(start2d, corner, nfz_blocks, margin=0.25):
                    continue
                if self._segment_hits_nfz(corner, goal2d, nfz_blocks, margin=0.25):
                    continue

                cost = (np.linalg.norm(corner - start2d) +
                        np.linalg.norm(goal2d - corner))

                if cost < best_cost:
                    best_cost = cost
                    wpt_3d = np.array([corner[0], corner[1], z])
                    best_path = [wpt_3d, goal]

        # If no single-waypoint path works, try two-waypoint route
        if best_path is None:
            for block in nfz_blocks:
                bmin = np.array(block['min'][:2])
                bmax = np.array(block['max'][:2])
                # Route via SE then NE (right side)
                se = np.clip(np.array([bmax[0] + MARGIN, bmin[1] - MARGIN]), -3.4, 3.4)
                ne = np.clip(np.array([bmax[0] + MARGIN, bmax[1] + MARGIN]), -3.4, 3.4)
                # Route via NW then NE (top side, going left then right)
                nw = np.clip(np.array([bmin[0] - MARGIN, bmax[1] + MARGIN]), -3.4, 3.4)

                two_wpt_routes = [
                    [se, ne],  # go right side
                    [nw, ne],  # go over top
                ]
                for route in two_wpt_routes:
                    ok = True
                    pts = [start2d] + route + [goal2d]
                    for i in range(len(pts) - 1):
                        if self._segment_hits_nfz(pts[i], pts[i+1], nfz_blocks, margin=0.2):
                            ok = False
                            break
                    if ok:
                        cost = sum(np.linalg.norm(pts[i+1] - pts[i]) for i in range(len(pts)-1))
                        if cost < best_cost:
                            best_cost = cost
                            best_path = [np.array([r[0], r[1], z]) for r in route] + [goal]

        return best_path if best_path else [goal]

    @staticmethod
    def _segment_hits_nfz(p1, p2, nfz_blocks, margin=0.2):
        d = p2 - p1
        for block in nfz_blocks:
            bmin = np.array(block['min'][:2]) - margin
            bmax = np.array(block['max'][:2]) + margin
            t_min, t_max = 0.0, 1.0
            ok = True
            for i in range(2):
                if abs(d[i]) < 1e-10:
                    if p1[i] < bmin[i] or p1[i] > bmax[i]:
                        ok = False
                        break
                else:
                    t1 = (bmin[i] - p1[i]) / d[i]
                    t2 = (bmax[i] - p1[i]) / d[i]
                    if t1 > t2:
                        t1, t2 = t2, t1
                    t_min = max(t_min, t1)
                    t_max = min(t_max, t2)
            if ok and t_min <= t_max:
                return True
        return False

    @staticmethod
    def _point_in_nfz(pos2d, nfz_blocks, margin=0.0):
        for block in nfz_blocks:
            if (block['min'][0] - margin <= pos2d[0] <= block['max'][0] + margin and
                    block['min'][1] - margin <= pos2d[1] <= block['max'][1] + margin):
                return True
        return False
