"""
3D Drone Navigation with No-Fly Zone (PyBullet)
==================================================
Same mixed-obstacle environment but with a GHOST-SHAPED
No-Fly Zone (NFZ) in the center of the arena.

NFZ Rules:
  - The NFZ is a hard wall — robot is physically pushed out
  - The NFZ is rendered as a translucent red region
  - All obstacles are placed OUTSIDE the NFZ

NFZ Shape: Ghost silhouette centered near (0, 0.5):
  - Rounded dome head (3 rows)
  - Wide body (3 rows)
  - 3 "feet" prongs at the bottom with gaps
"""

import pybullet as p
import pybullet_data
import numpy as np
import random
import math


class DroneNavNFZEnv:
    """
    3D drone navigation with ghost-shaped No-Fly Zone.
    Robot must navigate from start to goal while avoiding:
      - 10 dynamic obstacles (red spheres)
      - 15 static obstacles (blue spheres)
      - Ghost-shaped No-Fly Zone (translucent red region)
    """

    def __init__(self, gui=False, dt=0.05, obstacle_speed=1.0):
        self.dt = dt
        self.gui = gui
        self.obstacle_speed = obstacle_speed
        self.physics = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.setTimeStep(dt)

        # Robot
        self.robot_radius = 0.12
        self.max_speed = 3.0
        self.max_acc = 4.0
        self.goal_radius = 0.5
        self.flight_height = 1.0

        # Arena bounds
        self.arena_min = np.array([-4.0, -4.0, 0.3])
        self.arena_max = np.array([4.0, 4.0, 4.0])

        # Start and goal
        self.start_pos = np.array([-3.6, -3.6, self.flight_height])
        self.goal_pos = np.array([3.6, 3.6, self.flight_height])

        # NFZ: CROSS / PLUS (+) SHAPE
        # Centered at roughly (0, 0.5)
        # Vertical bar + horizontal bar forming a +
        self.nfz_blocks = [
            # --- VERTICAL BAR (tall, narrow) ---
            {"min": np.array([-0.5, -1.8]), "max": np.array([ 0.5,  2.8])},
            # --- HORIZONTAL BAR (wide, short) ---
            {"min": np.array([-2.0, -0.2]), "max": np.array([ 2.0,  1.2])},
        ]

        self.drone_pos = None
        self.drone_vel = None
        self.drone = None
        self.obstacles = []
        self.step_count = 0
        self.nfz_entered = False
        self.last_yaw = 0.0

        self.reset()

    def _point_in_nfz(self, xy):
        """Check if an xy point is inside the L-shaped NFZ."""
        x, y = xy[0], xy[1]
        for block in self.nfz_blocks:
            if (block["min"][0] <= x <= block["max"][0] and
                block["min"][1] <= y <= block["max"][1]):
                return True
        return False

    def reset(self):
        """Reset environment."""
        p.resetSimulation(physicsClientId=self.physics)
        p.setGravity(0, 0, 0, physicsClientId=self.physics)
        p.setTimeStep(self.dt, physicsClientId=self.physics)

        # Ground plane
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.physics)

        # Arena walls
        wall_color = [0.6, 0.6, 0.6, 0.3]
        walls = [
            ([-4, 0, 1.5], [0.05, 4, 1.5]),
            ([4, 0, 1.5],  [0.05, 4, 1.5]),
            ([0, -4, 1.5], [4, 0.05, 1.5]),
            ([0, 4, 1.5],  [4, 0.05, 1.5]),
        ]
        for wpos, whalf in walls:
            wvis = p.createVisualShape(p.GEOM_BOX, halfExtents=whalf,
                                       rgbaColor=wall_color,
                                       physicsClientId=self.physics)
            p.createMultiBody(0, -1, wvis, wpos, physicsClientId=self.physics)

        # === Render NFZ as translucent red boxes ===
        nfz_color = [1, 0.15, 0.15, 0.35]
        nfz_border_color = [1, 0, 0, 0.9]

        for block in self.nfz_blocks:
            center_x = (block["min"][0] + block["max"][0]) / 2
            center_y = (block["min"][1] + block["max"][1]) / 2
            half_x = (block["max"][0] - block["min"][0]) / 2
            half_y = (block["max"][1] - block["min"][1]) / 2

            # NFZ floor (translucent red)
            nfz_vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[half_x, half_y, 0.01],
                rgbaColor=nfz_color, physicsClientId=self.physics)
            p.createMultiBody(0, -1, nfz_vis,
                              [center_x, center_y, self.flight_height],
                              physicsClientId=self.physics)

            # NFZ vertical walls (thin red borders, visual)
            border_h = 0.3
            borders = [
                ([block["min"][0], center_y, self.flight_height],
                 [0.02, half_y + 0.02, border_h]),
                ([block["max"][0], center_y, self.flight_height],
                 [0.02, half_y + 0.02, border_h]),
                ([center_x, block["min"][1], self.flight_height],
                 [half_x + 0.02, 0.02, border_h]),
                ([center_x, block["max"][1], self.flight_height],
                 [half_x + 0.02, 0.02, border_h]),
            ]
            for bpos, bhalf in borders:
                bvis = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=bhalf,
                    rgbaColor=nfz_border_color,
                    physicsClientId=self.physics)
                p.createMultiBody(0, -1, bvis, bpos,
                                  physicsClientId=self.physics)

        # =========================================================
        # DRONE MODEL (compound body: fuselage + 4 arms + 4 rotors)
        # =========================================================
        self.drone_pos = self.start_pos.copy()
        self.drone_vel = np.zeros(3)

        body_half = [0.08, 0.08, 0.02]
        col_body = p.createCollisionShape(p.GEOM_BOX, halfExtents=body_half,
                                          physicsClientId=self.physics)
        vis_body = p.createVisualShape(p.GEOM_BOX, halfExtents=body_half,
                                       rgbaColor=[0.2, 0.2, 0.2, 1.0],
                                       physicsClientId=self.physics)

        arm_length = 0.15
        arm_radius = 0.012
        rotor_radius = 0.05
        rotor_height = 0.008

        arm_positions = [
            [arm_length, 0, 0], [-arm_length, 0, 0],
            [0, arm_length, 0], [0, -arm_length, 0],
        ]
        rotor_positions = [
            [arm_length, 0, 0.025], [-arm_length, 0, 0.025],
            [0, arm_length, 0.025], [0, -arm_length, 0.025],
        ]
        arm_colors = [
            [1.0, 0.2, 0.2, 1.0], [0.3, 0.3, 0.3, 1.0],
            [0.2, 0.8, 0.2, 1.0], [0.2, 0.8, 0.2, 1.0],
        ]
        rotor_color = [0.1, 0.1, 0.1, 0.8]

        lm, lc, lv, lp, lo = [], [], [], [], []
        lip, lio, lpa, ljt, lja = [], [], [], [], []

        for i in range(4):
            arm_orn = p.getQuaternionFromEuler(
                [0, math.pi/2, 0] if i < 2 else [math.pi/2, 0, 0])
            ac = p.createCollisionShape(p.GEOM_CYLINDER, radius=arm_radius,
                                        height=arm_length, physicsClientId=self.physics)
            av = p.createVisualShape(p.GEOM_CYLINDER, radius=arm_radius,
                                     length=arm_length, rgbaColor=arm_colors[i],
                                     physicsClientId=self.physics)
            mid = [ap / 2 for ap in arm_positions[i]]
            lm.append(0.01); lc.append(ac); lv.append(av)
            lp.append(mid); lo.append(arm_orn)
            lip.append([0,0,0]); lio.append([0,0,0,1])
            lpa.append(0); ljt.append(p.JOINT_FIXED); lja.append([0,0,0])

            rc = p.createCollisionShape(p.GEOM_CYLINDER, radius=rotor_radius,
                                        height=rotor_height, physicsClientId=self.physics)
            rv = p.createVisualShape(p.GEOM_CYLINDER, radius=rotor_radius,
                                     length=rotor_height, rgbaColor=rotor_color,
                                     physicsClientId=self.physics)
            lm.append(0.005); lc.append(rc); lv.append(rv)
            lp.append(rotor_positions[i]); lo.append([0,0,0,1])
            lip.append([0,0,0]); lio.append([0,0,0,1])
            lpa.append(0); ljt.append(p.JOINT_FIXED); lja.append([0,0,0])

        self.drone = p.createMultiBody(
            baseMass=0.5, baseCollisionShapeIndex=col_body,
            baseVisualShapeIndex=vis_body,
            basePosition=self.drone_pos.tolist(),
            linkMasses=lm, linkCollisionShapeIndices=lc,
            linkVisualShapeIndices=lv, linkPositions=lp,
            linkOrientations=lo, linkInertialFramePositions=lip,
            linkInertialFrameOrientations=lio, linkParentIndices=lpa,
            linkJointTypes=ljt, linkJointAxis=lja,
            physicsClientId=self.physics
        )

        # Goal marker
        goal_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.2,
                                       rgbaColor=[1, 0.85, 0, 0.9],
                                       physicsClientId=self.physics)
        p.createMultiBody(0, -1, goal_vis, self.goal_pos,
                          physicsClientId=self.physics)

        # Start marker
        start_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.1,
                                        rgbaColor=[0, 1, 0, 0.5],
                                        physicsClientId=self.physics)
        p.createMultiBody(0, -1, start_vis, self.start_pos,
                          physicsClientId=self.physics)

        # Spawn obstacles (all OUTSIDE NFZ)
        self.obstacles = []
        self._spawn_obstacles()
        self.step_count = 0
        self.nfz_entered = False

        return self._get_obs()

    def _spawn_obstacles(self):
        """
        Spawn 10 dynamic + 15 static obstacles, ALL outside the NFZ.
        Positions from nfz_env.xml.
        """
        # Dynamic obstacles (from nfz_env.xml, all outside NFZ)
        dynamic_positions = [
            [-3.0, -1.5], [-2.0, -2.5], [-2.5, 2.8], [2.5, -1.5],
            [3.0, -2.0],  [-2.5, 1.5],  [2.5, 2.8],  [2.5, 1.8],
            [-1.5, -3.0], [3.0, 2.5],
        ]

        dynamic_radius = 0.1
        for pos_2d in dynamic_positions:
            pos = np.array([pos_2d[0], pos_2d[1], self.flight_height])
            angle = random.uniform(0, 2 * np.pi)
            vel = np.array([
                self.obstacle_speed * np.cos(angle),
                self.obstacle_speed * np.sin(angle),
                0.0,
            ])
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=dynamic_radius,
                                         physicsClientId=self.physics)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=dynamic_radius,
                                      rgbaColor=[1, 0.3, 0.3, 1],
                                      physicsClientId=self.physics)
            body = p.createMultiBody(1.0, col, vis, pos,
                                     physicsClientId=self.physics)
            self.obstacles.append({
                "body": body, "pos": pos.copy(), "vel": vel.copy(),
                "is_dynamic": True, "radius": dynamic_radius,
            })

        # Static obstacles (from nfz_env.xml, all outside NFZ)
        static_positions = [
            [-3.0, -2.5], [3.0, -2.5],  [-3.0, 3.0],  [3.0, 2.8],
            [-2.5, -1.8], [2.5, -1.8],  [-2.5, 3.0],  [2.5, 3.0],
            [-2.8, -0.5], [2.8, 0.5],   [0.8, -2.5],  [-0.8, 3.2],
            [-2.5, 3.5],  [2.5, -3.0],  [-2.8, 1.5],
        ]

        static_radius = 0.08
        for pos_2d in static_positions:
            pos = np.array([pos_2d[0], pos_2d[1], self.flight_height])
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=static_radius,
                                         physicsClientId=self.physics)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=static_radius,
                                      rgbaColor=[0.15, 0.2, 0.8, 1],
                                      physicsClientId=self.physics)
            body = p.createMultiBody(0, col, vis, pos,
                                     physicsClientId=self.physics)
            self.obstacles.append({
                "body": body, "pos": pos.copy(), "vel": np.zeros(3),
                "is_dynamic": False, "radius": static_radius,
            })

    def _push_out_of_nfz(self, pos, vel):
        """
        STRICT NFZ enforcement.
        Check if the robot body (including radius) overlaps any NFZ block.
        If so, push it completely outside with a generous margin.
        Returns corrected (pos, vel, was_blocked).
        """
        margin = self.robot_radius + 0.1  # Generous margin outside NFZ edge
        corrected = False

        for block in self.nfz_blocks:
            # Inflate NFZ by robot radius — the center must stay this far out
            bmin_x = block["min"][0] - self.robot_radius
            bmax_x = block["max"][0] + self.robot_radius
            bmin_y = block["min"][1] - self.robot_radius
            bmax_y = block["max"][1] + self.robot_radius

            if (bmin_x <= pos[0] <= bmax_x and bmin_y <= pos[1] <= bmax_y):
                # Robot body overlaps this block — find nearest edge to push to
                dist_left  = pos[0] - bmin_x
                dist_right = bmax_x - pos[0]
                dist_bot   = pos[1] - bmin_y
                dist_top   = bmax_y - pos[1]

                min_dist = min(dist_left, dist_right, dist_bot, dist_top)

                if min_dist == dist_left:
                    pos[0] = bmin_x - margin
                    vel[0] = min(vel[0], 0)
                elif min_dist == dist_right:
                    pos[0] = bmax_x + margin
                    vel[0] = max(vel[0], 0)
                elif min_dist == dist_bot:
                    pos[1] = bmin_y - margin
                    vel[1] = min(vel[1], 0)
                else:
                    pos[1] = bmax_y + margin
                    vel[1] = max(vel[1], 0)

                corrected = True

        return pos, vel, corrected


    def step(self, action_acc):
        """Step the environment. NFZ is a HARD WALL — robot cannot enter."""
        action_acc = np.array(action_acc, dtype=np.float64)
        target_yaw = 0.0
        if len(action_acc) >= 4:
            target_yaw = action_acc[3]
            self.last_yaw = target_yaw

        action_acc = action_acc[:3].copy()
        action_acc[2] = 0.0  # Lock z
        action_acc = np.clip(action_acc, -self.max_acc, self.max_acc)

        # Save previous position
        prev_pos = self.drone_pos.copy()

        # Update drone
        self.drone_vel += action_acc * self.dt
        self.drone_vel[2] = 0.0
        speed = np.linalg.norm(self.drone_vel)
        if speed > self.max_speed:
            self.drone_vel = self.drone_vel / speed * self.max_speed

        self.drone_pos += self.drone_vel * self.dt
        self.drone_pos[2] = self.flight_height

        # Wall margin clamping
        wall_margin = 0.3
        self.drone_pos[:2] = np.clip(
            self.drone_pos[:2],
            self.arena_min[:2] + wall_margin,
            self.arena_max[:2] - wall_margin)

        # === HARD NFZ ENFORCEMENT ===
        self.drone_pos, self.drone_vel, nfz_blocked = self._push_out_of_nfz(
            self.drone_pos, self.drone_vel)

        # Calculate drone tilt
        forward_speed = np.linalg.norm(self.drone_vel[:2])
        pitch = -np.clip(forward_speed * 0.08, 0, math.radians(15))
        if hasattr(self, '_prev_yaw'):
            yaw_rate = target_yaw - self._prev_yaw
            yaw_rate = (yaw_rate + math.pi) % (2 * math.pi) - math.pi
            roll = np.clip(yaw_rate * 2.0, -math.radians(10), math.radians(10))
        else:
            roll = 0.0
        self._prev_yaw = target_yaw

        orn = p.getQuaternionFromEuler([roll, pitch, target_yaw])
        p.resetBasePositionAndOrientation(
            self.drone, self.drone_pos, orn,
            physicsClientId=self.physics)

        # Update dynamic obstacles
        for obs in self.obstacles:
            if obs["is_dynamic"]:
                obs["pos"] += obs["vel"] * self.dt
                for axis in range(2):
                    if obs["pos"][axis] < -3.5:
                        obs["vel"][axis] = abs(obs["vel"][axis])
                    elif obs["pos"][axis] > 3.5:
                        obs["vel"][axis] = -abs(obs["vel"][axis])
                obs["pos"][2] = self.flight_height
                obs["vel"][2] = 0.0

                if random.random() < 0.015:
                    angle = random.uniform(0, 2 * np.pi)
                    obs["vel"] = np.array([
                        self.obstacle_speed * np.cos(angle),
                        self.obstacle_speed * np.sin(angle), 0.0])

                p.resetBasePositionAndOrientation(
                    obs["body"], obs["pos"], [0, 0, 0, 1],
                    physicsClientId=self.physics)

        p.stepSimulation(physicsClientId=self.physics)
        self.step_count += 1

        # --- Evaluate ---
        done = False
        reward = -0.01
        info = {}

        # Goal check
        dist_to_goal = np.linalg.norm(self.drone_pos - self.goal_pos)
        if dist_to_goal < self.goal_radius:
            done = True
            reward += 10.0
            info["success"] = True

        # Collision check
        for obs in self.obstacles:
            dist = np.linalg.norm(self.drone_pos - obs["pos"])
            if dist < self.robot_radius + obs["radius"]:
                done = True
                reward -= 10.0
                info["collision"] = True
                break

        # NFZ blocked info (robot was pushed back — it tried but couldn't enter)
        if nfz_blocked:
            info["nfz_blocked"] = True

        info["dist_to_goal"] = dist_to_goal
        info["steps"] = self.step_count
        info["in_nfz"] = False  # Can never be in NFZ now (hard wall)

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """Return observation dict with NFZ info."""
        return {
            "pos": self.drone_pos.copy(),
            "vel": self.drone_vel.copy(),
            "goal": self.goal_pos.copy(),
            "obstacles": [
                {
                    "pos": obs["pos"].copy(),
                    "vel": obs["vel"].copy(),
                    "is_dynamic": obs["is_dynamic"],
                    "radius": obs["radius"],
                }
                for obs in self.obstacles
            ],
            "nfz_blocks": self.nfz_blocks,
            "in_nfz": self._point_in_nfz(self.drone_pos[:2]),
        }

    def render(self, width=640, height=480, camera="pov"):
        """Render a frame."""
        if camera == "pov":
            direction = self.goal_pos - self.drone_pos
            norm = np.linalg.norm(direction[:2])
            if norm > 0.01:
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.array([1, 0, 0])
            eye = self.drone_pos + np.array([0, 0, 0.1])
            target = eye + direction * 5.0
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=eye.tolist(),
                cameraTargetPosition=target.tolist(),
                cameraUpVector=[0, 0, 1],
                physicsClientId=self.physics)

        elif camera == "top":
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[0, 0, 16],
                cameraTargetPosition=[0, 0, 1],
                cameraUpVector=[1, 0, 0],
                physicsClientId=self.physics)

        elif camera == "front":
            cam_pos = self.drone_pos + np.array([4, -2, 2])
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=cam_pos.tolist(),
                cameraTargetPosition=self.drone_pos.tolist(),
                cameraUpVector=[0, 0, 1],
                physicsClientId=self.physics)

        elif camera == "chase":
            if not hasattr(self, '_smooth_cam_yaw'):
                self._smooth_cam_yaw = math.atan2(
                    self.goal_pos[1] - self.start_pos[1],
                    self.goal_pos[0] - self.start_pos[0])
            speed = np.linalg.norm(self.drone_vel[:2])
            if speed > 0.15:
                travel_yaw = math.atan2(self.drone_vel[1], self.drone_vel[0])
                diff = (travel_yaw - self._smooth_cam_yaw + math.pi) % (2 * math.pi) - math.pi
                self._smooth_cam_yaw += diff * 0.05
            yaw = self._smooth_cam_yaw
            cam_offset = np.array([
                -2.5 * math.cos(yaw),
                -2.5 * math.sin(yaw),
                1.2
            ])
            eye = self.drone_pos + cam_offset
            target = self.drone_pos + np.array([
                1.0 * math.cos(yaw),
                1.0 * math.sin(yaw),
                0.0
            ])
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=eye.tolist(),
                cameraTargetPosition=target.tolist(),
                cameraUpVector=[0, 0, 1],
                physicsClientId=self.physics)

        else:  # follow
            direction = self.goal_pos - self.drone_pos
            norm = np.linalg.norm(direction)
            if norm > 0.01:
                direction = direction / norm
            else:
                direction = np.array([1, 0, 0])
            cam_pos = self.drone_pos - direction * 3 + np.array([0, 0, 2])
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=cam_pos.tolist(),
                cameraTargetPosition=self.drone_pos.tolist(),
                cameraUpVector=[0, 0, 1],
                physicsClientId=self.physics)

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=70, aspect=width / height, nearVal=0.05, farVal=30,
            physicsClientId=self.physics)

        _, _, rgba, _, _ = p.getCameraImage(
            width, height, view_matrix, proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.physics)

        rgb = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        bgr = rgb[:, :, ::-1].copy()
        return bgr

    def close(self):
        try:
            p.disconnect(physicsClientId=self.physics)
        except Exception:
            pass
