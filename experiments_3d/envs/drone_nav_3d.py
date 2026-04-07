"""
3D Drone Navigation Environment (PyBullet)
============================================
Zero-gravity 3D environment with a drone navigating from start to goal
while avoiding mixed obstacles (static + dynamic).

Obstacle properties match the 2D MuJoCo mixed_random_env_20 environment:
  - 10 dynamic sphere obstacles (red, speed=1.0, 30% direction change)
  - 10 static sphere obstacles (blue, fixed position)
  - Arena: 8×8  (x: -4 to 4, y: -4 to 4, z: 0.3 to 4)
  - All obstacles at SAME HEIGHT as robot (z=1.0)
  - Robot: sphere radius 0.12
  - Start: [-3.6, -3.6, 1.0]  →  Goal: [3.6, 3.6, 1.0]
  - Collision threshold: 0.22 (robot_r + largest_obs_r)
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import random


class DroneNav3DEnv:
    """
    3D drone navigation environment using PyBullet.
    Mirrors the 2D MuJoCo mixed_random_env_20 but in full 3D.
    """

    def __init__(self, gui=False, dt=0.05, num_static=10, num_dynamic=10,
                 obstacle_speed=1.0):
        self.dt = dt
        self.gui = gui
        self.num_static = num_static
        self.num_dynamic = num_dynamic
        self.obstacle_speed = obstacle_speed
        self.physics = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.setTimeStep(dt)

        # Robot properties (matching 2D: robot_geom size=0.12)
        self.robot_radius = 0.12
        self.max_speed = 3.0
        self.max_acc = 4.0
        self.goal_radius = 0.5

        # Arena bounds (matching 2D: -4 to 4 in x/y)
        self.arena_min = np.array([-4.0, -4.0, 0.3])
        self.arena_max = np.array([4.0, 4.0, 4.0])

        # Start and goal (matching 2D)
        self.start_pos = np.array([-3.6, -3.6, 1.0])
        self.goal_pos = np.array([3.6, 3.6, 1.0])

        self.drone_pos = None
        self.drone_vel = None
        self.drone = None
        self.plane = None
        self.obstacles = []
        self.step_count = 0
        self.last_yaw = 0.0  # Track yaw for chase camera

        self.reset()

    def reset(self):
        """Reset environment: reposition drone, respawn obstacles."""
        p.resetSimulation(physicsClientId=self.physics)
        p.setGravity(0, 0, 0, physicsClientId=self.physics)
        p.setTimeStep(self.dt, physicsClientId=self.physics)

        # Ground plane
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.physics)

        # Arena boundary lines (thin ground-level markers, don't block camera)
        boundary_color = [0.4, 0.4, 0.4, 0.5]
        boundaries = [
            ([-4, 0, 0.01], [0.02, 4,  0.02]),
            ([4,  0, 0.01], [0.02, 4,  0.02]),
            ([0, -4, 0.01], [4,  0.02, 0.02]),
            ([0,  4, 0.01], [4,  0.02, 0.02]),
        ]
        for bpos, bhalf in boundaries:
            bvis = p.createVisualShape(p.GEOM_BOX, halfExtents=bhalf,
                                       rgbaColor=boundary_color,
                                       physicsClientId=self.physics)
            p.createMultiBody(0, -1, bvis, bpos, physicsClientId=self.physics)

        # =========================================================
        # DRONE MODEL (compound body: fuselage + 4 arms + 4 rotors)
        # =========================================================
        self.drone_pos = self.start_pos.copy()
        self.drone_vel = np.zeros(3)

        # Central fuselage (dark gray flat box)
        body_half = [0.08, 0.08, 0.02]
        col_body = p.createCollisionShape(p.GEOM_BOX, halfExtents=body_half,
                                          physicsClientId=self.physics)
        vis_body = p.createVisualShape(p.GEOM_BOX, halfExtents=body_half,
                                       rgbaColor=[0.2, 0.2, 0.2, 1.0],
                                       physicsClientId=self.physics)

        # 4 arms (thin cylinders) and 4 rotors (flat discs)
        arm_length = 0.15
        arm_radius = 0.012
        rotor_radius = 0.05
        rotor_height = 0.008

        arm_positions = [
            [arm_length, 0, 0],    # Front
            [-arm_length, 0, 0],   # Back
            [0, arm_length, 0],    # Left
            [0, -arm_length, 0],   # Right
        ]
        rotor_positions = [
            [arm_length, 0, 0.025],
            [-arm_length, 0, 0.025],
            [0, arm_length, 0.025],
            [0, -arm_length, 0.025],
        ]
        # Front arms: red, Back/Side: green to show heading direction
        arm_colors = [
            [1.0, 0.2, 0.2, 1.0],  # Front = RED
            [0.3, 0.3, 0.3, 1.0],  # Back
            [0.2, 0.8, 0.2, 1.0],  # Left = GREEN
            [0.2, 0.8, 0.2, 1.0],  # Right = GREEN
        ]
        rotor_color = [0.1, 0.1, 0.1, 0.8]

        # Build compound shape arrays
        link_masses = []
        link_col_shapes = []
        link_vis_shapes = []
        link_positions = []
        link_orientations = []
        link_inertial_pos = []
        link_inertial_orn = []
        link_parent = []
        link_joint_types = []
        link_joint_axes = []

        for i in range(4):
            # Arm
            arm_orn = [0, 0, 0, 1]
            if i < 2:  # Front/back arm along X
                arm_orn = p.getQuaternionFromEuler([0, math.pi/2, 0])
            else:  # Left/right arm along Y
                arm_orn = p.getQuaternionFromEuler([math.pi/2, 0, 0])

            arm_col = p.createCollisionShape(p.GEOM_CYLINDER,
                                             radius=arm_radius,
                                             height=arm_length,
                                             physicsClientId=self.physics)
            arm_vis = p.createVisualShape(p.GEOM_CYLINDER,
                                          radius=arm_radius,
                                          length=arm_length,
                                          rgbaColor=arm_colors[i],
                                          physicsClientId=self.physics)
            mid_pos = [ap / 2 for ap in arm_positions[i]]

            link_masses.append(0.01)
            link_col_shapes.append(arm_col)
            link_vis_shapes.append(arm_vis)
            link_positions.append(mid_pos)
            link_orientations.append(arm_orn)
            link_inertial_pos.append([0, 0, 0])
            link_inertial_orn.append([0, 0, 0, 1])
            link_parent.append(0)
            link_joint_types.append(p.JOINT_FIXED)
            link_joint_axes.append([0, 0, 0])

            # Rotor disc
            rotor_col = p.createCollisionShape(p.GEOM_CYLINDER,
                                               radius=rotor_radius,
                                               height=rotor_height,
                                               physicsClientId=self.physics)
            rotor_vis = p.createVisualShape(p.GEOM_CYLINDER,
                                            radius=rotor_radius,
                                            length=rotor_height,
                                            rgbaColor=rotor_color,
                                            physicsClientId=self.physics)

            link_masses.append(0.005)
            link_col_shapes.append(rotor_col)
            link_vis_shapes.append(rotor_vis)
            link_positions.append(rotor_positions[i])
            link_orientations.append([0, 0, 0, 1])
            link_inertial_pos.append([0, 0, 0])
            link_inertial_orn.append([0, 0, 0, 1])
            link_parent.append(0)
            link_joint_types.append(p.JOINT_FIXED)
            link_joint_axes.append([0, 0, 0])

        self.drone = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=col_body,
            baseVisualShapeIndex=vis_body,
            basePosition=self.drone_pos.tolist(),
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_col_shapes,
            linkVisualShapeIndices=link_vis_shapes,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_inertial_pos,
            linkInertialFrameOrientations=link_inertial_orn,
            linkParentIndices=link_parent,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axes,
            physicsClientId=self.physics
        )

        # Goal marker (golden sphere, visual only)
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

        # Spawn obstacles
        self.obstacles = []
        self._spawn_obstacles()
        self.step_count = 0

        return self._get_obs()

    def _spawn_obstacles(self):
        """
        Spawn obstacles matching the 2D mixed_random_env_20.
        ALL at z=1.0 (same height as robot) so they are visible
        in POV camera and create real collision threats.
        """
        # Static obstacle positions from mixed_random_env_20.xml
        static_positions_2d = [
            [-2.5, -2.5], [2.5, -2.5], [-2.5, 2.5], [2.5, 2.5],
            [0.0, -2.5],  [0.0, 2.5],  [-2.5, 0.0], [2.5, 0.0],
            [-1.0, -0.5], [1.0, 0.5],
        ]

        # --- Static obstacles (spheres, blue, r=0.08) ALL at z=1.0 ---
        static_radius = 0.08
        for pos_2d in static_positions_2d[:self.num_static]:
            pos = np.array([pos_2d[0], pos_2d[1], 1.0])
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=static_radius,
                                         physicsClientId=self.physics)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=static_radius,
                                      rgbaColor=[0.15, 0.2, 0.8, 1],
                                      physicsClientId=self.physics)
            body = p.createMultiBody(0, col, vis, pos,
                                     physicsClientId=self.physics)
            self.obstacles.append({
                "body": body,
                "pos": pos.copy(),
                "vel": np.zeros(3),
                "is_dynamic": False,
                "radius": static_radius,
            })

        # Dynamic obstacle positions from mixed_random_env_20.xml
        dynamic_positions_2d = [
            [-2.0, -1.5], [-1.0, -2.0], [0.0, -1.0], [1.0, -1.5],
            [2.0, -2.0],  [-1.5, 0.5],  [0.5, 0.0],  [1.5, 1.0],
            [-0.5, 1.5],  [0.5, 2.0],
        ]

        # --- Dynamic obstacles (spheres, red, r=0.1) ALL at z=1.0 ---
        dynamic_radius = 0.1
        for pos_2d in dynamic_positions_2d[:self.num_dynamic]:
            pos = np.array([pos_2d[0], pos_2d[1], 1.0])
            # Random velocity (matching 2D: speed=1.0, random angle)
            angle_xy = random.uniform(0, 2 * np.pi)
            vel = np.array([
                self.obstacle_speed * np.cos(angle_xy),
                self.obstacle_speed * np.sin(angle_xy),
                0.0,  # No z movement — stay at same height as robot
            ])
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=dynamic_radius,
                                         physicsClientId=self.physics)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=dynamic_radius,
                                      rgbaColor=[1, 0.3, 0.3, 1],
                                      physicsClientId=self.physics)
            body = p.createMultiBody(1.0, col, vis, pos,
                                     physicsClientId=self.physics)
            self.obstacles.append({
                "body": body,
                "pos": pos.copy(),
                "vel": vel.copy(),
                "is_dynamic": True,
                "radius": dynamic_radius,
            })

    def step(self, action_acc):
        """
        Step the environment with an acceleration command.

        Args:
            action_acc: 3D acceleration vector [ax, ay, az]

        Returns:
            obs, reward, done, info
        """
        action_acc = np.array(action_acc, dtype=np.float64)
        target_yaw = 0.0
        if len(action_acc) >= 4:
            target_yaw = action_acc[3]
            self.last_yaw = target_yaw  # Store for chase camera

        # Lock z-axis: no vertical acceleration allowed
        action_acc = action_acc[:3].copy()
        action_acc[2] = 0.0
        action_acc = np.clip(action_acc, -self.max_acc, self.max_acc)

        # Update drone velocity and position
        self.drone_vel += action_acc * self.dt
        self.drone_vel[2] = 0.0  # No vertical movement
        speed = np.linalg.norm(self.drone_vel)
        if speed > self.max_speed:
            self.drone_vel = self.drone_vel / speed * self.max_speed

        self.drone_pos += self.drone_vel * self.dt
        self.drone_pos[2] = 1.0  # Stay at obstacle height


        # Clamp to arena bounds with wall margin (drone cannot touch walls)
        wall_margin = 0.3  # Keep 0.3m from arena walls
        self.drone_pos[:2] = np.clip(
            self.drone_pos[:2],
            self.arena_min[:2] + wall_margin,
            self.arena_max[:2] - wall_margin)
        
        # Dampen velocity component toward wall (prevent wall-riding)
        for axis in range(2):
            if (self.drone_pos[axis] <= self.arena_min[axis] + wall_margin + 0.02 or
                self.drone_pos[axis] >= self.arena_max[axis] - wall_margin - 0.02):
                self.drone_vel[axis] *= 0.7  # Gently reduce velocity along wall

        # Calculate drone tilt for visual realism
        # Decompose acceleration into drone-local forward and lateral components
        # Forward = along yaw direction  →  pitch (lean forward/backward)
        # Lateral = perpendicular to yaw →  roll (bank left/right)

        # Smooth yaw transition to prevent jerky snapping
        if not hasattr(self, '_smooth_yaw'):
            self._smooth_yaw = target_yaw

        yaw_diff = (target_yaw - self._smooth_yaw + math.pi) % (2 * math.pi) - math.pi
        self._smooth_yaw += yaw_diff * 0.15  # Smooth interpolation factor

        yaw = self._smooth_yaw

        # Drone's local frame axes (in world coordinates)
        forward_dir = np.array([math.cos(yaw), math.sin(yaw)])   # Drone's forward
        lateral_dir = np.array([-math.sin(yaw), math.cos(yaw)])  # Drone's left

        # Project acceleration onto forward and lateral axes
        accel_xy = action_acc[:2]
        forward_accel = np.dot(accel_xy, forward_dir)  # + = accelerating forward
        lateral_accel = np.dot(accel_xy, lateral_dir)   # + = accelerating left

        # Pitch: lean forward when accelerating forward (max 20 degrees)
        pitch = -np.clip(forward_accel * 0.06, -math.radians(20), math.radians(20))

        # Roll: bank into turns — tilt right when accelerating right (max 25 degrees)
        roll = np.clip(-lateral_accel * 0.08, -math.radians(25), math.radians(25))

        orn = p.getQuaternionFromEuler([roll, pitch, yaw])
        p.resetBasePositionAndOrientation(
            self.drone, self.drone_pos, orn,
            physicsClientId=self.physics)

        # Update dynamic obstacles (matching 2D behavior)
        for obs in self.obstacles:
            if obs["is_dynamic"]:
                obs["pos"] = obs["pos"] + obs["vel"] * self.dt

                # Bounce off arena walls (matching 2D: walls at ±3.5)
                for axis in range(2):  # x and y only
                    if obs["pos"][axis] < -3.5:
                        obs["vel"][axis] = abs(obs["vel"][axis])
                    elif obs["pos"][axis] > 3.5:
                        obs["vel"][axis] = -abs(obs["vel"][axis])

                # Keep at z=1.0 (same height as robot)
                obs["pos"][2] = 1.0
                obs["vel"][2] = 0.0

                # 30% direction change per second (matching 2D)
                # At dt=0.05 → 20 steps/sec → 0.3/20 = 0.015 per step
                if random.random() < 0.015:
                    angle_xy = random.uniform(0, 2 * np.pi)
                    obs["vel"] = np.array([
                        self.obstacle_speed * np.cos(angle_xy),
                        self.obstacle_speed * np.sin(angle_xy),
                        0.0,
                    ])

                p.resetBasePositionAndOrientation(
                    obs["body"], obs["pos"], [0, 0, 0, 1],
                    physicsClientId=self.physics)

        p.stepSimulation(physicsClientId=self.physics)
        self.step_count += 1

        # --- Evaluate ---
        done = False
        reward = -0.01  # time penalty
        info = {}

        # Goal check
        dist_to_goal = np.linalg.norm(self.drone_pos - self.goal_pos)
        if dist_to_goal < self.goal_radius:
            done = True
            reward += 10.0
            info["success"] = True

        # Collision check (matching 2D: robot_r + obs_r)
        for obs in self.obstacles:
            dist = np.linalg.norm(self.drone_pos - obs["pos"])
            collision_dist = self.robot_radius + obs["radius"]
            if dist < collision_dist:
                done = True
                reward -= 10.0
                info["collision"] = True
                break

        # Ground collision
        if self.drone_pos[2] < 0.15:
            done = True
            reward -= 5.0
            info["collision"] = True

        # Wall collision check (drone too close to arena boundary)
        wall_margin = 0.3
        for axis in range(2):
            if (self.drone_pos[axis] <= self.arena_min[axis] + wall_margin + 0.01 or
                self.drone_pos[axis] >= self.arena_max[axis] - wall_margin - 0.01):
                reward -= 0.1  # Penalty near wall (not terminal, but discourages)

        # Progress reward
        reward += 0.001 * (np.linalg.norm(self.start_pos - self.goal_pos) - dist_to_goal)

        info["dist_to_goal"] = dist_to_goal
        info["steps"] = self.step_count

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """Return observation dict."""
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
            ]
        }

    def render(self, width=640, height=480, camera="pov"):
        """
        Render a frame from PyBullet.

        Args:
            width: Image width
            height: Image height
            camera: 'pov' (robot POV), 'front', 'top', 'follow'

        Returns:
            BGR numpy array (H, W, 3) for cv2.VideoWriter
        """
        if camera == "pov":
            # Robot POV: camera at drone, looking toward goal
            direction = self.goal_pos - self.drone_pos
            norm = np.linalg.norm(direction[:2])  # Only xy direction
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

        elif camera == "front":
            cam_pos = self.drone_pos + np.array([4, -2, 2])
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=cam_pos.tolist(),
                cameraTargetPosition=self.drone_pos.tolist(),
                cameraUpVector=[0, 0, 1],
                physicsClientId=self.physics)

        elif camera == "top":
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[0, 0, 16],
                cameraTargetPosition=[0, 0, 1],
                cameraUpVector=[1, 0, 0],
                physicsClientId=self.physics)

        elif camera == "chase":
            # Smooth chase camera: uses exponential smoothing on travel direction
            # to prevent jitter when drone yaw changes rapidly
            if not hasattr(self, '_smooth_cam_yaw'):
                self._smooth_cam_yaw = math.atan2(
                    self.goal_pos[1] - self.start_pos[1],
                    self.goal_pos[0] - self.start_pos[0])

            # Update smooth yaw from actual velocity direction (not raw yaw)
            speed = np.linalg.norm(self.drone_vel[:2])
            if speed > 0.15:
                travel_yaw = math.atan2(self.drone_vel[1], self.drone_vel[0])
                # Smooth interpolation (0.03 = very smooth, 0.15 = responsive)
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
        """Disconnect physics client."""
        try:
            p.disconnect(physicsClientId=self.physics)
        except Exception:
            pass
