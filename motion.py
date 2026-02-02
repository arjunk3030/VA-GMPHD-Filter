import numpy as np
import time

from util_files.TrajectorySettings import PATH
from scipy.interpolate import interp1d

class RobotController:
    def __init__(self, model, data, viewer):
        self.model = model
        self.data = data
        self.viewer = viewer

    def step_sim(self, dt=0.01):
        import mujoco
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
        time.sleep(dt)
        mujoco.mj_forward(self.model, self.data)

    def turn_robot(self, target_angle_degrees, duration=1.5, dt=0.01):
        current_angle = np.degrees(self.data.qpos[2])
        steps = int(duration / dt)
        step_size = (target_angle_degrees - current_angle) / steps

        for _ in range(steps):
            current_angle += step_size
            self.data.qpos[2] = np.radians(current_angle)
            self.data.ctrl[2] = np.radians(current_angle)
            self.step_sim(dt)

    def turn_camera(self, target_angle_degrees, duration=0.05, dt=0.01):
        current_angle = np.degrees(self.data.qpos[3])
        steps = int(duration / dt)
        step_size = (target_angle_degrees - current_angle) / steps

        for _ in range(steps):
            current_angle += step_size
            self.data.qpos[3] = np.radians(current_angle)
            self.data.ctrl[3] = np.radians(current_angle)
            self.step_sim(dt)

    def move_to_waypoint(self, waypoint_idx, frame_duration):
        # --- build spline ---
        points = np.array([wp.pos for wp in PATH])
        points[:, 0] -= self.model.body("base_link").pos[0]
        points[:, 1] -= self.model.body("base_link").pos[1]

        spline = interp1d(np.arange(len(points)), points, kind="linear", axis=0)

        # --- target position from spline ---
        target_pos = spline(waypoint_idx)
        current_pos = np.array([self.data.qpos[0], self.data.qpos[1]])

        distance = np.linalg.norm(target_pos - current_pos)
        duration = distance / frame_duration
        steps = max(int(duration / 0.01), 1)

        step_size = (target_pos - current_pos) / steps

        for _ in range(steps):
            current_pos += step_size
            self.data.qpos[0:2] = current_pos
            self.data.ctrl[0:2] = current_pos
            self.step_sim(0.01)
