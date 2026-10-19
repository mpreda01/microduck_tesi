"""Offscreen rendering: videos of episodes and the still preview of the situations.
On a headless machine set MUJOCO_GL=egl before importing mujoco."""
from __future__ import annotations

import math

import mujoco
import numpy as np


class Video:
    """Frames of one episode from a camera tracking the trunk."""

    def __init__(self, model, video_conf: dict):
        self.renderer = mujoco.Renderer(model, video_conf["height"], video_conf["width"])
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = model.body("trunk_base").id
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 0.9, -20.0, 135.0
        self.every = video_conf["every_n_steps"]
        self.frames = []

    def capture(self, data, step: int):
        if step % self.every == 0:
            self.renderer.update_scene(data, camera=self.cam)
            self.frames.append(self.renderer.render().copy())

    def save(self, path, fps: float):
        import imageio
        imageio.mimsave(path, self.frames, fps=fps)
        self.renderer.close()


def start_still(model, data, x, y, z, yaw, height=180, width=240):
    """One image of the robot at its start, seen from behind and above."""
    r = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [x, y, z - 0.1]
    cam.distance, cam.elevation, cam.azimuth = 2.0, -25.0, math.degrees(yaw) + 150.0
    r.update_scene(data, camera=cam)
    image = r.render().copy()
    r.close()
    return image


def tile(images: list, cols: int = 6):
    """Grid of equally sized images, `cols` per row, in order."""
    blank = np.zeros_like(images[0])
    rows = [np.concatenate(images[i:i + cols] + [blank] * (cols - len(images[i:i + cols])), axis=1)
            for i in range(0, len(images), cols)]
    return np.concatenate(rows, axis=0)
