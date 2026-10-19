"""Per-step data (plan sections 4.8-4.9): one row per control step of every episode, saved as one
compressed .npz per (controller, situation), described by a fields.json next to the files."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from steerduck.io import save_json
from steerduck.sim.metrics import fall_labels
from steerduck.sim.scene import DEFAULT_POSE, JOINT_ORDER
from steerduck.sim.terrains import TERRAIN_KINDS

# field -> (dtype in the .npz files, meaning)
STEP_FIELDS = {
    "episode": ("int16", "episode number in this situation (column `episode` of episodes.parquet)"),
    "step": ("int16", "control step since the start of the episode (50 Hz)"),
    "t_s": ("float32", "time since the start of the episode, at the end of this step [s]"),
    "obs": ("float32", "the 61 observations fed to the policy at this step (layout in obs_layout)"),
    "h1": ("float16", "activation of hidden layer 1 (after ELU), 512 numbers"),
    "h2": ("float16", "activation of hidden layer 2 (after ELU), 256 numbers"),
    "h3": ("float16", "activation of hidden layer 3 (after ELU), 128 numbers"),
    "action": ("float32", "the 14 outputs of the policy: motor targets = default pose + action"),
    "lin_vel_b": ("float32", "trunk linear velocity in the robot frame, after the step [m/s]"),
    "ang_vel_b": ("float32", "trunk angular velocity in the robot frame, after the step [rad/s]"),
    "gravity_b": ("float32", "gravity direction in the robot frame (tilt), after the step"),
    "foot_contact": ("bool", "left / right foot touching the terrain"),
    "foot_contact_time": ("float32", "time each foot has been on the ground [s] (0 in the air)"),
    "foot_air_time": ("float32", "time each foot has been in the air [s] (0 on the ground)"),
    "foot_height": ("float32", "height of the lowest point of each sole above the terrain under it [m]"),
    "torque": ("float32", "torques of the 14 motors at the last physics step [Nm]"),
    "joint_vel": ("float32", "velocities of the 14 joints [rad/s]"),
    "power": ("float32", "mechanical motor power, sum of |torque x joint velocity|, mean over the step [W]"),
    "command": ("float32", "velocity command fed to the policy: vx, vy [m/s], yaw rate [rad/s] (heading hold)"),
    "terrain_kind": ("int8", "index in terrain_kinds"),
    "terrain_value": ("float32", "size of the terrain: obstacle height or wave amplitude [cm], or slope [deg]"),
    "terrain_direction": ("int8", "0 none, 1 down / downhill, 2 up / uphill"),
    "ground_slope_deg": ("float32", "measured slope of the ground under the trunk along the heading (+ = uphill) [deg]"),
    "ground_cross_slope_deg": ("float32", "measured slope of the ground under the trunk across the heading (+ = rising to the left) [deg]"),
    "friction": ("float32", "current friction coefficient of feet and floor"),
    "payload_kg": ("float32", "payload carried at this step [kg] (0 before it appears)"),
    "com_shift_mm": ("float32", "shift of the robot's centre of mass caused by the payload, x fwd / y left / z up [mm]"),
    "event_active": ("bool", "True from the event of a transition situation onwards"),
    "vin_v": ("float32", "battery voltage of the episode [V]"),
    "falls_within_2s": ("bool", "True if the episode ends with a fall within the next fall_horizon_s (2 s)"),
}
OBS_LAYOUT = {"base_ang_vel": [0, 3], "projected_gravity": [3, 6], "joint_pos_rel": [6, 20], "joint_vel": [20, 34],
              "last_action": [34, 48], "twist_command": [48, 51], "head_command": [51, 55], "body_command": [55, 61]}


class StepRecorder:
    """Rows of all the episodes of one (controller, situation), column by column."""

    def __init__(self):
        self.columns: dict[str, list] = {}
        self.episode = 0
        self._first_row = 0

    def __len__(self):
        return len(self.columns.get("step", []))

    def start_episode(self, episode: int):
        self.episode, self._first_row = episode, len(self)

    def add(self, row: dict):
        for key, value in {"episode": self.episode, **row}.items():
            self.columns.setdefault(key, []).append(value)

    def end_episode(self, last_step: int, fell: bool, horizon_steps: int):
        steps = self.columns.get("step", [])[self._first_row:]
        self.columns.setdefault("falls_within_2s", []).extend(fall_labels(steps, last_step, fell, horizon_steps).tolist())

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.asarray(v, dtype=STEP_FIELDS[k][0]) for k, v in self.columns.items()}
        np.savez_compressed(path, **arrays)
        return path


def write_fields_json(path: Path, control_dt: float, fall_horizon_s: float):
    """Meaning and type of every field, observation layout, joint order and terrain codes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json({"fields": {k: {"dtype": v[0], "description": v[1]} for k, v in STEP_FIELDS.items()},
               "obs_layout": OBS_LAYOUT, "terrain_kinds": TERRAIN_KINDS, "joint_order": JOINT_ORDER,
               "default_pose": DEFAULT_POSE.tolist(), "control_dt_s": control_dt, "fall_horizon_s": fall_horizon_s,
               "files": "<controller>/<situation>.npz, one row per control step of every episode"}, path)
