"""The simulation loop: episodes of ANY Controller in a situation, with their metrics; optionally the
per-step data (plan sections 4.8-4.9), a video of the first episode, or a still of the start."""
from __future__ import annotations

import logging
import math
import multiprocessing as mp
import time
from pathlib import Path

import mujoco
import numpy as np

from steerduck.config import save_json, setup_logging, situation_seed
from steerduck.policy import build_controller
from steerduck.scene import (DEFAULT_POSE, JOINT_ORDER, TERRAINS, Robot, Scenario, build_model, robot_com_in_trunk,
                             start_pose)

log = logging.getLogger("steerduck")

# Per-step data: field -> (dtype in the .npz files, meaning)
STEP_FIELDS = {
    "episode": ("int16", "episode number (column `episode` of episodes.parquet)"),
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
    "ground_cross_slope_deg": ("float32", "measured slope across the heading (+ = rising to the left) [deg]"),
    "friction": ("float32", "current friction coefficient of feet and floor"),
    "payload_kg": ("float32", "payload carried at this step [kg] (0 before it appears)"),
    "com_shift_mm": ("float32", "shift of the robot's centre of mass caused by the payload, x fwd / y left / z up [mm]"),
    "event_active": ("bool", "True from the event of a transition onwards"),
    "vin_v": ("float32", "battery voltage of the episode [V]"),
    "falls_within_2s": ("bool", "True if the episode ends with a fall within the next fall_horizon_s"),
}
OBS_LAYOUT = {"base_ang_vel": [0, 3], "projected_gravity": [3, 6], "joint_pos_rel": [6, 20], "joint_vel": [20, 34],
              "last_action": [34, 48], "twist_command": [48, 51], "head_command": [51, 55], "body_command": [55, 61]}


def control_dt(conf: dict) -> float:
    return conf["decimation"] * conf["physics_dt_s"]


def write_fields_json(path: Path, conf: dict):
    """Meaning and type of every per-step field, observation layout, joint order, terrain codes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json({"fields": {k: {"dtype": t, "description": d} for k, (t, d) in STEP_FIELDS.items()},
               "obs_layout": OBS_LAYOUT, "terrain_kinds": TERRAINS, "joint_order": JOINT_ORDER,
               "default_pose": DEFAULT_POSE.tolist(), "control_dt_s": control_dt(conf),
               "fall_horizon_s": conf["fall_horizon_s"],
               "files": "<controller>/<situation>.npz, one row per control step of every episode"}, path)


# ---------------------------------------------------------------------------
# Metrics of one episode
# ---------------------------------------------------------------------------
def walking_metrics(trace: dict, conf: dict, event_step=None) -> dict:
    """From the per-step trace (vx, vy, wz, power), after the warm-up: velocity error |v_xy - command|,
    instantaneous and on 1 s windows (the tracking without the in-step sway); speed along the robot's
    forward axis; yaw rate; motor power. After an event, the same from 1 s after it."""
    vx, vy, wz, power = (np.asarray(trace[k]) for k in ("vx", "vy", "wz", "power"))
    dt = control_dt(conf)
    cmd = conf["command_vx_m_s"]
    after = slice(round(conf["warmup_s"] / dt), None)
    win = round(1.0 / dt)
    n_win = len(vx[after]) // win
    mean = lambda a: float(a.mean()) if len(a) else None  # noqa: E731
    out = {
        "vel_error_m_s": mean(np.hypot(vx[after] - cmd, vy[after])),
        "vel_error_1s_m_s": mean(np.hypot(vx[after][:n_win * win].reshape(n_win, win).mean(1) - cmd,
                                          vy[after][:n_win * win].reshape(n_win, win).mean(1))),
        "path_speed_m_s": mean(vx[after]), "yaw_rate_rad_s": mean(wz[after]), "power_w": mean(power[after]),
    }
    if event_step is not None:
        after_event = slice(event_step + win, None)
        out["vel_error_after_event_m_s"] = mean(np.hypot(vx[after_event] - cmd, vy[after_event]))
        out["power_after_event_w"] = mean(power[after_event])
    return out


# ---------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------
def run_episode(robot: Robot, controller, conf: dict, s: dict, origin, size, rng, robot_mass,
                video=None, steps: dict | None = None, tag: str = "") -> dict:
    """Run one episode -> its results. The draws come from `rng` in a fixed order (start, battery
    voltage, joint noise), so that every controller gets the same episode. With action_noise_std > 0 a
    Gaussian noise is added to every action, as PPO's exploration does: it is drawn after the start, so
    the episode itself does not change. If `steps` is given, one row per control step is appended."""
    d, dt = robot.data, control_dt(conf)
    x, y, yaw0 = start_pose(s, origin, size, rng, conf)
    vin = rng.uniform(*conf["vin_range"])
    noise = rng.uniform(-conf["init_joint_noise_rad"], conf["init_joint_noise_rad"], len(DEFAULT_POSE))
    robot.place(s, x, y, yaw0, vin, noise, conf["spawn_height_m"])
    scenario = Scenario(s, robot, robot_mass, dt)
    controller.reset()

    episode_s = s.get("episode_s", conf["episode_s"])
    n_steps, log_every = round(episode_s / dt), round(conf["log_every_s"] / dt)
    hold, cos_fall = conf["heading_hold"], math.cos(math.radians(conf["fall_tilt_deg"]))
    command = np.zeros(13, dtype=np.float32)
    command[0] = conf["command_vx_m_s"]
    action = np.zeros(len(DEFAULT_POSE), dtype=np.float32)
    start = robot.position_xy()
    trace = {k: [] for k in ("vx", "vy", "wz", "power")}
    outcome, step, max_tilt = "completed", 0, 0.0
    contact_time, air_time, first_row = np.zeros(2), np.zeros(2), len(steps["step"]) if steps else 0

    for step in range(n_steps):
        if scenario.update(step):
            log.info(f"      {tag} t={step * dt:.1f} s: event {s['event']} applied")
        if hold["enabled"]:   # mjlab-style heading controller: yaw-rate command from the heading error
            err = math.atan2(math.sin(yaw0 - robot.yaw()), math.cos(yaw0 - robot.yaw()))
            command[2] = float(np.clip(hold["stiffness"] * err, -hold["max_rad_s"], hold["max_rad_s"]))
        obs = robot.observation(action, command)
        action = controller.act(obs)
        if conf["action_noise_std"]:   # the noisy action is also the last action the policy observes
            action = (action + rng.normal(0.0, conf["action_noise_std"], action.shape)).astype(np.float32)
        power = sum(robot.step(DEFAULT_POSE + conf["action_scale"] * action) for _ in range(conf["decimation"]))
        power /= conf["decimation"]
        if video is not None:
            video.capture(d, step)
        if not np.all(np.isfinite(d.qpos)):
            outcome = "nan"
            break
        v, gravity = robot.lin_vel_b(), robot.gravity_b()
        tilt = math.degrees(math.acos(float(np.clip(-gravity[2], -1.0, 1.0))))
        max_tilt = max(max_tilt, tilt)
        for k, value in zip(trace, (v[0], v[1], d.sensordata[robot.gyro_adr + 2], power)):
            trace[k].append(float(value))
        if steps is not None:
            # the observation the policy received, its action, and the state reached after the action
            contact, foot_height = robot.feet_state()
            contact_time = np.where(contact, contact_time + dt, 0.0)
            air_time = np.where(contact, 0.0, air_time + dt)
            slope, cross_slope = robot.ground_slope()
            row = {"episode": steps["_episode"], "step": step, "t_s": (step + 1) * dt, "obs": obs, "action": action,
                   "lin_vel_b": v, "ang_vel_b": robot.ang_vel_b(), "gravity_b": gravity, "foot_contact": contact,
                   "foot_contact_time": contact_time.copy(), "foot_air_time": air_time.copy(), "foot_height": foot_height,
                   "torque": d.actuator_force.copy(), "joint_vel": d.qvel[robot.dof_idx].copy(), "power": power,
                   "command": command[:3].copy(), "ground_slope_deg": slope, "ground_cross_slope_deg": cross_slope,
                   "vin_v": vin, **scenario.fields(step), **controller.activations()}
            for k, value in row.items():
                steps.setdefault(k, []).append(value)
        if -gravity[2] < cos_fall:
            outcome = "fell"
            break
        if log_every and (step + 1) % log_every == 0:
            log.info(f"      {tag} sim {(step + 1) * dt:5.1f}/{episode_s:.0f} s | tilt {tilt:4.1f} deg | "
                     f"vx {v[0]:+.2f} m/s | {np.linalg.norm(robot.position_xy() - start):.2f} m")

    duration = (step + 1) * dt
    if steps is not None:   # outcome label: the rows less than fall_horizon_s before a fall
        rows = np.asarray(steps["step"][first_row:])
        horizon = round(conf["fall_horizon_s"] / dt)
        steps.setdefault("falls_within_2s", []).extend((rows > step - horizon) & (outcome == "fell"))
    end = robot.position_xy()
    res = {"outcome": outcome, "duration_s": duration, "fall_time_s": duration if outcome == "fell" else None,
           **walking_metrics(trace, conf, scenario.event_step), "max_tilt_deg": max_tilt,
           "distance_m": float(np.linalg.norm(end - start)),
           "start_x": float(start[0] - origin[0]), "start_y": float(start[1] - origin[1]),
           "end_x": float(end[0] - origin[0]), "end_y": float(end[1] - origin[1]),
           "heading_deg": math.degrees(yaw0), "vin_v": float(vin), **controller.episode_diagnostics()}
    if "event" in s:
        res["fell_after_event"] = outcome == "fell" and scenario.after_event(step)
    vel = res["vel_error_1s_m_s"]
    log.info(f"      {tag} -> {outcome.upper()} at {duration:.1f}/{episode_s:.0f} s | max tilt {max_tilt:.0f} deg | "
             f"vel err {vel if vel is None else round(vel, 3)} m/s")
    return res


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
class Video:
    """Frames of one episode from a camera tracking the trunk (needs MUJOCO_GL=egl on a headless machine)."""

    def __init__(self, model, conf: dict):
        width, height = conf["video_size"]
        self.renderer = mujoco.Renderer(model, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = model.body("trunk_base").id
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 0.9, -20.0, 135.0
        self.every, self.frames = conf["video_every_n_steps"], []

    def capture(self, data, step):
        if step % self.every == 0:
            self.renderer.update_scene(data, camera=self.cam)
            self.frames.append(self.renderer.render().copy())

    def save(self, path, fps):
        import imageio
        imageio.mimsave(path, self.frames, fps=fps)
        self.renderer.close()


def start_still(conf: dict, s: dict, seed: int, com_trunk, robot_mass):
    """Image of the robot at the start of episode 0 of a situation, seen from behind and above."""
    model, origin, size = build_model(conf, s, situation_seed(s["name"], seed), com_trunk, robot_mass)
    robot = Robot(model, conf)
    x, y, yaw = start_pose(s, origin, size, np.random.default_rng([situation_seed(s["name"], seed), 0]), conf)
    z = robot.place(s, x, y, yaw, conf["actuator"]["vin"], np.zeros(len(DEFAULT_POSE)), conf["spawn_height_m"])
    renderer = mujoco.Renderer(model, 180, 240)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [x, y, z - 0.1]
    cam.distance, cam.elevation, cam.azimuth = 2.0, -25.0, math.degrees(yaw) + 150.0
    renderer.update_scene(robot.data, camera=cam)
    image = renderer.render().copy()
    renderer.close()
    return image


# ---------------------------------------------------------------------------
# All the episodes of one (controller, situation, seed), and of many in parallel
# ---------------------------------------------------------------------------
def run_situation(task: dict) -> list[dict]:
    """task: conf, spec (controller), situation, seed, run_dir, steps_dir (None = no per-step data)."""
    conf, spec, s, seed = task["conf"], task["spec"], task["situation"], task["seed"]
    name, n_episodes, t0 = s["name"], conf["episodes_per_situation"], time.time()
    com_trunk, robot_mass = robot_com_in_trunk(conf)
    model, origin, size = build_model(conf, s, situation_seed(name, seed), com_trunk, robot_mass)
    robot = Robot(model, conf)
    controller = build_controller(spec, record_activations=bool(task["steps_dir"]) and conf["record_activations"])
    steps = {"step": []} if task["steps_dir"] else None
    log.info(f"[{spec['name']}] {name} ({s['group']}), seed {seed}: {n_episodes} episodes")
    episodes = []
    for k in range(n_episodes):
        rng = np.random.default_rng([situation_seed(name, seed), k])   # same draws for every controller
        video = None
        if conf["video"] and k == 0 and (not conf["video_situations"] or name in conf["video_situations"]):
            try:
                video = Video(model, conf)
            except Exception as e:  # noqa: BLE001  rendering is optional
                log.info(f"[WARNING] no video, the renderer could not start ({e}); is MUJOCO_GL=egl set?")
        if steps is not None:
            steps["_episode"] = k
        result = run_episode(robot, controller, conf, s, origin, size, rng, robot_mass, video, steps,
                             tag=f"[{spec['name']}] {name} ep {k + 1}/{n_episodes}")
        episodes.append({"controller": spec["name"], "situation": name, "group": s["group"], "seed": seed,
                         "episode": k, "action_noise_std": conf["action_noise_std"], **result})
        if video is not None:
            (Path(task["run_dir"]) / "videos").mkdir(exist_ok=True)
            video.save(Path(task["run_dir"]) / "videos" / f"{spec['name']}__{name}__seed{seed}.mp4",
                       fps=1.0 / (control_dt(conf) * conf["video_every_n_steps"]))
    if steps is not None:
        path = Path(task["steps_dir"]) / spec["name"] / f"{name}__seed{seed}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: np.asarray(v, dtype=STEP_FIELDS[k][0]) for k, v in steps.items()
                                     if not k.startswith("_")})
        log.info(f"[{spec['name']}] {name}: {len(steps['step'])} steps -> {path} ({path.stat().st_size / 1e6:.1f} MB)")
    falls = sum(e["outcome"] == "fell" for e in episodes)
    log.info(f"[{spec['name']}] {name} done: falls {falls}/{n_episodes} ({time.time() - t0:.0f} s)")
    return episodes


def run_all(tasks: list[dict], workers: int, log_file=None) -> list[dict]:
    """Run the tasks, `workers` processes in parallel (1 = in this process) -> all the episodes."""
    if workers == 1:
        return [e for t in tasks for e in run_situation(t)]
    with mp.Pool(workers, initializer=setup_logging, initargs=(log_file,)) as pool:
        return [e for episodes in pool.imap_unordered(run_situation, tasks) for e in episodes]
