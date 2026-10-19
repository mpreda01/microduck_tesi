"""The simulation loop: episodes of ANY Controller in a situation, optionally recording every step.
Used by evaluation, data collection (and, later, the DAgger rollouts)."""
from __future__ import annotations

import logging
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from steerduck.io import setup_logging
from steerduck.policies.registry import build_controller
from steerduck.sim.conditions import Scenario, spawn_radius, start_pose
from steerduck.sim.metrics import TRACE_KEYS, tilt_deg, walking_metrics
from steerduck.sim.recording import StepRecorder
from steerduck.sim.render import Video
from steerduck.sim.robot import Robot
from steerduck.sim.scene import DEFAULT_POSE, build_model, ground_height, robot_com_in_trunk

log = logging.getLogger("steerduck")


def control_dt(conf: dict) -> float:
    return conf["decimation"] * conf["physics_dt_s"]


def place_robot(robot: Robot, situation: dict, x, y, yaw, vin, joint_noise, conf: dict) -> float:
    """Put the robot at (x, y, yaw), its feet spawn_height_m above the highest ground under it -> trunk z."""
    robot.reset(x, y, 100.0, yaw, vin, joint_noise)          # robot out of the way of the ground rays
    z = ground_height(robot.model, robot.data, x, y, spawn_radius(situation)) + conf["spawn_height_m"]
    robot.reset(x, y, z, yaw, vin, joint_noise)
    return z


def run_episode(robot: Robot, controller, conf: dict, situation: dict, origin, size, rng, robot_mass: float,
                video: Video | None = None, recorder: StepRecorder | None = None, tag: str = "") -> dict:
    """Run one episode -> its results. The draws (start, battery voltage, joint noise) come from `rng`
    in a fixed order, so that every controller gets the same episode."""
    m, d = robot.model, robot.data
    x, y, yaw0 = start_pose(situation, origin, size, rng, conf)
    vin = rng.uniform(*conf["vin_range"]) if conf.get("vin_range") else conf["actuator"]["vin"]
    noise = rng.uniform(-conf["init_joint_noise_rad"], conf["init_joint_noise_rad"], 14)
    place_robot(robot, situation, x, y, yaw0, vin, noise, conf)
    controller.reset()

    decimation, dt = conf["decimation"], control_dt(conf)
    scenario = Scenario(situation, robot, robot_mass, dt)   # after the reset: sets the friction
    episode_s = situation.get("episode_s", conf["episode_s"])
    n_steps = int(round(episode_s / dt))
    log_every = int(round(conf.get("log_every_s", 0) / dt))
    hold = conf["heading_hold"]
    cos_fall = math.cos(math.radians(conf["fall_tilt_deg"]))

    command = np.zeros(13, dtype=np.float32)
    command[0] = conf["command_vx_m_s"]
    last_action = np.zeros(14, dtype=np.float32)
    start = robot.position_xy()
    trace = {k: [] for k in TRACE_KEYS}
    outcome, step, max_tilt = "completed", 0, 0.0
    contact_time, air_time = np.zeros(2), np.zeros(2)

    for step in range(n_steps):
        if scenario.update(step):
            log.info(f"      {tag} t={step * dt:.1f} s: event {scenario.event['type']} applied")
        if hold["enabled"]:  # mjlab-style heading controller: yaw-rate command from the heading error
            err = math.atan2(math.sin(yaw0 - robot.yaw()), math.cos(yaw0 - robot.yaw()))
            command[2] = float(np.clip(hold["stiffness"] * err, -hold["max_rad_s"], hold["max_rad_s"]))
        obs = robot.observation(last_action, command)
        action = controller.act(obs)
        last_action = action
        target = DEFAULT_POSE + conf["action_scale"] * action
        p = 0.0
        for _ in range(decimation):
            p += robot.step(target)
        if video is not None:
            video.capture(d, step)
        if not np.all(np.isfinite(d.qpos)):
            outcome = "nan"
            break
        v = robot.lin_vel_b()
        gravity = robot.gravity_b()
        tilt = tilt_deg(gravity)
        max_tilt = max(max_tilt, tilt)
        for k, val in (("vx", v[0]), ("vy", v[1]), ("wz", d.sensordata[robot.gyro_adr + 2]),
                       ("power", p / decimation), ("tilt", tilt)):
            trace[k].append(float(val))
        if recorder is not None:
            # the observation the policy received, its action, and the state reached after the action
            contact, foot_height = robot.feet_state()
            contact_time = np.where(contact, contact_time + dt, 0.0)
            air_time = np.where(contact, 0.0, air_time + dt)
            slope_along, slope_across = robot.ground_slope()
            recorder.add({
                "step": step, "t_s": (step + 1) * dt, "obs": obs, "action": action,
                "lin_vel_b": v, "ang_vel_b": robot.ang_vel_b(), "gravity_b": gravity,
                "foot_contact": contact, "foot_contact_time": contact_time.copy(), "foot_air_time": air_time.copy(),
                "foot_height": foot_height, "torque": d.actuator_force.copy(), "joint_vel": d.qvel[robot.dof_idx].copy(),
                "power": p / decimation, "command": command[:3].copy(),
                "ground_slope_deg": slope_along, "ground_cross_slope_deg": slope_across,
                **scenario.fields(step), "vin_v": vin,
                **(controller.activations() or {}),
            })
        if -gravity[2] < cos_fall:
            outcome = "fell"
            break
        if log_every and (step + 1) % log_every == 0:
            log.info(f"      {tag} sim {(step + 1) * dt:5.1f}/{episode_s:.0f} s | tilt {tilt:4.1f} deg | "
                     f"vx {v[0]:+.2f} m/s | {np.linalg.norm(robot.position_xy() - start):.2f} m")

    duration = (step + 1) * dt
    if recorder is not None:
        recorder.end_episode(step, outcome == "fell", int(round(conf["fall_horizon_s"] / dt)))
    pos = robot.position_xy()
    res = {
        "outcome": outcome, "duration_s": duration,
        "fall_time_s": duration if outcome == "fell" else None,
        **walking_metrics(trace, conf["command_vx_m_s"], dt, conf["warmup_s"], scenario.event_step),
        "max_tilt_deg": max_tilt,
        "distance_m": float(np.linalg.norm(pos - start)),
        "start_x": float(start[0] - origin[0]), "start_y": float(start[1] - origin[1]),
        "end_x": float(pos[0] - origin[0]), "end_y": float(pos[1] - origin[1]),
        "heading_deg": math.degrees(yaw0), "vin_v": float(vin),
        **controller.episode_diagnostics(),
    }
    if scenario.event:
        res["fell_after_event"] = outcome == "fell" and scenario.after_event(step)
    vel = res["vel_error_1s_m_s"]
    log.info(f"      {tag} -> {outcome.upper()} at {duration:.1f}/{episode_s:.0f} s | max tilt {max_tilt:.0f} deg | "
             f"vel err {vel if vel is None else round(vel, 3)} m/s")
    return res


# ---------------------------------------------------------------------------
# One task = all the episodes of one (controller, situation)
# ---------------------------------------------------------------------------
def run_task(task: dict) -> list[dict]:
    """task: conf, spec (controller), index and situation, n_situations, run_dir, steps_dir (or None)."""
    conf, spec, index, situation = task["conf"], task["spec"], task["index"], task["situation"]
    name, n_episodes, seed = situation["name"], conf["episodes_per_situation"], conf["seed"]
    t0 = time.time()
    com_trunk, robot_mass = robot_com_in_trunk(conf)
    model, origin, size = build_model(conf, situation, seed + index, com_trunk, robot_mass)
    robot = Robot(model, conf)
    record = conf["record"]
    controller = build_controller(spec, record_activations=record["steps"] and record["activations"])
    recorder = StepRecorder() if task["steps_dir"] else None
    vconf = conf["video"]
    log.info(f"[{spec['name']}] situation {task['position']}/{task['n_situations']}: {name} ({situation['group']}) - "
             f"{n_episodes} episodes")
    episodes = []
    for k in range(n_episodes):
        rng = np.random.default_rng([seed, index, k])   # same draws for every controller (paired)
        video = None
        if vconf["enabled"] and k == 0 and (not vconf["situations"] or name in vconf["situations"]):
            try:
                video = Video(model, vconf)
            except Exception as e:  # noqa: BLE001  rendering is optional
                log.info(f"[WARNING] no video, the renderer could not start ({e}); is MUJOCO_GL=egl set?")
        if recorder is not None:
            recorder.start_episode(k)
        ep = run_episode(robot, controller, conf, situation, origin, size, rng, robot_mass, video, recorder,
                         tag=f"[{spec['name']}] {name} ep {k + 1}/{n_episodes}")
        episodes.append({"controller": spec["name"], "situation": name, "group": situation["group"],
                         "situation_index": index, "seed": seed, "episode": k, **ep})
        if video is not None:
            out = Path(task["run_dir"]) / "videos"
            out.mkdir(exist_ok=True)
            video.save(out / f"{spec['name']}__{name}.mp4", fps=1.0 / (control_dt(conf) * vconf["every_n_steps"]))
    if recorder is not None:
        path = recorder.save(Path(task["steps_dir"]) / spec["name"] / f"{name}.npz")
        log.info(f"[{spec['name']}] {name}: {len(recorder)} steps saved to {path} ({path.stat().st_size / 1e6:.1f} MB)")
    falls = sum(e["outcome"] == "fell" for e in episodes)
    log.info(f"[{spec['name']}] {name} done: falls {falls}/{len(episodes)} ({time.time() - t0:.0f} s)")
    return episodes


def run_tasks(tasks: list[dict], workers: int, log_file=None) -> list[dict]:
    """Run the tasks, one process each (`workers` in parallel; 1 = in this process) -> all the episodes."""
    episodes = []
    if workers == 1:
        for t in tasks:
            episodes += run_task(t)
        return episodes
    with mp.Pool(workers, initializer=setup_logging, initargs=(log_file,)) as pool:
        for eps in pool.imap_unordered(run_task, tasks):
            episodes += eps
    return episodes
