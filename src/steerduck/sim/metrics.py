"""Metrics of one episode (plan section 5): defined here only. Averages over episodes and seeds are
computed later, in analysis/aggregate.py."""
from __future__ import annotations

import math

import numpy as np

TRACE_KEYS = ("vx", "vy", "wz", "power", "tilt")


def tilt_deg(gravity_b) -> float:
    """Tilt of the trunk from vertical [deg]."""
    return math.degrees(math.acos(float(np.clip(-gravity_b[2], -1.0, 1.0))))


def _mean(a):
    return float(a.mean()) if len(a) else None


def walking_metrics(trace: dict, command_vx: float, control_dt: float, warmup_s: float,
                    event_step: int | None = None) -> dict:
    """Walking metrics from the per-step trace of an episode (lists vx, vy, wz, power, tilt), after the
    warm-up: velocity error |v_xy - command| (instantaneous, and on 1 s windows: the tracking without the
    in-step sway), speed along the robot's forward axis, yaw rate, mechanical motor power. After an
    event, the same metrics from 1 s after it."""
    r = {k: np.asarray(v) for k, v in trace.items()}
    warmup = int(round(warmup_s / control_dt))
    sl = slice(warmup, None)
    cmd = np.array([command_vx, 0.0])
    err = np.hypot(r["vx"][sl] - cmd[0], r["vy"][sl] - cmd[1]) if len(r["vx"]) > warmup else np.array([])
    win = max(1, int(round(1.0 / control_dt)))
    n_win = len(r["vx"][sl]) // win
    err_1s = (np.hypot(r["vx"][sl][:n_win * win].reshape(n_win, win).mean(1) - cmd[0],
                       r["vy"][sl][:n_win * win].reshape(n_win, win).mean(1) - cmd[1]) if n_win else np.array([]))
    out = {
        "vel_error_m_s": _mean(err), "vel_error_1s_m_s": _mean(err_1s),
        "path_speed_m_s": _mean(r["vx"][sl]), "yaw_rate_rad_s": _mean(r["wz"][sl]),
        "power_w": _mean(r["power"][sl]),
    }
    if event_step is not None:
        after = slice(event_step + win, None)   # 1 s after the event
        out["vel_error_after_event_m_s"] = _mean(np.hypot(r["vx"][after] - cmd[0], r["vy"][after] - cmd[1]))
        out["power_after_event_w"] = _mean(r["power"][after])
    return out


def fall_labels(steps_of_episode, last_step: int, fell: bool, horizon_steps: int):
    """Per-step outcome label (plan 4.8): True on the steps within the horizon before a fall."""
    steps_of_episode = np.asarray(steps_of_episode)
    if not fell:
        return np.zeros(len(steps_of_episode), dtype=bool)
    return steps_of_episode > last_step - horizon_steps
