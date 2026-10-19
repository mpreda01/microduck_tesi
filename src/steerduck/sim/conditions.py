"""Situations of configs/situations.yaml: selection by suite, start of the episodes, and what changes
during an episode (friction, payload, events)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from steerduck import paths
from steerduck.io import load_config
from steerduck.sim.scene import set_friction
from steerduck.sim.terrains import DIRECTIONS, TERRAIN_KINDS, terrain_value


# ---------------------------------------------------------------------------
# Situations and suites
# ---------------------------------------------------------------------------
def load_situations(path: str | Path | None = None):
    """-> (list of situations in file order, dict of suites)."""
    conf = load_config(path or paths.CONFIGS / "situations.yaml")
    situations = conf["situations"]
    names = [s["name"] for s in situations]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"situation names must be unique: {duplicates}")
    return situations, conf.get("suites", {})


def select(situations: list, suites: dict, suite: str = "all", only=None):
    """-> [(index, situation)] of a suite, optionally restricted to the names in `only`. The index is the
    position in situations.yaml: it seeds the terrain and the episodes, so a subset run gives the same
    episodes as the full run."""
    if suite == "all":
        chosen = {s["name"] for s in situations}
    elif suite not in suites:
        raise ValueError(f"unknown suite {suite!r}; known: all, {', '.join(suites)}")
    elif isinstance(suites[suite], dict):
        groups = set(suites[suite]["groups"])
        chosen = {s["name"] for s in situations if s["group"] in groups}
    else:
        chosen = set(suites[suite])
    unknown = sorted((chosen | set(only or [])) - {s["name"] for s in situations})
    if unknown:
        raise ValueError(f"unknown situations: {unknown}")
    if only:
        chosen &= set(only)
    return [(i, s) for i, s in enumerate(situations) if s["name"] in chosen]


# ---------------------------------------------------------------------------
# Start of an episode
# ---------------------------------------------------------------------------
def start_pose(situation: dict, origin, size, rng, conf: dict):
    """(x, y, yaw) of the start; the heading hold keeps the start yaw."""
    t = situation["terrain"]
    jitter = math.radians(conf["heading_jitter_deg"])
    lat = conf["start_lateral_jitter_m"]
    kind, direction = t["kind"], t.get("direction")
    if kind == "face_slope":   # middle of the +x face (planar), facing the edge (downhill) or the top
        d = t.get("start_m", size[0] / 4)
        yaw = (0.0 if direction == "downhill" else math.pi) + rng.uniform(-jitter, jitter)
        return origin[0] + d, origin[1] + rng.uniform(-lat, lat), yaw
    if kind == "stairs" and direction == "up":   # near the edge of the patch, facing the top
        return origin[0] + size[0] / 2 - 0.35, origin[1] + rng.uniform(-lat, lat), math.pi + rng.uniform(-jitter, jitter)
    if kind == "strip":        # on the first segment, walking towards +x along the profile
        return origin[0] + t.get("start_m", 0.6), origin[1] + rng.uniform(-lat, lat), rng.uniform(-jitter, jitter)
    return origin[0], origin[1], rng.uniform(-math.pi, math.pi)   # centre of the patch, random heading


def spawn_radius(situation: dict) -> float:
    """Radius of the highest-ground search under the start. Smaller on the long slopes, otherwise the
    uphill side of the circle would lift the start (10 cm x tan 15 deg = 2.7 cm above the surface)."""
    return 0.05 if situation["terrain"]["kind"] in ("face_slope", "strip") else 0.1


# ---------------------------------------------------------------------------
# What changes during an episode
# ---------------------------------------------------------------------------
class Scenario:
    """State of the situation during one episode: applies the friction at the start and the event at its
    step, and gives the scenario fields of every recorded step (plan section 4.8)."""

    def __init__(self, situation: dict, robot, robot_mass: float, control_dt: float):
        self.situation, self.model, self.robot_mass = situation, robot.model, robot_mass
        t = situation["terrain"]
        self.static = {"terrain_kind": TERRAIN_KINDS.index(t["kind"]), "terrain_value": terrain_value(t),
                       "terrain_direction": DIRECTIONS[t.get("direction")]}
        self.event = situation.get("event")
        self.event_step = int(round(self.event["t_s"] / control_dt)) if self.event else None
        friction = situation.get("friction")
        if friction is not None:
            set_friction(self.model, friction)
        self.friction = friction if friction is not None else float(robot.geom_friction0[robot.feet[0], 0])
        payload = situation.get("payload")
        self.payload_kg = payload["mass_frac"] * robot_mass if payload else 0.0
        self.com_shift_mm = (np.asarray(payload.get("offset_m", [0, 0, 0])) * 1000 * self.payload_kg
                             / (robot_mass + self.payload_kg) if payload else np.zeros(3))
        self.payload_on = bool(payload) and not (self.event and self.event["type"] == "payload")

    def update(self, step: int) -> bool:
        """Apply the event if it happens at this control step; True when it was applied."""
        if not self.event or step != self.event_step:
            return False
        m = self.model
        if self.event["type"] == "friction":
            set_friction(m, self.event["value"])
            self.friction = self.event["value"]
        elif self.event["type"] == "payload":
            pid = m.body("payload").id
            # mass and inertia enter the dynamics at every step (mj_setConst would overwrite the state)
            m.body_mass[pid] = self.payload_kg
            m.body_inertia[pid] = 0.4 * self.payload_kg * 0.02 ** 2   # solid sphere, 2 cm radius
            self.payload_on = True
        else:
            raise ValueError(f"unknown event type {self.event['type']!r}")
        return True

    def after_event(self, step: int) -> bool:
        return bool(self.event) and step >= self.event_step

    def fields(self, step: int) -> dict:
        return {**self.static, "friction": self.friction,
                "payload_kg": self.payload_kg if self.payload_on else 0.0,
                "com_shift_mm": self.com_shift_mm if self.payload_on else np.zeros(3),
                "event_active": self.after_event(step)}
