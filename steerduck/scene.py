"""A situation of situations.yaml -> a MuJoCo model: terrain, robot, payload, friction, bam servos.
Also the robot's observations and measured state, and what changes during an episode."""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains import heightfield_terrains as hf
from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg, BoxPyramidStairsTerrainCfg, BoxRandomGridTerrainCfg
from mjlab.terrains.terrain_generator import (SubTerrainCfg, TerrainGenerator, TerrainGeneratorCfg,
                                              TerrainGeometry, TerrainOutput)

from steerduck.config import robot_dir

# Default pose of scripts/infer_policy.py (STAND keyframe of scene.xml), in actuator order:
# actions are offsets from it, joint observations are relative to it.
DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,    # left leg: hip yaw, hip roll, hip pitch, knee, ankle
    0.3491, 0.3491, 0.0, 0.0,                  # neck pitch, head pitch, head yaw, head roll
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,      # right leg
], dtype=np.float32)
JOINT_ORDER = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
               "neck_pitch", "head_pitch", "head_yaw", "head_roll",
               "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")
TERRAINS = ["flat", "grid", "stairs", "rubble", "waves", "slope", "face_slope", "strip"]   # codes of terrain_kind
DIRECTIONS = {None: 0, "down": 1, "downhill": 1, "up": 2, "uphill": 2}                     # codes of terrain_direction


# ---------------------------------------------------------------------------
# Terrains
# ---------------------------------------------------------------------------
def add_hfield(spec, heights, size_x, size_y, vertical_scale):
    """Heightfield geom on the terrain body, built like mjlab's. heights (in vertical_scale units) has
    shape (n_y, n_x): MuJoCo rows run along y, and the samples of each axis span the whole size."""
    noise = np.rint(heights).astype(np.int16)
    lo = int(noise.min())
    span = int(noise.max()) - lo or 1
    height = span * vertical_scale
    norm = (noise - lo) / span
    uid = uuid.uuid4().hex
    field = spec.add_hfield(name=f"hfield_{uid}", size=[size_x / 2, size_y / 2, height, height],
                            nrow=noise.shape[0], ncol=noise.shape[1], userdata=norm.flatten().astype(np.float32).tolist())
    geom = spec.body("terrain").add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=field.name,
                                         pos=[size_x / 2, size_y / 2, 0.0],
                                         material=hf.color_by_height(spec, noise, uid, norm))
    return geom, field, height


@dataclass(kw_only=True)
class FlatFacePyramidTerrainCfg(SubTerrainCfg):
    """Pyramid with four PLANAR faces. mjlab's pyramid is H * tent(x) * tent(y): its creases run along the
    centre lines, so a robot walking straight up or down the middle of a "face" has its feet on opposite
    cross slopes. Here H * min(tent(x), tent(y)): creases on the diagonals, and on the centre line of a
    face exactly the nominal slope."""
    slope_deg: float
    platform_width: float = 1.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.001

    def function(self, difficulty, spec, rng):
        n = int(self.size[0] / self.horizontal_scale)
        c = int(n / 2)
        h_max = math.tan(math.radians(self.slope_deg)) * self.size[0] / 2 / self.vertical_scale
        tent = (c - np.abs(c - np.arange(n))) / c
        top = h_max * tent[n // 2 - int(self.platform_width / self.horizontal_scale / 2)]
        heights = np.clip(h_max * np.minimum(tent.reshape(-1, 1), tent.reshape(1, -1)), 0, top)
        geom, field, height = add_hfield(spec, heights, self.size[0], self.size[1], self.vertical_scale)
        return TerrainOutput(origin=np.array([self.size[0] / 2, self.size[1] / 2, height]),
                             geometries=[TerrainGeometry(geom=geom, hfield=field)])


@dataclass(kw_only=True)
class ProfileStripTerrainCfg(SubTerrainCfg):
    """A strip whose height changes only along x, made of segments [length_m, slope_deg] (positive =
    uphill towards +x). The size along x is the sum of the lengths."""
    segments: list
    horizontal_scale: float = 0.05
    vertical_scale: float = 0.001

    def function(self, difficulty, spec, rng):
        nx = int(round(self.size[0] / self.horizontal_scale)) + 1
        ny = int(round(self.size[1] / self.horizontal_scale)) + 1
        x = np.linspace(0.0, self.size[0], nx)
        z = np.zeros(nx)
        x0, z0 = 0.0, 0.0
        for length, deg in self.segments:
            inside = (x >= x0) & (x <= x0 + length + 1e-9)
            z[inside] = z0 + (x[inside] - x0) * math.tan(math.radians(deg))
            x0, z0 = x0 + length, z0 + length * math.tan(math.radians(deg))
        z -= z.min()
        geom, field, height = add_hfield(spec, np.tile(z / self.vertical_scale, (ny, 1)),
                                         self.size[0], self.size[1], self.vertical_scale)
        return TerrainOutput(origin=np.array([0.0, self.size[1] / 2, 0.0]),
                             geometries=[TerrainGeometry(geom=geom, hfield=field)])


def strip_segments(s: dict) -> list:
    deg = s["deg"] if s["direction"] == "up" else -s["deg"]
    return [[s["flat_before_m"], 0], [s["slope_m"], deg], [s["flat_after_m"], 0]]


def make_subterrain(s: dict, conf: dict):
    """-> (mjlab sub-terrain, size of the patch). Obstacle sizes are fixed: difficulty plays no role."""
    kind, patch = s["terrain"], (conf["patch_size_m"], conf["patch_size_m"])
    if kind == "flat":
        return BoxFlatTerrainCfg(proportion=1.0), patch
    if kind == "grid":
        h = s["height_cm"] / 100
        return BoxRandomGridTerrainCfg(proportion=1.0, grid_width=0.45, grid_height_range=(h, h),
                                       platform_width=1.0, border_width=0.25), patch
    if kind == "stairs":
        h = s["height_cm"] / 100
        return BoxPyramidStairsTerrainCfg(proportion=1.0, step_height_range=(h, h), step_width=0.12,
                                          platform_width=0.8, border_width=0.2), patch
    if kind == "rubble":
        h = s["height_cm"] / 100
        return hf.HfRandomUniformTerrainCfg(proportion=1.0, noise_range=(-h, h), noise_step=0.002, downsampled_scale=0.08,
                                            horizontal_scale=0.04, vertical_scale=0.001, border_width=0.2), patch
    if kind == "waves":
        a = s["amplitude_cm"] / 100
        return hf.HfWaveTerrainCfg(proportion=1.0, amplitude_range=(a, a), num_waves=s["num_waves"],
                                   horizontal_scale=0.05, vertical_scale=0.001), patch
    if kind == "slope":   # up: the pyramid inverted (a pit), so the robot starts at the bottom and walks uphill
        g = math.tan(math.radians(s["deg"]))
        return hf.HfPyramidSlopedTerrainCfg(proportion=1.0, slope_range=(g, g), platform_width=0.8,
                                            inverted=s["direction"] == "up", horizontal_scale=0.1,
                                            vertical_scale=0.001), patch
    if kind == "face_slope":
        size = conf["face_slope_size_m"]
        return FlatFacePyramidTerrainCfg(proportion=1.0, slope_deg=s["deg"], platform_width=1.0), (size, size)
    if kind == "strip":
        segments = strip_segments(s)
        return ProfileStripTerrainCfg(proportion=1.0, segments=segments), (sum(x[0] for x in segments), conf["strip_width_m"])
    raise ValueError(f"unknown terrain {kind!r}")


def terrain_value(s: dict) -> float:
    """Size of the terrain: obstacle height or wave amplitude [cm], or slope [deg]."""
    return float(next((s[k] for k in ("height_cm", "amplitude_cm", "deg") if k in s), 0.0))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_spec(conf: dict):
    return mujoco.MjSpec.from_file(str(robot_dir() / conf["scene_xml"]))


def robot_com_in_trunk(conf: dict):
    """-> (centre of mass of the robot in the trunk frame, in the default pose; robot mass [kg])."""
    model = load_spec(conf).compile()
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]] = DEFAULT_POSE
    mujoco.mj_forward(model, data)
    trunk = model.body("trunk_base").id
    rot = data.xmat[trunk].reshape(3, 3)
    return rot.T @ (data.subtree_com[trunk] - data.xpos[trunk]), float(model.body_subtreemass[trunk])


def payload_kg(s: dict, robot_mass: float) -> float:
    return s["mass_pct"] / 100 * robot_mass if "mass_pct" in s else 0.0


def build_model(conf: dict, s: dict, terrain_seed: int, com_trunk, robot_mass: float):
    """-> (compiled model, origin of the terrain patch, size of the patch)."""
    spec = load_spec(conf)
    if spec.geom("floor") is not None:
        spec.delete(spec.geom("floor"))   # replaced by the generated terrain (patch + flat border at z = 0)
    sub, size = make_subterrain(s, conf)
    gen = TerrainGenerator(TerrainGeneratorCfg(
        size=size, border_width=conf["border_width_m"], num_rows=1, num_cols=1, curriculum=False,
        difficulty_range=(1.0, 1.0), sub_terrains={s["name"]: sub}, add_lights=False, seed=terrain_seed))
    gen.compile(spec)

    if "mass_pct" in s:
        # Rigid payload welded to the trunk: a non-colliding 2 cm sphere at the robot's centre of mass +
        # offset. If it appears during the episode it starts with a negligible mass (Scenario switches it on).
        body = spec.body("trunk_base").add_body(name="payload", pos=(com_trunk + np.asarray(s["offset_m"])).tolist())
        mass = 1e-6 if s.get("event") == "payload" else payload_kg(s, robot_mass)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], mass=mass, contype=0, conaffinity=0,
                      group=1, rgba=[0.9, 0.15, 0.15, 1.0])   # group 1: visible, ignored by the ground rays

    if conf["actuator"]["model"] == "bam":
        # As microduck_rl's scripts/testbench_sim2real.py: torque motors driven by bam's MujocoController,
        # which writes joint friction and damping at every physics step.
        for a in spec.actuators:
            a.set_to_motor()
            a.forcelimited = True
            a.gear = [1.0, 0, 0, 0, 0, 0]
        for j in spec.joints:
            if j.type == mujoco.mjtJoint.mjJNT_HINGE:
                j.damping = np.zeros_like(j.damping) if np.ndim(j.damping) else 0.0
                j.frictionloss = 0.0
    model = spec.compile()
    model.opt.timestep = conf["physics_dt_s"]
    if conf["actuator"]["model"] == "position" and conf["actuator"].get("max_current_a"):
        from bam.model import load_model
        torque = load_model(motor_name="xl330", model="m6").kt.value * conf["actuator"]["max_current_a"]
        model.actuator_forcerange[:] = [-torque, torque]
        model.actuator_forcelimited[:] = 1
    return model, gen.terrain_origins[0, 0].copy(), size


def set_friction(model, mu: float):
    """Same sliding coefficient on both feet and every terrain geom: in a contact MuJoCo takes the larger
    of the two, so changing only one would have no effect."""
    terrain = model.body("terrain").id
    ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == terrain] + [model.geom(n).id for n in FOOT_GEOMS]
    model.geom_friction[ids, 0] = mu


def ground_height(model, data, x, y, radius) -> float:
    """Highest terrain point within `radius` of (x, y), by vertical rays (robot moved out of the way)."""
    best, geomid = -np.inf, np.zeros(1, dtype=np.int32)
    for dx in np.linspace(-radius, radius, 5):
        for dy in np.linspace(-radius, radius, 5):
            d = mujoco.mj_ray(model, data, np.array([x + dx, y + dy, 50.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
            if d >= 0:
                best = max(best, 50.0 - d)
    return float(best)


def start_pose(s: dict, origin, size, rng, conf: dict):
    """(x, y, yaw) of the start. Directional terrains: small heading and lateral jitter; patches: centre,
    random heading. The heading hold then keeps the start yaw."""
    jitter, lat = math.radians(conf["heading_jitter_deg"]), conf["start_lateral_jitter_m"]
    if s["terrain"] == "face_slope":   # middle of the +x face, facing the edge (downhill) or the top
        yaw = (0.0 if s["direction"] == "downhill" else math.pi) + rng.uniform(-jitter, jitter)
        return origin[0] + size[0] / 4, origin[1] + rng.uniform(-lat, lat), yaw
    if s["terrain"] == "stairs" and s["direction"] == "up":   # near the edge of the patch, facing the top
        return origin[0] + size[0] / 2 - 0.35, origin[1] + rng.uniform(-lat, lat), math.pi + rng.uniform(-jitter, jitter)
    if s["terrain"] == "strip":   # on the first flat, walking towards +x
        return origin[0] + 0.6, origin[1] + rng.uniform(-lat, lat), rng.uniform(-jitter, jitter)
    return origin[0], origin[1], rng.uniform(-math.pi, math.pi)


def spawn_radius(s: dict) -> float:
    """Smaller on the long slopes: the uphill side of a 10 cm circle would lift the start above the surface."""
    return 0.05 if s["terrain"] in ("face_slope", "strip") else 0.1


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------
class Robot:
    def __init__(self, model, conf: dict):
        self.model, self.data = model, mujoco.MjData(model)
        self.trunk = model.body("trunk_base").id
        fj = model.joint("trunk_base_freejoint").id
        self.free_qpos, self.free_dof = model.jnt_qposadr[fj], model.jnt_dofadr[fj]
        self.gyro_adr = model.sensor_adr[model.sensor("imu_ang_vel").id]
        jids = model.actuator_trnid[:, 0]
        self.qpos_idx, self.dof_idx = model.jnt_qposadr[jids], model.jnt_dofadr[jids]
        if model.nu != len(DEFAULT_POSE):
            raise ValueError(f"expected {len(DEFAULT_POSE)} actuators, found {model.nu}")
        self.ctrl = self.bam = None
        act = conf["actuator"]
        if act["model"] == "bam":
            from bam.model import load_model
            from bam.mujoco import MujocoController
            self.bam = load_model(motor_name="xl330", model="m6")
            self.bam.actuator.kp = act["kp"]
            if hasattr(self.bam.actuator, "max_current"):
                self.bam.actuator.max_current = act.get("max_current_a")
            self.kt_over_r = self.bam.kt.value / self.bam.R.value
            self.ctrl = MujocoController(self.bam, [model.actuator(i).name for i in range(model.nu)], model, self.data)
        # terrain geoms alone in one geom group, so that the rays for feet and slope hit only the terrain
        self.is_terrain = model.geom_bodyid == model.body("terrain").id
        groups = set(model.geom_group[self.is_terrain].tolist())
        if len(groups) != 1 or groups & set(model.geom_group[~self.is_terrain].tolist()):
            raise ValueError("terrain geoms must be alone in one geom group")
        self.ray_group = np.zeros(6, dtype=np.uint8)
        self.ray_group[groups.pop()] = 1
        self.feet = [model.geom(n).id for n in FOOT_GEOMS]
        self.foot_verts = [model.mesh_vert[model.mesh_vertadr[i]:model.mesh_vertadr[i] + model.mesh_vertnum[i]].copy()
                           for i in model.geom_dataid[self.feet]]   # sole meshes, in the geom frame
        self.default_friction = float(model.geom_friction[self.feet[0], 0])
        self.saved = {k: getattr(model, k).copy()
                      for k in ("dof_frictionloss", "dof_damping", "geom_friction", "body_mass", "body_inertia")}

    def reset(self, x, y, z, yaw, vin, joint_noise):
        """Restore the model, set the battery voltage, place the robot in the default pose."""
        m, d = self.model, self.data
        for k, v in self.saved.items():
            getattr(m, k)[:] = v
        mujoco.mj_setConst(m, d)
        if self.bam is not None:
            self.bam.actuator.vin = vin
            m.actuator_forcerange[:] = [-vin * self.kt_over_r, vin * self.kt_over_r]
        mujoco.mj_resetData(m, d)
        d.qpos[self.free_qpos:self.free_qpos + 7] = [x, y, z, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        d.qpos[self.qpos_idx] = DEFAULT_POSE + joint_noise
        if self.ctrl is None:
            d.ctrl[:] = DEFAULT_POSE
        mujoco.mj_forward(m, d)
        if self.ctrl is not None:
            self.ctrl.last_ts = d.time
            self.ctrl.reset(d.qpos)
            self.ctrl.q_target[:] = DEFAULT_POSE

    def place(self, s: dict, x, y, yaw, vin, joint_noise, spawn_height_m) -> float:
        """Reset at (x, y, yaw) with the trunk spawn_height_m above the highest ground under it -> z."""
        self.reset(x, y, 100.0, yaw, vin, joint_noise)   # out of the way of the ground rays
        z = ground_height(self.model, self.data, x, y, spawn_radius(s)) + spawn_height_m
        self.reset(x, y, z, yaw, vin, joint_noise)
        return z

    def step(self, target) -> float:
        """One physics step towards the motor targets -> mechanical power sum |torque x joint speed| [W]."""
        if self.ctrl is not None:
            self.ctrl.q_target[:] = target
            self.ctrl.update()
        else:
            self.data.ctrl[:] = target
        mujoco.mj_step(self.model, self.data)
        return float(np.abs(self.data.actuator_force * self.data.qvel[self.dof_idx]).sum())

    def observation(self, last_action, command):
        """The 61 observations of the policy (layout: rollout.OBS_LAYOUT)."""
        d = self.data
        return np.concatenate([d.sensordata[self.gyro_adr:self.gyro_adr + 3], self.gravity_b(),
                               d.qpos[self.qpos_idx] - DEFAULT_POSE, d.qvel[self.dof_idx], last_action, command]
                              ).astype(np.float32)

    # ---- measured state ----
    def to_body(self, vec):
        q = self.data.xquat[self.trunk]
        t = np.cross(q[1:4], vec) * 2
        return vec - q[0] * t + np.cross(q[1:4], t)

    def gravity_b(self):
        return self.to_body(np.array([0.0, 0.0, -1.0]))

    def lin_vel_b(self):
        return self.to_body(self.data.qvel[self.free_dof:self.free_dof + 3].copy())

    def ang_vel_b(self):
        return self.data.qvel[self.free_dof + 3:self.free_dof + 6].copy()   # free joint: already in the body frame

    def position_xy(self):
        return self.data.qpos[self.free_qpos:self.free_qpos + 2].copy()

    def yaw(self):
        w, x, y, z = self.data.xquat[self.trunk]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _ground(self, x, y, z):
        """Terrain height under (x, y), by a ray from height z."""
        dist = mujoco.mj_ray(self.model, self.data, np.array([x, y, z]), np.array([0.0, 0.0, -1.0]),
                             self.ray_group, 1, -1, np.zeros(1, dtype=np.int32))
        return z - dist if dist >= 0 else np.nan

    def feet_state(self):
        """Per foot: contact with the terrain, and height of the lowest point of the sole above the ground [m]."""
        d = self.data
        g1, g2 = d.contact.geom1[:d.ncon], d.contact.geom2[:d.ncon]
        contact = np.array([bool(np.any(((g1 == f) & self.is_terrain[g2]) | ((g2 == f) & self.is_terrain[g1])))
                            for f in self.feet])
        height = np.empty(2)
        for i, (f, verts) in enumerate(zip(self.feet, self.foot_verts)):
            w = d.geom_xpos[f] + verts @ d.geom_xmat[f].reshape(3, 3).T
            p = w[np.argmin(w[:, 2])]
            height[i] = p[2] - self._ground(p[0], p[1], p[2] + 0.03)
        return contact, height

    def ground_slope(self, r=0.1):
        """Slope of the ground under the trunk [deg], along the heading (+ = uphill) and across it
        (+ = rising to the left), from four rays 2r apart: the true situation at every step."""
        x, y, z = self.data.xpos[self.trunk]
        z -= 0.03
        c, s = math.cos(self.yaw()), math.sin(self.yaw())
        front, back = self._ground(x + r * c, y + r * s, z), self._ground(x - r * c, y - r * s, z)
        left, right = self._ground(x - r * s, y + r * c, z), self._ground(x + r * s, y - r * c, z)
        return math.degrees(math.atan2(front - back, 2 * r)), math.degrees(math.atan2(left - right, 2 * r))


# ---------------------------------------------------------------------------
# What changes during an episode
# ---------------------------------------------------------------------------
class Scenario:
    """The situation during one episode: friction at the start, the event at its step, and the scenario
    fields of every recorded step (plan section 4.8). Create it after placing the robot."""

    def __init__(self, s: dict, robot: Robot, robot_mass: float, control_dt: float):
        self.s, self.model = s, robot.model
        self.event_step = round(s["event_t_s"] / control_dt) if "event" in s else None
        if "friction" in s:
            set_friction(self.model, s["friction"])
        self.friction = s.get("friction", robot.default_friction)
        self.payload_kg = payload_kg(s, robot_mass)
        self.com_shift_mm = (np.asarray(s["offset_m"]) * 1000 * self.payload_kg / (robot_mass + self.payload_kg)
                             if "mass_pct" in s else np.zeros(3))
        self.payload_on = "mass_pct" in s and s.get("event") != "payload"
        self.fixed = {"terrain_kind": TERRAINS.index(s["terrain"]), "terrain_value": terrain_value(s),
                      "terrain_direction": DIRECTIONS[s.get("direction")]}

    def update(self, step: int) -> bool:
        """Apply the event if it happens at this control step; True if it did."""
        if step != self.event_step:
            return False
        if self.s["event"] == "friction":
            set_friction(self.model, self.s["friction_after"])
            self.friction = self.s["friction_after"]
        else:   # payload: mass and inertia enter the dynamics at every step (mj_setConst would reset the state)
            pid = self.model.body("payload").id
            self.model.body_mass[pid] = self.payload_kg
            self.model.body_inertia[pid] = 0.4 * self.payload_kg * 0.02 ** 2   # solid sphere, 2 cm radius
            self.payload_on = True
        return True

    def after_event(self, step: int) -> bool:
        return self.event_step is not None and step >= self.event_step

    def fields(self, step: int) -> dict:
        return {**self.fixed, "friction": self.friction, "payload_kg": self.payload_kg if self.payload_on else 0.0,
                "com_shift_mm": self.com_shift_mm if self.payload_on else np.zeros(3),
                "event_active": self.after_event(step)}
