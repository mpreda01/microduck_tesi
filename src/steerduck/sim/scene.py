"""MuJoCo model of a situation: robot + generated terrain + payload + servo model."""
from __future__ import annotations

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg

from steerduck import paths
from steerduck.sim.terrains import make_subterrain

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


def load_spec(conf: dict) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(paths.robot_dir() / conf["scene_xml"]))


def robot_com_in_trunk(conf: dict):
    """-> (centre of mass of the whole robot in the trunk frame, in the default pose; robot mass)."""
    model = load_spec(conf).compile()
    data = mujoco.MjData(model)
    jids = model.actuator_trnid[:, 0]
    data.qpos[model.jnt_qposadr[jids]] = DEFAULT_POSE
    mujoco.mj_forward(model, data)
    trunk = model.body("trunk_base").id
    rot = data.xmat[trunk].reshape(3, 3)
    com_world = data.subtree_com[trunk]
    return rot.T @ (com_world - data.xpos[trunk]), float(model.body_subtreemass[trunk])


def build_model(conf: dict, situation: dict, terrain_seed: int, com_trunk, robot_mass: float):
    """-> (compiled model, origin of the terrain patch, size of the patch)."""
    spec = load_spec(conf)
    floor = spec.geom("floor")
    if floor is not None:
        spec.delete(floor)  # replaced by the generated terrain (patch + flat border at z = 0)
    sub, size = make_subterrain(situation["terrain"], conf["patch_size_m"])
    gen = TerrainGenerator(TerrainGeneratorCfg(
        size=size, border_width=conf["border_width_m"], num_rows=1, num_cols=1, curriculum=False,
        difficulty_range=(1.0, 1.0), sub_terrains={situation["name"]: sub}, add_lights=False, seed=terrain_seed))
    gen.compile(spec)

    payload = situation.get("payload")
    if payload:
        # Rigid payload welded to the trunk: a non-colliding sphere at the robot's centre of mass + offset
        # (offset 0: mass added, centre of mass unchanged). If it appears during the episode, it starts
        # with a negligible mass and the event switches its mass on (sim/conditions.py).
        mass = payload["mass_frac"] * robot_mass
        pos = com_trunk + np.asarray(payload.get("offset_m", [0, 0, 0]), dtype=float)
        body = spec.body("trunk_base").add_body(name="payload", pos=pos.tolist())
        start_mass = 1e-6 if (situation.get("event") or {}).get("type") == "payload" else mass
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], mass=start_mass,
                      contype=0, conaffinity=0, group=1,   # group 1: visible, ignored by the ground rays
                      rgba=[0.9, 0.15, 0.15, 1.0])

    act = conf["actuator"]
    if act["model"] == "bam":
        # As scripts/testbench_sim2real.py: torque motors driven by bam's MujocoController; the force
        # range (voltage bound) is set per episode from the battery voltage; joint friction and damping
        # are written by bam at every physics step.
        for a in spec.actuators:
            a.set_to_motor()
            a.forcelimited = True
            a.gear = [1.0, 0, 0, 0, 0, 0]
        for j in spec.joints:
            if j.type == mujoco.mjtJoint.mjJNT_HINGE:
                j.damping = np.zeros_like(j.damping) if np.ndim(j.damping) else 0.0  # array in newer MuJoCo
                j.frictionloss = 0.0
    model = spec.compile()
    model.opt.timestep = conf["physics_dt_s"]
    if act["model"] == "position" and act.get("max_current_a"):
        from bam.model import load_model
        kt = load_model(motor_name="xl330", model="m6").kt.value
        model.actuator_forcerange[:] = [-kt * act["max_current_a"], kt * act["max_current_a"]]
        model.actuator_forcelimited[:] = 1
    return model, gen.terrain_origins[0, 0].copy(), size


def friction_geoms(model):
    """-> (ids of the geoms whose friction a situation sets: both feet and every terrain geom, foot ids)."""
    terrain = model.body("terrain").id
    ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == terrain]
    feet = [model.geom(n).id for n in FOOT_GEOMS]
    return ids + feet, feet


def set_friction(model, mu: float):
    """Same sliding coefficient on the feet and the floor: in a contact MuJoCo takes the larger of the
    two, so changing only one would have no effect. Torsional and rolling friction unchanged."""
    ids, _ = friction_geoms(model)
    model.geom_friction[ids, 0] = mu


def ground_height(model, data, x: float, y: float, radius: float = 0.1) -> float:
    """Highest terrain point within `radius` of (x, y), by vertical rays (robot moved out of the way)."""
    best = -np.inf
    geomid = np.zeros(1, dtype=np.int32)
    for dx in np.linspace(-radius, radius, 5):
        for dy in np.linspace(-radius, radius, 5):
            d = mujoco.mj_ray(model, data, np.array([x + dx, y + dy, 50.0]), np.array([0.0, 0.0, -1.0]),
                              None, 1, -1, geomid)
            if d >= 0:
                best = max(best, 50.0 - d)
    return float(best)
