"""Evaluate a walking policy under one fixed environment condition (work plan, sections 4.6-4.8).

A condition combines a terrain (optionally a slope walked in a fixed direction),
a floor/foot friction, a payload attached to the trunk and a velocity command.
The script runs N episodes of fixed length and records at every control step:
observations, hidden-layer activations, action, body motion, feet, motor
torques, scenario and outcome (fall within 2 s). It also saves a video of the
first evaluated episode.

Usage:
    uv run --with "imageio[ffmpeg]" python eval_scenario.py eval_config.json
    uv run python eval_scenario.py --list        # available terrain types
"""
import dataclasses
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from importlib.metadata import version

import mujoco
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper

import mjlab.terrains as terrain_gen
import mjlab_microduck.tasks  # noqa: F401  registers the tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
from mjlab_microduck.tasks.slope_terrain import FlatRampTerrainCfg

# ---------------------------------------------------------------------------
# Terrain types (section 4.6). Parameters are scaled for the Microduck, which
# lifts its feet only 1-2 cm; `difficulty` (0-1) picks a point inside each range.
# For types with no parameters below, the parameter names in mjlab 1.3.0 are
# unknown: mjlab's defaults are used, which target larger robots.
# Directional slopes use "slope" (the pyramid heightfield of the rough-terrain training) with
# `slope_deg` and `direction` ("downhill" / "uphill"): the robot walks along the centre line of
# one face. "ramp" (one long tilted box, from the RollerSlope task) is kept only for reference:
# it made both policies fall 75-85% of the time already at 2.5 deg, against 0% on a 5 deg
# pyramid face, so it is not used for the evaluation.
# ---------------------------------------------------------------------------
HF = {"vertical_scale": 0.001}  # 1 mm quantization steps on heightfield surfaces
PYRAMID_PLATFORM_M = 1.0        # side of the flat top platform of the slope pyramid
SLOPE_TILE_SIZE_M = 40.0        # pyramid size for directional slopes: faces about 19 m long, so that from
                                # the middle of a face there are ~9 m both uphill and downhill

CATALOG = {
    "flat": ("BoxFlatTerrainCfg", {}),
    "ramp": (FlatRampTerrainCfg, {}),
    "slope": ("HfPyramidSlopedTerrainCfg", {"slope_range": (0.0, 0.18), "platform_width": 1.0, **HF}),
    "waves": ("HfWaveTerrainCfg", {"amplitude_range": (0.0, 0.02), "num_waves": 4, **HF}),
    "random_uniform": ("HfRandomUniformTerrainCfg", {"noise_range": (0.0, 0.02), "noise_step": 0.005, **HF}),
    "perlin": ("HfPerlinNoiseTerrainCfg", {**HF}),
    "random_grid": ("BoxRandomGridTerrainCfg", {"grid_width": 0.45, "grid_height_range": (0.0, 0.010),
                                                "platform_width": 1.5}),
    "pyramid_stairs": ("BoxPyramidStairsTerrainCfg", {"step_height_range": (0.0, 0.015), "step_width": 0.15,
                                                      "platform_width": 2.0, "border_width": 1.0}),
    "random_stairs": ("BoxRandomStairsTerrainCfg", {"step_height_range": (0.0, 0.015), "step_width": 0.15}),
    "open_stairs": ("BoxOpenStairsTerrainCfg", {"step_height_range": (0.0, 0.015), "step_width": 0.15}),
    "tilted_grid": ("BoxTiltedGridTerrainCfg", {"grid_width": 0.45}),
    "stepping_stones": ("BoxSteppingStonesTerrainCfg", {}),
    "narrow_beams": ("BoxNarrowBeamsTerrainCfg", {}),
    "nested_rings": ("BoxNestedRingsTerrainCfg", {}),
    "random_spread": ("BoxRandomSpreadTerrainCfg", {}),
    "discrete_obstacles": ("HfDiscreteObstaclesTerrainCfg", {}),
}

# Observation layout taken from the mjlab log; used only if the observation
# manager does not expose the term names.
LAYOUT_ACTOR = {"base_ang_vel": [0, 3], "projected_gravity": [3, 6], "joint_pos": [6, 20], "joint_vel": [20, 34],
                "actions": [34, 48], "command": [48, 51], "head_command": [51, 55], "body_command": [55, 61]}
LAYOUT_CRITIC = {"base_lin_vel": [0, 3], "base_ang_vel": [3, 6], "projected_gravity": [6, 9], "joint_pos": [9, 23],
                 "joint_vel": [23, 37], "actions": [37, 51], "command": [51, 54], "foot_height": [54, 56],
                 "foot_air_time": [56, 58], "foot_contact": [58, 60], "foot_contact_forces": [60, 66],
                 "head_command": [66, 70], "body_command": [70, 76]}

FIELD_DESCRIPTIONS = {
    "episode": "episode id (key into 'episodes' in the JSON)",
    "env": "index of the robot copy in the simulator",
    "step": "control step since the start of the episode (50 Hz)",
    "t_s": "time since the start of the episode [s]",
    "obs_actor": "61 observations fed to the policy (layout in meta.obs_layout_actor)",
    "obs_critic": "76 noise-free privileged observations (layout in meta.obs_layout_critic)",
    "action": "14 motor position targets (policy output)",
    "lin_vel_b": "trunk linear velocity in the robot frame [m/s]",
    "ang_vel_b": "trunk angular velocity in the robot frame [rad/s]",
    "gravity_b": "gravity projected in the robot frame (tilt)",
    "foot_contact": "left/right foot contact with the ground",
    "foot_contact_time": "time each foot has been on the ground [s]",
    "foot_air_time": "time each foot has been in the air [s]",
    "foot_height": "height of each foot above the terrain [m] (from the critic foot_height term)",
    "torques": "torques applied by the 14 motors [Nm]",
    "joint_vel": "velocities of the 14 joints [rad/s]",
    "power": "mechanical motor power, sum of |torque x velocity| [W]",
    "vel_error": "|xy velocity - xy command| [m/s]",
    "falls_within_2s": "True if the episode ends with a fall within the next 2 s",
}


# ---------------------------------------------------------------------------
# Policy: ONNX -> PyTorch nn.Sequential (section 4.5)
# ---------------------------------------------------------------------------
class Elementwise(torch.nn.Module):
    """x op c, for the observation normalization baked into the graph."""

    def __init__(self, op, c):
        super().__init__()
        self.op = op
        self.register_buffer("c", torch.as_tensor(np.asarray(c, dtype=np.float32)))

    def forward(self, x):
        if self.op == "Sub":
            return x - self.c
        if self.op == "Add":
            return x + self.c
        if self.op == "Mul":
            return x * self.c
        return x / self.c

    def extra_repr(self):
        return f"{self.op}, shape={tuple(self.c.shape)}"


class Clip(torch.nn.Module):
    def __init__(self, lo, hi):
        super().__init__()
        self.lo, self.hi = lo, hi

    def forward(self, x):
        return torch.clamp(x, self.lo, self.hi)


def load_policy(path):
    """Rebuild the exported MLP as an nn.Sequential (Linear / ELU / normalization)."""
    g = onnx.load(path).graph
    const = {i.name: numpy_helper.to_array(i).copy() for i in g.initializer}
    folding = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide}
    mods = []
    for n in g.node:
        at = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        op, ins = n.op_type, list(n.input)
        if op == "Constant":
            const[n.output[0]] = numpy_helper.to_array(at["value"])
            continue
        if all(i in const for i in ins if i):  # node on constants only: fold it now
            v = [const[i] for i in ins if i]
            if op in folding:
                const[n.output[0]] = folding[op](*v)
            elif op == "Sqrt":
                const[n.output[0]] = np.sqrt(v[0])
            elif op in ("Identity", "Cast"):
                const[n.output[0]] = v[0]
            else:
                raise ValueError(f"unsupported ONNX op on constants: {op}")
            continue
        if op == "Gemm":
            assert at.get("alpha", 1.0) == 1.0 and at.get("beta", 1.0) == 1.0 and not at.get("transA", 0), at
            w = const[ins[1]] if at.get("transB", 0) else const[ins[1]].T
            lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=len(ins) > 2 and ins[2] in const)
            with torch.no_grad():
                lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)))
                if lin.bias is not None:
                    lin.bias.copy_(torch.from_numpy(const[ins[2]].astype(np.float32).reshape(-1)))
            mods.append(lin)
        elif op == "MatMul":
            w = const[ins[1]].T
            lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
            with torch.no_grad():
                lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)))
            mods.append(lin)
        elif op in ("Add", "Sub", "Mul", "Div"):
            if ins[0] in const:
                raise ValueError(f"{op} with the constant as first operand is not supported")
            c = const[ins[1]]
            if op == "Add" and mods and isinstance(mods[-1], torch.nn.Linear) and mods[-1].bias is None:
                mods[-1].bias = torch.nn.Parameter(torch.from_numpy(c.astype(np.float32).reshape(-1)))
            else:
                mods.append(Elementwise(op, c))
        elif op == "Elu":
            mods.append(torch.nn.ELU(alpha=at.get("alpha", 1.0)))
        elif op == "Clip":
            lo = float(const[ins[1]]) if len(ins) > 1 and ins[1] else at.get("min", -np.inf)
            hi = float(const[ins[2]]) if len(ins) > 2 and ins[2] else at.get("max", np.inf)
            mods.append(Clip(lo, hi))
        elif op in ("Identity", "Cast", "Flatten", "Reshape", "Squeeze", "Unsqueeze"):
            continue
        else:
            raise ValueError(f"unsupported ONNX op: {op}")
    return torch.nn.Sequential(*mods).eval()


def check_policy(policy, path, obs):
    """Feed the same inputs to the ONNX model (onnxruntime) and the PyTorch one: outputs must match."""
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    x = obs[:64].detach()
    with torch.no_grad():
        y_torch = policy(x).cpu().numpy()
    x_np = x.cpu().numpy().astype(np.float32)
    y_onnx = np.concatenate([sess.run(None, {input_name: x_np[i:i + 1]})[0] for i in range(len(x_np))])
    diff = float(np.abs(y_torch - y_onnx).max())
    print(f"[check] ONNX vs PyTorch on {len(x_np)} real observations: max difference {diff:.2e}")
    if diff > 1e-4:
        raise RuntimeError(f"The PyTorch policy does not match the ONNX one (diff {diff:.2e}): data would be unreliable")
    return diff


# ---------------------------------------------------------------------------
# Condition -> environment configuration
# ---------------------------------------------------------------------------
def to_tuple(v):
    """JSON turns tuples into lists: convert them back for mjlab parameters."""
    return tuple(v) if isinstance(v, list) else v


def make_terrain(cond, episode_length_s, warnings):
    """Return (TerrainGeneratorCfg, parameters used)."""
    name = cond.get("terrain", "flat")
    if name not in CATALOG:
        raise SystemExit(f"Unknown terrain '{name}'. Available: {', '.join(CATALOG)}")
    difficulty = float(cond.get("difficulty", 1.0))
    overrides = {k: to_tuple(v) for k, v in (cond.get("terrain_params") or {}).items()}

    if name == "ramp":
        # Flat start -> long constant ramp descending towards +x -> flat runout.
        # The robot spawns halfway along the ramp, with room to walk the whole episode
        # in either direction.
        deg = float(cond["slope_deg"])
        speed = abs(float(cond["command"]["vx"])) if isinstance(cond.get("command"), dict) else 0.4
        half = speed * episode_length_s + 1.0
        ramp_len = 2.0 * half
        params = {"flat_length": 1.0, "ramp_length_range": (ramp_len, ramp_len), "runout_length": 3.0,
                  "spawn_on_ramp": half, "deg_min": deg, "deg_max": deg, **overrides}
        sub = FlatRampTerrainCfg(**params)
        size = (params["flat_length"] + ramp_len + params["runout_length"], max(6.0, ramp_len))
        return TerrainGeneratorCfg(size=size, num_rows=1, num_cols=1, curriculum=False,
                                   sub_terrains={"ramp": sub}, add_lights=False), params

    cls, base = CATALOG[name]
    cls = getattr(terrain_gen, cls)
    fields = {f.name for f in dataclasses.fields(cls)}
    params = dict(base)
    if name == "slope" and "slope_deg" in cond:
        s = math.tan(math.radians(float(cond["slope_deg"])))
        params["slope_range"] = (s, s)
        params["platform_width"] = PYRAMID_PLATFORM_M
    params.update(overrides)
    if "proportion" in fields:
        params["proportion"] = 1.0
    try:
        sub = cls(**params)
    except TypeError as e:
        if overrides:
            raise SystemExit(f"{cls.__name__}: invalid parameters ({e}). Available fields: {sorted(fields)}")
        msg = (f"{cls.__name__}: the Microduck-scaled parameters are not valid in this mjlab version ({e}); "
               f"using mjlab defaults. Available fields: {sorted(fields)}")
        print("[WARNING]", msg)
        warnings.append(msg)
        params = {"proportion": 1.0} if "proportion" in fields else {}
        sub = cls(**params)
    if name == "slope" and cond.get("direction"):
        # Directional slope: one large pyramid, so that a face is long enough to walk on for the whole
        # episode in either direction (8 m patches give 3.5 m faces, after which the robot climbs the
        # next pyramid). One patch is enough: the robot copies are independent simulations.
        size = float(cond.get("tile_size_m", SLOPE_TILE_SIZE_M))
        tile, rows_cols = (size, size), 1
    else:
        # A 4 x 4 grid of 8 m patches, all at the same difficulty: random terrains get 16 different samples
        tile, rows_cols = (8.0, 8.0), 4
    gen = TerrainGeneratorCfg(size=tile, border_width=20.0, num_rows=rows_cols, num_cols=rows_cols,
                              curriculum=False, difficulty_range=(difficulty, difficulty),
                              sub_terrains={name: sub}, add_lights=False)
    return gen, params


def pyramid_axis_height(size, slope, platform_width, d, hs=0.1, vs=0.001):
    """Surface height of mjlab's HfPyramidSlopedTerrainCfg on the x axis through the patch centre.

    mjlab builds the height map as H * tent(x) * tent(y), clipped at the level of the top platform:
    only on the axes through the centre does the surface have exactly the nominal slope. This mirrors
    that computation (same pixel rounding). Returns (height at distance d from the centre, height of
    the top platform), in metres; the platform height is the spawn origin of the patch.
    """
    width_px = int(size / hs)
    centre = int(width_px / 2)
    h_units = int(slope * size / 2 / vs)
    half_platform_px = int(platform_width / hs / 2)
    top_units = h_units * ((centre - half_platform_px) / centre) ** 2
    d_px = min(d / hs, centre)
    units = min(h_units * (centre - d_px) / centre, top_units)
    return round(units) * vs, round(top_units) * vs


def spec_geoms(spec):
    if hasattr(spec, "geoms"):
        return list(spec.geoms)
    out, stack = [], [spec.worldbody]
    while stack:
        b = stack.pop()
        out += list(b.geoms)
        stack += list(b.bodies)
    return out


def find_trunk(spec):
    try:
        body = spec.body("robot/trunk_base")
        if body is not None:
            return body
    except Exception:  # noqa: BLE001
        pass
    for b in getattr(spec, "bodies", []):
        if b.name.endswith("trunk_base"):
            return b
    raise RuntimeError("trunk_base body not found in the scene spec")


def set_friction(spec, mu):
    """MuJoCo uses the MAXIMUM of the two friction coefficients in a contact:
    set both the floor and the feet, so the effective friction is exactly mu."""
    n_terrain = 0
    for g in spec.body("terrain").geoms:
        g.friction = [mu, g.friction[1], g.friction[2]]
        n_terrain += 1
    feet = [g for g in spec_geoms(spec) if g.name.endswith(("left_foot_collision", "right_foot_collision"))]
    if len(feet) != 2:
        raise RuntimeError(f"expected 2 foot collision geoms, found {[g.name for g in feet]}")
    for g in feet:
        g.friction = [mu, g.friction[1], g.friction[2]]
    print(f"[friction] mu={mu} on {n_terrain} terrain geoms and feet {[g.name for g in feet]}")


def add_payload(spec, mass, offset):
    """Rigid payload welded to the trunk: a child body with a non-colliding sphere of the given
    mass, placed at `offset` (trunk frame: x forward, y left, z up) from the trunk centre of mass.
    MuJoCo recomputes total mass, centre of mass and inertia."""
    trunk = find_trunk(spec)
    com = np.asarray(trunk.ipos if trunk.ipos is not None else np.zeros(3), dtype=float)
    pos = (com + np.asarray(offset, dtype=float)).tolist()
    body = trunk.add_body(name="payload", pos=pos)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0.0, 0.0], mass=mass,
                  contype=0, conaffinity=0, rgba=[0.85, 0.2, 0.2, 1.0])
    print(f"[payload] {mass} kg at {pos} in the trunk frame (trunk CoM {com.tolist()})")


def add_floor_texture(spec):
    """Checkerboard on every terrain geom (20 cm squares), to see the direction of motion in the
    videos. Textures and materials are only used by the renderer: contacts and dynamics are unchanged."""
    spec.add_texture(name="eval_checker_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=512, height=512,
                     rgb1=[0.82, 0.82, 0.82], rgb2=[0.42, 0.45, 0.50])
    # texuniform: the texture repeats per metre; the builtin checker has 2 x 2 squares,
    # so 2.5 repeats per metre give 20 cm squares
    mat = spec.add_material(name="eval_checker_mat", texrepeat=[2.5, 2.5], texuniform=True, reflectance=0.05)
    try:
        textures = list(mat.textures)
        textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "eval_checker_tex"
        mat.textures = textures
    except AttributeError:  # older MuJoCo spec API
        mat.texture = "eval_checker_tex"
    n = 0
    for g in spec.body("terrain").geoms:
        g.material = "eval_checker_mat"
        g.rgba = [1.0, 1.0, 1.0, 1.0]  # a non-default geom colour would hide the material
        n += 1
    print(f"[texture] checkerboard on {n} terrain geoms (visual only)")


def make_env(conf, warnings, device):
    cond = conf["condition"]
    num_envs = min(conf["num_episodes"], conf["max_envs"])
    terrain = cond.get("terrain", "flat")

    if terrain == "plane":
        # Exactly the flat environment alpha_walking was trained in (Mjlab-Velocity-Flat-MicroDuck):
        # infinite plane, default contacts and solver settings. Reference for the generated terrains,
        # which use softened contacts and more solver iterations.
        cfg = make_microduck_velocity_env_cfg(play=True, rough=False)
        terrain_params = {"terrain_type": "plane"}
    else:
        terrain_cfg, terrain_params = make_terrain(cond, conf["episode_length_s"], warnings)
        # rough=True: softened terrain contacts, more contacts and solver iterations
        cfg = make_microduck_velocity_env_cfg(play=True, rough=True)
        terrain_cfg.seed = conf["seed"]
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = terrain_cfg
        cfg.curriculum.pop("terrain_levels", None)
        if terrain == "flat":
            # Walking off the generated grid only leads onto the flat border: on flat ground it is
            # not a reason to end the episode (it cut the faster constant-speed walks short).
            cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.scene.num_envs = num_envs
    cfg.seed = conf["seed"]
    cfg.episode_length_s = conf["episode_length_s"] + 5.0  # the time-out must not fire before the end

    # Heading: a fixed spawn yaw
    pose_range = cfg.events["reset_base"].params["pose_range"]
    spawn_yaw = None
    if cond.get("heading_deg") is not None:
        spawn_yaw = math.radians(float(cond["heading_deg"]))
        pose_range["yaw"] = (spawn_yaw, spawn_yaw)

    if terrain == "slope" and cond.get("direction"):
        # Directional slope on the +x face of the pyramid, exactly on its centre line (y = 0), where
        # the surface has the nominal slope. Uphill and downhill are the same task in opposite
        # directions: same start in the middle of the face, at the surface height; downhill faces
        # the edge of the pyramid (+x), uphill faces the top (-x).
        size = float(cond.get("tile_size_m", SLOPE_TILE_SIZE_M))
        slope = math.tan(math.radians(float(cond["slope_deg"])))
        d = float(cond.get("slope_start_m", size / 4))  # middle of the face: halfway from centre to edge
        if cond["direction"] == "downhill":
            spawn_yaw = 0.0
        elif cond["direction"] == "uphill":
            spawn_yaw = math.pi
        else:
            raise SystemExit(f"direction must be 'downhill' or 'uphill', not {cond['direction']!r}")
        h, z_top = pyramid_axis_height(size, slope, PYRAMID_PLATFORM_M, d)
        dz = h - z_top  # spawn heights are relative to the patch origin, i.e. the top platform
        z_lo, z_hi = pose_range["z"]
        pose_range["x"] = (d, d)
        pose_range["y"] = (0.0, 0.0)
        pose_range["z"] = (z_lo + dz, z_hi + dz)
        pose_range["yaw"] = (spawn_yaw, spawn_yaw)
        terrain_params["spawn"] = {"x_offset_m": d, "z_offset_m": dz, "yaw_deg": math.degrees(spawn_yaw),
                                   "top_platform_height_m": z_top}
        print(f"[slope] {cond['slope_deg']} deg {cond['direction']}: spawn {d:.2f} m from the pyramid centre, "
              f"{dz:+.3f} m from the top platform height, yaw {math.degrees(spawn_yaw):.0f} deg")

    if cond.get("terrain") == "ramp":
        # no random x/y spawn offset: on a slope it would put the robot inside or above the ramp
        cfg.events["reset_base"].params["pose_range"]["x"] = (0.0, 0.0)
        cfg.events["reset_base"].params["pose_range"]["y"] = (0.0, 0.0)
        # As in the RollerSlope task: out_of_terrain_bounds does not suit the single long ramp tile
        # (it can end episodes right after the spawn), so it is replaced by "fell into the void":
        # the trunk below the lowest flat part of the ramp means the robot left the solid ground.
        cfg.terminations.pop("out_of_terrain_bounds", None)
        drop = terrain_params["ramp_length_range"][1] * math.tan(math.radians(terrain_params["deg_max"]))
        cfg.terminations["fell_into_void"] = TerminationTermCfg(
            func=microduck_mdp.root_height_below,
            params={"min_height": -drop - 0.5, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
        )

    # Velocity command: "random" = training distribution; {"vx": v} = constant forward walk
    command = cond.get("command", "random")
    if isinstance(command, dict):
        c = cfg.commands["twist"]
        c.ranges.lin_vel_x = (float(command["vx"]), float(command["vx"]))
        c.ranges.lin_vel_y = (0.0, 0.0)
        c.ranges.ang_vel_z = (0.0, 0.0)
        c.rel_standing_envs = 0.0
        c.rel_turn_in_place_envs = 0.0
        # mjlab's "forward-only" envs raise vx to at least 0.3 m/s, and "world-frame" envs rotate
        # the command with the robot's yaw: both would change the constant command
        for field in ("rel_forward_envs", "rel_world_envs"):
            if hasattr(c, field):
                setattr(c, field, 0.0)
        if cond.get("hold_heading"):
            # Heading hold, like an operator with a joystick: mjlab's heading controller sets
            # wz = clip(heading_control_stiffness * heading error, ang_vel_z range) every step,
            # with the spawn yaw as target. Without it the policies drift (up to ~0.3 rad/s).
            if spawn_yaw is None:
                raise SystemExit("hold_heading needs a fixed spawn heading (heading_deg or direction)")
            c.heading_command = True
            c.rel_heading_envs = 1.0
            c.ranges.heading = (spawn_yaw, spawn_yaw)
            c.ranges.ang_vel_z = (-1.0, 1.0)  # limits of the corrective yaw-rate command
        else:
            c.rel_heading_envs = 0.0
        cfg.curriculum.pop("standing_envs", None)  # it would reset rel_standing_envs

    # Pushes: in play mode they come every 0.5-1 s; use the training interval or remove them
    if conf["pushes"]:
        cfg.events["push_robot"].interval_range_s = (3.0, 6.0)
    else:
        cfg.events.pop("push_robot", None)

    # Friction and payload are applied to the scene before compilation
    friction = cond.get("friction")
    payload = cond.get("payload")
    if friction is not None:
        cfg.events.pop("foot_friction", None)  # otherwise the training randomization overwrites the feet
    prev_spec_fn = cfg.scene.spec_fn

    floor_texture = conf.get("floor_texture", True)

    def spec_fn(spec):
        if prev_spec_fn is not None:
            prev_spec_fn(spec)
        if friction is not None:
            set_friction(spec, float(friction))
        if payload is not None:
            add_payload(spec, float(payload["mass_kg"]), payload.get("offset_m", [0.0, 0.0, 0.0]))
        if floor_texture:
            try:
                add_floor_texture(spec)
            except Exception as e:  # noqa: BLE001  the texture is cosmetic: never stop the evaluation
                msg = f"floor texture not added: {e}"
                print("[WARNING]", msg)
                warnings.append(msg)

    cfg.scene.spec_fn = spec_fn
    return ManagerBasedRlEnv(cfg=cfg, device=device), num_envs, terrain_params


def obs_layout(env, group, fallback, warnings):
    try:
        om = env.observation_manager
        names, dims = om.active_terms[group], om.group_obs_term_dim[group]
        out, s = {}, 0
        for n, d in zip(names, dims):
            w = int(np.prod(d))
            out[n] = [s, s + w]
            s += w
        return out
    except Exception as e:  # noqa: BLE001
        warnings.append(f"'{group}' layout not read from the observation manager ({e}): using the logged one")
        return fallback


def ids_by_suffix(mj, obj_type, count, suffix):
    return [i for i in range(count) if (mujoco.mj_id2name(mj, obj_type, i) or "").endswith(suffix)]


def scenario_per_env(env, num_envs, warnings):
    """Foot friction, terrain friction, trunk mass, terrain level and column, per copy."""
    out = {}
    mj = env.sim.mj_model
    try:
        feet = (ids_by_suffix(mj, mujoco.mjtObj.mjOBJ_GEOM, mj.ngeom, "left_foot_collision")
                + ids_by_suffix(mj, mujoco.mjtObj.mjOBJ_GEOM, mj.ngeom, "right_foot_collision"))
        gf = torch.as_tensor(env.sim.model.geom_friction)
        f = gf[:, feet, 0] if gf.ndim == 3 else gf[feet, 0].expand(num_envs, -1)
        out["foot_friction"] = f.cpu().numpy()
        terrain_bodies = ids_by_suffix(mj, mujoco.mjtObj.mjOBJ_BODY, mj.nbody, "terrain")
        terrain_geoms = [i for i in range(mj.ngeom) if mj.geom_bodyid[i] in terrain_bodies]
        out["terrain_friction"] = float(mj.geom_friction[terrain_geoms, 0].max()) if terrain_geoms else None
    except Exception as e:  # noqa: BLE001
        warnings.append(f"friction not read: {e}")
    try:
        trunk = ids_by_suffix(mj, mujoco.mjtObj.mjOBJ_BODY, mj.nbody, "trunk_base")[0]
        bm = torch.as_tensor(env.sim.model.body_mass)
        m = bm[:, trunk] if bm.ndim == 2 else bm[trunk].expand(num_envs)
        out["trunk_mass"] = m.cpu().numpy()
        out["robot_mass_nominal"] = float(mj.body_subtreemass[trunk])  # includes the payload
    except Exception as e:  # noqa: BLE001
        warnings.append(f"mass not read: {e}")
    terrain = getattr(env.scene, "terrain", None)
    for key, attr in (("level", "terrain_levels"), ("column", "terrain_types")):
        v = getattr(terrain, attr, None)
        if v is None:
            warnings.append(f"{attr} not available on the scene terrain")
            out[key] = np.full(num_envs, -1)
        else:
            out[key] = torch.as_tensor(v).cpu().numpy().astype(int)
    return out


def software_versions():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        commit = None
    versions = {"microduck_rl_commit": commit, "torch": torch.__version__, "mujoco": mujoco.__version__}
    for pkg in ("mjlab", "mujoco-warp", "warp-lang", "onnxruntime"):
        try:
            versions[pkg] = version(pkg)
        except Exception:  # noqa: BLE001
            versions[pkg] = None
    return versions


def stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    return {"mean": float(x.mean()), "min": float(x.min()), "max": float(x.max()), "std": float(x.std())}


class EpisodeVideo:
    """Video of one evaluated episode (robot copy 0).

    At every control step the state of copy 0 is copied from the GPU simulator into a
    CPU MuJoCo model of the same scene (terrain, ramp and payload included) and rendered
    with a camera that follows the trunk. Rendering needs MUJOCO_GL=egl on Colab.
    """

    def __init__(self, env, width=640, height=480):
        self.model = env.sim.mj_model
        self.data = mujoco.MjData(self.model)
        self.qpos = env.sim.data.qpos
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera.trackbodyid = ids_by_suffix(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.nbody,
                                                "trunk_base")[0]
        self.camera.distance = 0.9
        self.camera.elevation = -15.0
        self.camera.azimuth = 120.0  # three-quarter side view: shows the slope and the gait
        self.frames = []

    def capture(self):
        self.data.qpos[:] = self.qpos[0].detach().cpu().numpy()
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        self.frames.append(self.renderer.render().copy())

    def save(self, path, fps):
        import imageio  # installed with: uv run --with "imageio[ffmpeg]" ...
        imageio.mimsave(path, self.frames, fps=fps)
        self.renderer.close()


def wilson_ci(k, n, z=1.96):
    """95% Wilson confidence interval of a proportion, in percent."""
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [100.0 * max(0.0, centre - half), 100.0 * min(1.0, centre + half)]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def main(conf):
    t_start = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(conf["seed"])
    np.random.seed(conf["seed"])
    warnings = []
    cond = conf["condition"]
    print(f"=== condition '{cond['name']}' | policy {os.path.basename(conf['policy'])} | "
          f"{conf['num_episodes']} episodes ===")

    env, num_envs, terrain_params = make_env(conf, warnings, device)
    dt = env.step_dt
    T = int(round(conf["episode_length_s"] / dt))
    steps_2s = int(round(2.0 / dt))
    robot = env.scene["robot"]
    feet_sensor = env.scene.sensors["feet_ground_contact"]
    layout_a = obs_layout(env, "actor", LAYOUT_ACTOR, warnings)
    layout_c = obs_layout(env, "critic", LAYOUT_CRITIC, warnings)
    cmd_slice = slice(*layout_a.get("command", [48, 51]))

    policy = load_policy(conf["policy"]).to(device)
    print(policy)
    elus = [m for m in policy if isinstance(m, torch.nn.ELU)]
    if not elus:
        raise RuntimeError("no ELU hidden layer found in the policy")
    hidden_names = [f"h{k + 1}" for k in range(len(elus))]
    activations = {}
    for k, m in enumerate(elus):  # forward hook (section 4.10): store each ELU output
        m.register_forward_hook(lambda mod, i, o, k=k: activations.__setitem__(k, o.detach()))

    video = None
    if conf.get("save_video", True):
        try:
            video = EpisodeVideo(env)
        except Exception as e:  # noqa: BLE001
            msg = f"video disabled, the renderer could not start ({e}); is MUJOCO_GL=egl set?"
            print("[WARNING]", msg)
            warnings.append(msg)

    save_steps = conf["save_steps"]
    rows = {}

    def append(name, x):
        rows.setdefault(name, []).append(x.detach().to("cpu"))

    episodes, num_done, round_idx, check_diff, scen = [], 0, 0, None, {}
    while num_done < conf["num_episodes"]:
        num_now = min(num_envs, conf["num_episodes"] - num_done)
        obs, _ = env.reset(seed=conf["seed"] + round_idx)
        if check_diff is None:
            check_diff = check_policy(policy, conf["policy"], obs["actor"])
        scen = scenario_per_env(env, num_envs, warnings)
        alive = torch.zeros(num_envs, dtype=torch.bool, device=device)
        alive[:num_now] = True
        ep_id = torch.arange(num_envs, device=device) + num_done
        end_step = [-1] * num_envs
        outcome = ["completed"] * num_envs
        err_sum = torch.zeros(num_envs, device=device)
        power_sum = torch.zeros(num_envs, device=device)
        step_count = torch.zeros(num_envs, device=device)
        max_tilt = torch.zeros(num_envs, device=device)
        # velocity tracking averaged over ~1 s (an exponential moving average), which removes the
        # oscillation of the trunk within each step; counted after a 1 s warm-up
        err_1s_sum = torch.zeros(num_envs, device=device)
        err_1s_count = torch.zeros(num_envs, device=device)
        fwd_vel_sum = torch.zeros(num_envs, device=device)
        cmd_fwd_sum = torch.zeros(num_envs, device=device)
        yaw_rate_sum = torch.zeros(num_envs, device=device)  # integrated: heading drift of the episode
        ema_alpha, warmup_steps = dt / 1.0, int(round(1.0 / dt))

        for t in range(T):
            if not alive.any():
                break
            idx = alive.nonzero(as_tuple=True)[0]
            if video is not None and round_idx == 0 and bool(alive[0]):
                try:
                    video.capture()  # the first episode of copy 0, frame by frame until it ends
                except Exception as e:  # noqa: BLE001
                    warnings.append(f"video stopped at step {t}: {e}")
                    video = None
            a_obs, c_obs = obs["actor"], obs["critic"]
            with torch.no_grad():
                action = policy(a_obs)  # the hooks fill `activations`

            # ground truth at the same instant as the observations (before the step)
            d = robot.data
            tau, qd = d.actuator_force[:, :14], d.joint_vel[:, :14]
            power = (tau * qd).abs().sum(-1)
            vel_xy = d.root_link_lin_vel_b[:, :2]
            cmd_xy = a_obs[:, cmd_slice][:, :2]
            vel_err = (vel_xy - cmd_xy).norm(dim=-1)  # instantaneous: includes the gait oscillation
            zero = torch.zeros_like(vel_err)
            err_sum += torch.where(alive, vel_err, zero)
            power_sum += torch.where(alive, power, zero)
            step_count += alive.float()
            fwd_vel_sum += torch.where(alive, vel_xy[:, 0], zero)
            cmd_fwd_sum += torch.where(alive, cmd_xy[:, 0], zero)
            yaw_rate_sum += torch.where(alive, d.root_link_ang_vel_b[:, 2], zero)

            pos_xy = d.root_link_pos_w[:, :2]
            if t == 0:
                start_pos, last_pos = pos_xy.clone(), pos_xy.clone()
                vel_ema, cmd_ema = vel_xy.clone(), cmd_xy.clone()
            last_pos = torch.where(alive[:, None], pos_xy, last_pos)
            vel_ema += ema_alpha * (vel_xy - vel_ema)
            cmd_ema += ema_alpha * (cmd_xy - cmd_ema)
            if t >= warmup_steps:
                err_1s = (vel_ema - cmd_ema).norm(dim=-1)
                err_1s_sum += torch.where(alive, err_1s, zero)
                err_1s_count += alive.float()
            # trunk tilt from vertical: catches near-falls that the fall termination does not count
            g = d.projected_gravity_b
            tilt = torch.rad2deg(torch.acos((-g[:, 2] / g.norm(dim=-1).clamp_min(1e-6)).clamp(-1.0, 1.0)))
            max_tilt = torch.where(alive, torch.maximum(max_tilt, tilt), max_tilt)

            if save_steps:
                n = len(idx)
                append("episode", ep_id[idx].int())
                append("env", idx.short())
                append("step", torch.full((n,), t, dtype=torch.int16))
                append("obs_actor", a_obs[idx])
                append("obs_critic", c_obs[idx])
                for k, name in enumerate(hidden_names):
                    append(name, activations[k][idx].half())
                append("action", action[idx])
                append("lin_vel_b", d.root_link_lin_vel_b[idx])
                append("ang_vel_b", d.root_link_ang_vel_b[idx])
                append("gravity_b", d.projected_gravity_b[idx])
                for name, attr in (("foot_contact", "found"), ("foot_contact_time", "current_contact_time"),
                                   ("foot_air_time", "current_air_time")):
                    v = getattr(feet_sensor.data, attr, None)
                    if v is not None:
                        append(name, v[idx].float().reshape(n, -1))
                append("torques", tau[idx])
                append("joint_vel", qd[idx])
                append("power", power[idx])
                append("vel_error", vel_err[idx])

            obs, _, term, trunc, _ = env.step(action)
            finished = alive & (term | trunc)
            for i in finished.nonzero(as_tuple=True)[0].tolist():
                end_step[i] = t
                outcome[i] = "fell" if bool(term[i]) else "truncated"  # truncated = time-out or off the terrain
            alive &= ~finished

        displacement = (last_pos - start_pos).norm(dim=-1)
        for i in range(num_now):
            n_i = max(float(step_count[i]), 1.0)
            duration = (end_step[i] + 1 if end_step[i] >= 0 else T) * dt
            episodes.append({
                "id": int(ep_id[i]),
                "env": i,
                "round": round_idx,
                "terrain_level": int(scen["level"][i]),
                "terrain_column": int(scen["column"][i]),
                "outcome": outcome[i],
                "duration_s": duration,
                "fall_time_s": (end_step[i] + 1) * dt if outcome[i] == "fell" else None,
                "mean_vel_error_m_s": float(err_sum[i]) / n_i,
                "mean_vel_error_1s_m_s": (float(err_1s_sum[i]) / float(err_1s_count[i])
                                          if float(err_1s_count[i]) > 0 else None),
                "mean_forward_vel_m_s": float(fwd_vel_sum[i]) / n_i,
                "mean_cmd_forward_vel_m_s": float(cmd_fwd_sum[i]) / n_i,
                "displacement_m": float(displacement[i]),
                "mean_speed_m_s": float(displacement[i]) / duration,
                "mean_yaw_rate_rad_s": float(yaw_rate_sum[i]) / n_i,          # > 0 = turning left
                "heading_change_deg": math.degrees(float(yaw_rate_sum[i]) * dt),
                "mean_power_W": float(power_sum[i]) / n_i,
                "max_tilt_deg": float(max_tilt[i]),
                "foot_friction": scen["foot_friction"][i].tolist() if "foot_friction" in scen else None,
                "terrain_friction": scen.get("terrain_friction"),
                "trunk_mass_kg": float(scen["trunk_mass"][i]) if "trunk_mass" in scen else None,
            })
        num_done += num_now
        round_idx += 1
        falls = sum(e["outcome"] == "fell" for e in episodes)
        print(f"[round {round_idx}] episodes {num_done}/{conf['num_episodes']} | falls so far {falls} | "
              f"{time.time() - t_start:.0f} s")

    # ---------------- summary ----------------
    n_ep = len(episodes)
    fallen = [e for e in episodes if e["outcome"] == "fell"]
    summary = {
        "condition": cond["name"],
        "episodes": n_ep,
        "falls": len(fallen),
        "falls_per_100": 100.0 * len(fallen) / n_ep,
        "falls_per_100_ci95": wilson_ci(len(fallen), n_ep),
        "truncated": sum(e["outcome"] == "truncated" for e in episodes),
        "completed": sum(e["outcome"] == "completed" for e in episodes),
        "duration_s": stats([e["duration_s"] for e in episodes]),
        "time_to_fall_s": stats([e["fall_time_s"] for e in fallen]),
        "vel_error_m_s": stats([e["mean_vel_error_m_s"] for e in episodes]),
        "vel_error_1s_m_s": stats([e["mean_vel_error_1s_m_s"] for e in episodes
                                   if e["mean_vel_error_1s_m_s"] is not None]),
        "forward_vel_m_s": stats([e["mean_forward_vel_m_s"] for e in episodes]),
        "cmd_forward_vel_m_s": stats([e["mean_cmd_forward_vel_m_s"] for e in episodes]),
        "mean_speed_m_s": stats([e["mean_speed_m_s"] for e in episodes]),
        "abs_yaw_rate_rad_s": stats([abs(e["mean_yaw_rate_rad_s"]) for e in episodes]),
        "abs_heading_change_deg": stats([abs(e["heading_change_deg"]) for e in episodes]),
        "episodes_turning_left": sum(e["mean_yaw_rate_rad_s"] > 0.02 for e in episodes),
        "episodes_turning_right": sum(e["mean_yaw_rate_rad_s"] < -0.02 for e in episodes),
        "mean_power_W": stats([e["mean_power_W"] for e in episodes]),
        "max_tilt_deg": stats([e["max_tilt_deg"] for e in episodes]),
    }
    if summary["truncated"]:
        msg = (f"{summary['truncated']} of {n_ep} episodes ended early WITHOUT a fall (time-out or out of terrain "
               f"bounds), mean duration {summary['duration_s']['mean']:.2f} s: check the video and the episode "
               f"durations before using this condition")
        print("[WARNING]", msg)
        warnings.append(msg)

    # ---------------- saving ----------------
    local_dir = os.path.join("/content/eval_runs", os.path.basename(conf["drive_dir"].rstrip("/")), conf["run_name"])
    os.makedirs(local_dir, exist_ok=True)
    fields = {}
    if save_steps and rows:
        arr = {k: torch.cat(v).numpy() for k, v in rows.items()}
        arr["t_s"] = arr["step"].astype(np.float32) * dt
        if "foot_height" in layout_c:
            arr["foot_height"] = arr["obs_critic"][:, slice(*layout_c["foot_height"])]
        fall_step = np.full(n_ep, -1, dtype=np.int64)
        for e in fallen:
            fall_step[e["id"]] = int(round(e["duration_s"] / dt)) - 1
        fs = fall_step[arr["episode"]]
        arr["falls_within_2s"] = (fs >= 0) & (fs - arr["step"] < steps_2s)
        np.savez_compressed(os.path.join(local_dir, "steps.npz"), **arr)
        descriptions = dict(FIELD_DESCRIPTIONS)
        for k, name in enumerate(hidden_names):
            descriptions[name] = f"hidden layer {k + 1} activations after ELU, float16"
        fields = {k: {"shape": list(v.shape), "dtype": str(v.dtype), "description": descriptions.get(k, "")}
                  for k, v in arr.items()}
        print(f"[data] {len(arr['step'])} steps recorded")

    video_info = None
    if video is not None and video.frames:
        try:
            video.save(os.path.join(local_dir, "episode_video.mp4"), fps=round(1 / dt))
            first = episodes[0]
            video_info = {"file": "episode_video.mp4", "episode_id": first["id"], "outcome": first["outcome"],
                          "duration_s": first["duration_s"], "frames": len(video.frames)}
            print(f"[video] episode {first['id']} ({first['outcome']}, {first['duration_s']:.1f} s)")
        except Exception as e:  # noqa: BLE001
            msg = f"video not saved: {e}"
            print("[WARNING]", msg)
            warnings.append(msg)

    with open(conf["policy"], "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    meta = {
        "run_name": conf["run_name"],
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": conf,
        "terrain_params_used": {k: list(v) if isinstance(v, tuple) else v for k, v in terrain_params.items()},
        "robot_mass_nominal_kg": scen.get("robot_mass_nominal"),
        "policy": {"name": conf.get("policy_name"), "file": os.path.basename(conf["policy"]), "md5": md5},
        "onnx_pytorch_max_diff": check_diff,
        "versions": software_versions(),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "parallel_envs": num_envs,
        "control_dt_s": dt,
        "steps_per_episode": T,
        "obs_layout_actor": layout_a,
        "obs_layout_critic": layout_c,
        "video": video_info,
        "steps_file": "steps.npz" if fields else None,
        "steps_fields": fields,
        "runtime_s": time.time() - t_start,
        "warnings": sorted(set(warnings)),
    }
    with open(os.path.join(local_dir, "summary.json"), "w") as f:
        json.dump({"meta": meta, "summary": summary, "episodes": episodes}, f, indent=1)

    drive_dir = conf["drive_dir"]
    if os.path.isdir("/content/drive/MyDrive") or not drive_dir.startswith("/content/drive"):
        dest = os.path.join(drive_dir, conf["run_name"])
        shutil.copytree(local_dir, dest, dirs_exist_ok=True)
        print(f"[saved] {dest}")
    else:
        print(f"[WARNING] Drive not mounted: files kept in {local_dir}")
    print(json.dumps(summary, indent=1))
    env.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        print(f"{'plane':18s} {'(training flat environment)':32s} infinite plane, default contacts and solver")
        for name, (cls, params) in CATALOG.items():
            cls_name = cls if isinstance(cls, str) else cls.__name__
            print(f"{name:18s} {cls_name:32s} {params}")
        sys.exit(0)
    with open(sys.argv[1]) as f:
        main(json.load(f))
