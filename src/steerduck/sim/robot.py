"""The simulated robot: reset, servo control, observations and the measured state (plan section 4.8)."""
from __future__ import annotations

import math

import mujoco
import numpy as np

from steerduck.sim.scene import DEFAULT_POSE, FOOT_GEOMS


class Robot:
    def __init__(self, model, conf: dict):
        self.model, self.conf = model, conf
        self.data = mujoco.MjData(model)
        self.trunk = model.body("trunk_base").id
        fj = model.joint("trunk_base_freejoint").id
        self.free_qpos, self.free_dof = model.jnt_qposadr[fj], model.jnt_dofadr[fj]
        self.gyro_adr = model.sensor_adr[model.sensor("imu_ang_vel").id]
        jids = model.actuator_trnid[:, 0]
        self.qpos_idx = model.jnt_qposadr[jids]
        self.dof_idx = model.jnt_dofadr[jids]
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
            names = [model.actuator(i).name for i in range(model.nu)]
            self.ctrl = MujocoController(self.bam, names, model, self.data)
        # feet and terrain, for contacts, foot heights and the slope under the robot
        terrain = model.body("terrain").id
        self.is_terrain = model.geom_bodyid == terrain
        groups = set(model.geom_group[self.is_terrain].tolist())
        robot_groups = set(model.geom_group[~self.is_terrain].tolist())
        if len(groups) != 1 or groups & robot_groups:
            raise ValueError(f"terrain geoms must be alone in one geom group (terrain {groups}, others {robot_groups})")
        self.ray_group = np.zeros(6, dtype=np.uint8)
        self.ray_group[groups.pop()] = 1          # rays hit the terrain only
        self.feet = [model.geom(n).id for n in FOOT_GEOMS]
        self.foot_verts = [model.mesh_vert[model.mesh_vertadr[i]:model.mesh_vertadr[i] + model.mesh_vertnum[i]].copy()
                           for i in model.geom_dataid[self.feet]]   # sole meshes, in the geom frame
        self._down = np.array([0.0, 0.0, -1.0])
        self._gid = np.zeros(1, dtype=np.int32)
        self.frictionloss0 = model.dof_frictionloss.copy()
        self.damping0 = model.dof_damping.copy()
        self.geom_friction0 = model.geom_friction.copy()
        self.body_mass0 = model.body_mass.copy()
        self.body_inertia0 = model.body_inertia.copy()

    def reset(self, x, y, z, yaw, vin, joint_noise):
        """Restore the model's parameters, set the battery voltage and place the robot in the default pose."""
        m, d = self.model, self.data
        m.dof_frictionloss[:] = self.frictionloss0
        m.dof_damping[:] = self.damping0
        m.geom_friction[:] = self.geom_friction0
        m.body_mass[:] = self.body_mass0
        m.body_inertia[:] = self.body_inertia0
        mujoco.mj_setConst(m, d)
        if self.bam is not None:
            self.bam.actuator.vin = vin
            m.actuator_forcerange[:] = [-vin * self.kt_over_r, vin * self.kt_over_r]
        mujoco.mj_resetData(m, d)
        a = self.free_qpos
        d.qpos[a:a + 3] = [x, y, z]
        d.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        d.qpos[self.qpos_idx] = DEFAULT_POSE + joint_noise
        if self.ctrl is None:
            d.ctrl[:] = DEFAULT_POSE
        mujoco.mj_forward(m, d)
        if self.ctrl is not None:
            self.ctrl.last_ts = d.time
            self.ctrl.reset(d.qpos)
            self.ctrl.q_target[:] = DEFAULT_POSE

    # ---- state of the trunk ----
    def to_body(self, vec):
        q = self.data.xquat[self.trunk]
        w, v = q[0], q[1:4]
        t = np.cross(v, vec) * 2
        return vec - w * t + np.cross(v, t)

    def gravity_b(self):
        return self.to_body(np.array([0.0, 0.0, -1.0]))

    def lin_vel_b(self):
        return self.to_body(self.data.qvel[self.free_dof:self.free_dof + 3].copy())

    def ang_vel_b(self):
        return self.data.qvel[self.free_dof + 3:self.free_dof + 6].copy()   # free joint: already in the body frame

    def yaw(self):
        w, x, y, z = self.data.xquat[self.trunk]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def position_xy(self):
        return self.data.qpos[self.free_qpos:self.free_qpos + 2].copy()

    # ---- feet and ground ----
    def _ground(self, x, y, z):
        """Terrain height under (x, y), by a ray from height z (terrain geoms only)."""
        dist = mujoco.mj_ray(self.model, self.data, np.array([x, y, z]), self._down, self.ray_group, 1, -1, self._gid)
        return z - dist if dist >= 0 else np.nan

    def feet_state(self):
        """Per foot: contact with the terrain, and height of the lowest point of the sole above the ground [m]."""
        d = self.data
        n = d.ncon
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        contact = np.array([bool(np.any(((g1 == f) & self.is_terrain[g2]) | ((g2 == f) & self.is_terrain[g1])))
                            for f in self.feet])
        height = np.empty(2)
        for i, (f, verts) in enumerate(zip(self.feet, self.foot_verts)):
            w = d.geom_xpos[f] + verts @ d.geom_xmat[f].reshape(3, 3).T
            p = w[np.argmin(w[:, 2])]
            height[i] = p[2] - self._ground(p[0], p[1], p[2] + 0.03)
        return contact, height

    def ground_slope(self, r=0.1):
        """Slope of the ground under the trunk [deg]: along the robot's heading (+ = uphill) and across it
        (+ = rising to the robot's left), from four rays 2r apart. The true situation at every step."""
        x, y, z = self.data.xpos[self.trunk]
        z0 = z - 0.03
        c, s = math.cos(self.yaw()), math.sin(self.yaw())
        hf, hb = self._ground(x + r * c, y + r * s, z0), self._ground(x - r * c, y - r * s, z0)
        hl, hr = self._ground(x - r * s, y + r * c, z0), self._ground(x + r * s, y - r * c, z0)
        return math.degrees(math.atan2(hf - hb, 2 * r)), math.degrees(math.atan2(hl - hr, 2 * r))

    # ---- policy interface ----
    def observation(self, last_action, command):
        """The 61 observations of the policy (layout: sim/recording.py, OBS_LAYOUT)."""
        d = self.data
        return np.concatenate([
            d.sensordata[self.gyro_adr:self.gyro_adr + 3],
            self.gravity_b(),
            d.qpos[self.qpos_idx] - DEFAULT_POSE,
            d.qvel[self.dof_idx],
            last_action,
            command,
        ]).astype(np.float32)

    def step(self, target):
        """One physics step towards the motor targets -> mechanical power sum |torque x joint speed| [W]."""
        m, d = self.model, self.data
        if self.ctrl is not None:
            self.ctrl.q_target[:] = target
            self.ctrl.update()
        else:
            d.ctrl[:] = target
        mujoco.mj_step(m, d)
        return float(np.abs(d.actuator_force * d.qvel[self.dof_idx]).sum())
