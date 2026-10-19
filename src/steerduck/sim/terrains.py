"""Terrain of a situation as an mjlab sub-terrain, including two custom heightfields.

The custom terrains are built like mjlab 1.3.0's heightfields, so they have the same contact physics
as the other slopes:
  * FlatFacePyramidTerrainCfg: a pyramid with planar faces, for the directional slopes;
  * ProfileStripTerrainCfg: a strip whose height changes only along x, for the terrain transitions.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains import heightfield_terrains as hf_terrains
from mjlab.terrains.heightfield_terrains import (HfPyramidSlopedTerrainCfg, HfRandomUniformTerrainCfg,
                                                 HfWaveTerrainCfg)
from mjlab.terrains.primitive_terrains import (BoxFlatTerrainCfg, BoxPyramidStairsTerrainCfg,
                                               BoxRandomGridTerrainCfg)
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

# Codes of the per-step fields terrain_kind and terrain_direction
TERRAIN_KINDS = ["flat", "grid", "stairs", "rubble", "waves", "slope", "face_slope", "strip"]
DIRECTIONS = {None: 0, "down": 1, "downhill": 1, "up": 2, "uphill": 2}


def add_hfield(spec, heights_units, size_x, size_y, vertical_scale):
    """Heightfield geom on the terrain body. heights_units has shape (n_y, n_x): MuJoCo rows run
    along y and columns along x, and the n samples of each axis span the whole size."""
    noise = np.rint(heights_units).astype(np.int16)
    lo = int(noise.min())
    span = int(noise.max()) - lo or 1
    height = span * vertical_scale
    norm = (noise - lo) / span
    uid = uuid.uuid4().hex
    field = spec.add_hfield(name=f"hfield_{uid}", size=[size_x / 2, size_y / 2, height, height],
                            nrow=noise.shape[0], ncol=noise.shape[1],
                            userdata=norm.flatten().astype(np.float32).tolist())
    material = hf_terrains.color_by_height(spec, noise, uid, norm)
    geom = spec.body("terrain").add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=field.name,
                                         pos=[size_x / 2, size_y / 2, 0.0], material=material)
    return geom, field, height


@dataclass(kw_only=True)
class FlatFacePyramidTerrainCfg(SubTerrainCfg):
    """Pyramid with four PLANAR faces. mjlab's HfPyramidSlopedTerrainCfg is H * tent(x) * tent(y):
    its creases run along the centre lines, so a robot walking straight up or down the middle of a
    "face" straddles a crease, with its feet on opposite cross slopes (about half the nominal slope).
    Here the height is H * min(tent(x), tent(y)): planar faces, creases on the diagonals, and on the
    centre line of a face exactly the nominal slope with no cross slope."""
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
    """A strip whose height changes only along x, built from segments [length_m, slope_deg]
    (positive = uphill when walking towards +x). E.g. [[1.5, 0], [1.5, -10], [4, 0]] is
    flat -> 10 deg downhill -> flat. The size along x must equal the sum of the lengths."""
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


def make_subterrain(terrain: dict, patch: float):
    """Terrain entry of a situation -> (mjlab sub-terrain, size of the patch). Obstacle sizes are FIXED
    (difficulty plays no role); shapes of the phase 0 battery / hostile-terrain training menu."""
    kind = terrain["kind"]
    if kind == "flat":
        return BoxFlatTerrainCfg(proportion=1.0), (patch, patch)
    if kind == "grid":
        h = terrain["height_cm"] / 100
        return BoxRandomGridTerrainCfg(proportion=1.0, grid_width=0.45, grid_height_range=(h, h),
                                       platform_width=1.0, border_width=0.25), (patch, patch)
    if kind == "stairs":
        h = terrain["height_cm"] / 100
        return BoxPyramidStairsTerrainCfg(proportion=1.0, step_height_range=(h, h), step_width=0.12,
                                          platform_width=0.8, border_width=0.2), (patch, patch)
    if kind == "rubble":
        h = terrain["height_cm"] / 100
        return HfRandomUniformTerrainCfg(proportion=1.0, noise_range=(-h, h), noise_step=0.002,
                                         downsampled_scale=0.08, horizontal_scale=0.04, vertical_scale=0.001,
                                         border_width=0.2), (patch, patch)
    if kind == "waves":  # mjlab waves: peak-to-peak height = 2 x amplitude, num_waves along the patch
        a = terrain["amplitude_cm"] / 100
        return HfWaveTerrainCfg(proportion=1.0, amplitude_range=(a, a), num_waves=terrain.get("num_waves", 4),
                                horizontal_scale=0.05, vertical_scale=0.001), (patch, patch)
    if kind == "slope":  # training pyramid (with its creases), walked down from the top platform
        s = math.tan(math.radians(terrain["deg"]))
        return HfPyramidSlopedTerrainCfg(proportion=1.0, slope_range=(s, s), platform_width=0.8,
                                         horizontal_scale=0.1, vertical_scale=0.001), (patch, patch)
    if kind == "face_slope":
        size = terrain.get("size_m", 40.0)
        return FlatFacePyramidTerrainCfg(proportion=1.0, slope_deg=terrain["deg"], platform_width=1.0), (size, size)
    if kind == "strip":
        length = sum(s[0] for s in terrain["segments"])
        return ProfileStripTerrainCfg(proportion=1.0, segments=terrain["segments"]), (length, terrain.get("width_m", 4.0))
    raise ValueError(f"unknown terrain kind {kind!r}")


def terrain_value(terrain: dict) -> float:
    """The size of the terrain: obstacle height [cm], wave amplitude [cm] or slope [deg]."""
    for key in ("height_cm", "amplitude_cm", "deg"):
        if key in terrain:
            return float(terrain[key])
    if terrain["kind"] == "strip":
        return float(max((abs(s[1]) for s in terrain["segments"]), default=0.0))
    return 0.0
