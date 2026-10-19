"""Still preview of the situations: one image per situation with the robot at its start (payload in red).

Writes runs/<phase>/<YYYYMMDD-HHMM>_preview_<suite>/situations_preview.png, situations in file order.

Example:
    python scripts/preview_situations.py configs/eval/fase1.yaml
    python scripts/preview_situations.py configs/eval/fase1.yaml --only flat,face_10deg_uphill
"""
from __future__ import annotations

import argparse
import logging
import math

import imageio
import numpy as np

from steerduck import paths
from steerduck.io import load_config, setup_logging
from steerduck.sim.conditions import load_situations, select, start_pose
from steerduck.sim.render import start_still, tile
from steerduck.sim.robot import Robot
from steerduck.sim.rollout import place_robot
from steerduck.sim.scene import build_model, robot_com_in_trunk

log = logging.getLogger("steerduck")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="evaluation config (simulator parameters and seed)")
    p.add_argument("--suite", help="suite of configs/situations.yaml (default: the config's)")
    p.add_argument("--only", help="comma-separated situation names")
    args = p.parse_args()

    conf = load_config(args.config)
    suite = args.suite or conf["suite"]
    situations, suites = load_situations()
    selected = select(situations, suites, suite, args.only.split(",") if args.only else None)
    run_dir = paths.new_run_dir(conf["phase"], "preview", suite)
    setup_logging(run_dir / "log.txt")

    com_trunk, robot_mass = robot_com_in_trunk(conf)
    vin = conf["actuator"]["vin"]
    images = []
    for k, (index, sit) in enumerate(selected):
        model, origin, size = build_model(conf, sit, conf["seed"] + index, com_trunk, robot_mass)
        robot = Robot(model, conf)
        rng = np.random.default_rng([conf["seed"], index, 0])   # the start of episode 0
        x, y, yaw = start_pose(sit, origin, size, rng, conf)
        z = place_robot(robot, sit, x, y, yaw, vin, np.zeros(14), conf)
        images.append(start_still(model, robot.data, x, y, z, yaw))
        log.info(f"[preview] {k + 1:2d} {sit['name']}: start ({x - origin[0]:+.2f}, {y - origin[1]:+.2f}) m, "
                 f"heading {math.degrees(yaw):+.0f} deg, trunk z {z:.3f} m")
    path = run_dir / "situations_preview.png"
    imageio.imwrite(path, tile(images, cols=6))
    log.info(f"saved {path} (situations in order, 6 per row)")


if __name__ == "__main__":
    main()
