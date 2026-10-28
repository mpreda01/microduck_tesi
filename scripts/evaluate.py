"""Evaluate controllers on a suite of situations (REPO_STRUCTURE.md, section 5).

Writes runs/<YYYYMMDD-HHMM>_evaluate_<name>/ with config.yaml (the config actually used, with the
situations), run.json, log.txt, episodes.parquet and videos/; the per-step data go to data/<run_id>/.
With --preview it writes only situations_preview.png: one image per situation, robot at the start.

Examples:
    python scripts/evaluate.py --preview
    python scripts/evaluate.py alpha_walking --only flat,friction_0.3 --episodes 2 --no-record
    python scripts/evaluate.py alpha_walking rough_walk_g
"""
from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd

from steerduck import config
from steerduck.policy import resolve
from steerduck.rollout import run_all, start_still, write_fields_json
from steerduck.scene import robot_com_in_trunk

log = logging.getLogger("steerduck")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("controllers", nargs="*", help="names in configs/policies.yaml")
    p.add_argument("--suite", help="suite of configs/situations.yaml (default: eval.yaml's)")
    p.add_argument("--only", help="comma-separated situation names: only these of the suite")
    p.add_argument("--episodes", type=int, help="episodes per situation")
    p.add_argument("--workers", type=int, help="parallel processes (0 = one per CPU core)")
    p.add_argument("--action-noise", type=float, help="std of the Gaussian noise added to every action")
    p.add_argument("--no-record", action="store_true", help="no per-step data")
    p.add_argument("--no-activations", action="store_true", help="per-step data without the activations")
    p.add_argument("--no-video", action="store_true", help="no videos")
    p.add_argument("--preview", action="store_true", help="only an image of each situation at the start")
    p.add_argument("--name", help="name of the run folder (default: controllers and suite)")
    args = p.parse_args()
    if not args.controllers and not args.preview:
        p.error("give at least one controller, or --preview")
    return args


def main():
    args = parse_args()
    conf = config.load_yaml(config.CONFIGS / "eval.yaml")
    overrides = {"suite": args.suite, "episodes_per_situation": args.episodes, "workers": args.workers,
                 "action_noise_std": args.action_noise, "record_steps": False if args.no_record else None,
                 "record_activations": False if args.no_activations else None,
                 "video": False if args.no_video else None}
    conf |= {k: v for k, v in overrides.items() if v is not None}
    situations = config.select_situations(conf["suite"], args.only.split(",") if args.only else None)
    if conf["video_situations"] == "one_per_family":
        # the first situation of each block of situations.yaml and direction: downhill and uphill walk differently
        first_of_family = {}
        for s in situations:
            first_of_family.setdefault((s["family"], s.get("direction")), s["name"])
        conf["video_situations"] = list(first_of_family.values())

    name = args.name or "+".join(args.controllers or ["preview"]) + f"_{conf['suite']}" + ("_only" if args.only else "")
    if conf["action_noise_std"]:
        name += f"_noise{conf['action_noise_std']}"
    run_dir = config.new_run("evaluate", name)
    config.setup_logging(run_dir / "log.txt")
    com_trunk, robot_mass = robot_com_in_trunk(conf)
    log.info(f"run {run_dir}\nrobot mass {robot_mass:.3f} kg, centre of mass in the trunk frame "
             f"{np.round(com_trunk * 1000, 1)} mm")

    if args.preview:
        import imageio
        images = [start_still(conf, s, conf["seeds"][0], com_trunk, robot_mass) for s in situations]
        images += [np.zeros_like(images[0])] * (-len(images) % 6)   # 6 per row
        grid = np.concatenate([np.concatenate(images[i:i + 6], axis=1) for i in range(0, len(images), 6)])
        imageio.imwrite(run_dir / "situations_preview.png", grid)
        log.info(f"{run_dir / 'situations_preview.png'}: {len(situations)} situations in order, 6 per row")
        return

    specs = [resolve(c) for c in args.controllers]
    steps_dir = config.DATA / run_dir.name if conf["record_steps"] else None
    config.save_yaml({**conf, "controllers": specs, "situations": situations}, run_dir / "config.yaml")
    info = {"robot_mass_kg": robot_mass, "steps_dir": str(steps_dir) if steps_dir else None}
    config.write_manifest(run_dir, status="running", **info)
    if steps_dir:
        write_fields_json(steps_dir / "fields.json", conf)

    tasks = [{"conf": conf, "spec": spec, "situation": s, "seed": seed, "run_dir": str(run_dir),
              "steps_dir": str(steps_dir) if steps_dir else None}
             for spec in specs for seed in conf["seeds"] for s in situations]
    workers = conf["workers"] or os.cpu_count()
    log.info(f"{len(tasks)} tasks (controller x seed x situation), {conf['episodes_per_situation']} episodes each, "
             f"{workers} workers")
    t0 = time.time()
    episodes = pd.DataFrame(run_all(tasks, workers, run_dir / "log.txt"))
    episodes["order"] = episodes.situation.map({s["name"]: i for i, s in enumerate(situations)})
    episodes = episodes.sort_values(["controller", "seed", "order", "episode"], ignore_index=True).drop(columns="order")
    episodes.insert(0, "run_id", run_dir.name)
    episodes.to_parquet(run_dir / "episodes.parquet", index=False)
    config.write_manifest(run_dir, status="done", runtime_s=time.time() - t0, episodes=len(episodes), **info)
    log.info(f"done in {(time.time() - t0) / 60:.1f} min -> {run_dir}\n"
             f"table: python scripts/report.py --run {run_dir.name}")


if __name__ == "__main__":
    main()
