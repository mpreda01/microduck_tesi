"""Single evaluation and data-collection script for every controller (REPO_STRUCTURE.md, section 6).

Runs every (controller, situation) of a suite, `episodes_per_situation` episodes each, and writes a run:
    runs/<phase>/<YYYYMMDD-HHMM>_evaluate_<name>/
        config.yaml        resolved config, with the situations and controllers actually used
        run.json           manifest: code versions, machine, robot mass, runtime
        log.txt            log of the run
        episodes.parquet   one row per episode
        videos/            first episode of the situations listed in the config
    data/steps/<run_id>/   per-step data, if record.steps (fields.json + <controller>/<situation>.npz)

Examples:
    python scripts/evaluate.py configs/eval/fase1.yaml
    python scripts/evaluate.py configs/eval/fase1.yaml --controller alpha_walking,rough_walk_g
    python scripts/evaluate.py configs/eval/fase1.yaml --only flat,friction_0.3 --episodes 2 --no-steps
"""
from __future__ import annotations

import argparse
import logging
import os
import time

import pandas as pd

from steerduck import paths
from steerduck.io import load_config, save_yaml, setup_logging, write_manifest
from steerduck.policies.registry import load_registry, resolve
from steerduck.sim.conditions import load_situations, select
from steerduck.sim.recording import write_fields_json
from steerduck.sim.rollout import control_dt, run_tasks
from steerduck.sim.scene import robot_com_in_trunk

log = logging.getLogger("steerduck")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="evaluation config, e.g. configs/eval/fase1.yaml")
    p.add_argument("--controller", help="comma-separated names from configs/policies.yaml (overrides the config)")
    p.add_argument("--suite", help="suite of configs/situations.yaml (overrides the config)")
    p.add_argument("--only", help="comma-separated situation names: run only these of the suite")
    p.add_argument("--episodes", type=int, help="episodes per situation (overrides the config)")
    p.add_argument("--workers", type=int, help="parallel processes, 0 = one per CPU core (overrides the config)")
    p.add_argument("--no-steps", action="store_true", help="do not record the per-step data")
    p.add_argument("--no-activations", action="store_true", help="record the per-step data without activations")
    p.add_argument("--no-video", action="store_true", help="no videos")
    p.add_argument("--name", help="name of the run folder (default: controllers and suite)")
    return p.parse_args()


def apply_overrides(conf: dict, args) -> dict:
    if args.controller:
        conf["controllers"] = args.controller.split(",")
    if args.suite:
        conf["suite"] = args.suite
    if args.episodes is not None:
        conf["episodes_per_situation"] = args.episodes
    if args.workers is not None:
        conf["workers"] = args.workers
    if args.no_steps:
        conf["record"]["steps"] = False
    if args.no_activations:
        conf["record"]["activations"] = False
    if args.no_video:
        conf["video"]["enabled"] = False
    conf["only"] = args.only.split(",") if args.only else None
    return conf


def main():
    args = parse_args()
    conf = apply_overrides(load_config(args.config), args)
    situations, suites = load_situations()
    selected = select(situations, suites, conf["suite"], conf["only"])
    registry = load_registry()
    specs = [resolve(name, registry) for name in conf["controllers"]]

    name = args.name or "+".join(conf["controllers"]) + f"_{conf['suite']}" + ("_subset" if conf["only"] else "")
    run_dir = paths.new_run_dir(conf["phase"], "evaluate", name)
    log_file = run_dir / "log.txt"
    setup_logging(log_file)
    steps_dir = paths.steps_dir(run_dir.name) if conf["record"]["steps"] else None
    save_yaml({**conf, "controllers_resolved": specs, "situations": [s for _, s in selected],
               "situation_indices": [i for i, _ in selected]}, run_dir / "config.yaml")

    t0 = time.time()
    com_trunk, robot_mass = robot_com_in_trunk(conf)
    info = {"status": "running", "robot_mass_kg": robot_mass, "robot_com_trunk_m": com_trunk.tolist(),
            "steps_dir": str(steps_dir) if steps_dir else None}
    write_manifest(run_dir, **info)
    log.info(f"run {run_dir}")
    log.info(f"robot mass {robot_mass:.3f} kg, centre of mass in the trunk frame {(com_trunk * 1000).round(1)} mm")
    if steps_dir:
        write_fields_json(steps_dir / "fields.json", control_dt(conf), conf["fall_horizon_s"])

    tasks = [{"conf": conf, "spec": spec, "index": index, "situation": sit, "position": k + 1,
              "n_situations": len(selected), "run_dir": str(run_dir), "steps_dir": str(steps_dir) if steps_dir else None}
             for spec in specs for k, (index, sit) in enumerate(selected)]
    workers = conf.get("workers") or os.cpu_count() or 1
    log.info(f"{len(tasks)} tasks (controller x situation), {conf['episodes_per_situation']} episodes each, "
             f"{workers} worker(s)")
    episodes = run_tasks(tasks, workers, log_file)

    df = pd.DataFrame(episodes).sort_values(["controller", "situation_index", "episode"], kind="stable")
    df.insert(0, "run_id", run_dir.name)
    df.to_parquet(run_dir / "episodes.parquet", index=False)
    runtime = time.time() - t0
    write_manifest(run_dir, **{**info, "status": "done", "runtime_s": runtime, "episodes": len(df)})
    log.info(f"done in {runtime / 60:.1f} min -> {run_dir}")
    log.info(f"tables: python scripts/make_report.py {conf['phase']} --run {run_dir.name}")


if __name__ == "__main__":
    main()
