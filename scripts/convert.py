"""Convert policies between formats (REPO_STRUCTURE.md, section 5). For now `checkpoint`: the ONNX of one
checkpoint of a training run, to evaluate it like any other policy (docs/TO_DO.md, point A).

Writes runs/<YYYYMMDD-HHMM>_convert_<train run>_it<iteration>/ with <run_id>.onnx, config.yaml, run.json and
log.txt. Before converting, the method is checked: the last checkpoint of the run must give back the run's
own ONNX (the policy evaluated so far).

Example:
    python scripts/convert.py checkpoint --run 20261028-0021_train_ppo_slope_up --iteration 400
"""
from __future__ import annotations

import argparse
import logging

from steerduck import config
from steerduck.policy import check_checkpoint_conversion, checkpoint_to_onnx, training_checkpoints

log = logging.getLogger("steerduck")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(dest="command", required=True)
    ckpt = commands.add_parser("checkpoint", help="ONNX of one checkpoint of a training run")
    ckpt.add_argument("--run", required=True, help="training run (name under runs/ or path)")
    ckpt.add_argument("--iteration", type=int, required=True, help="iteration of the checkpoint model_<it>.pt")
    args = p.parse_args()

    train = config.find_run(args.run)
    template = train / f"{train.name}.onnx"
    checkpoints = training_checkpoints(train)
    if args.iteration not in checkpoints:
        raise SystemExit(f"{train.name}: no model_{args.iteration}.pt; checkpoints: {list(checkpoints)}")
    diff = check_checkpoint_conversion(train)

    run_dir = config.new_run("convert", f"{train.name}_it{args.iteration}")
    config.setup_logging(run_dir / "log.txt")
    checkpoint_to_onnx(checkpoints[args.iteration], template, run_dir / f"{run_dir.name}.onnx")
    config.save_yaml({"source_run": train.name, "checkpoint": checkpoints[args.iteration].name,
                      "template": template.name}, run_dir / "config.yaml")
    config.write_manifest(run_dir, status="done", last_checkpoint_max_diff=diff)
    log.info(f"model_{args.iteration}.pt -> {run_dir.name}.onnx (check on model_{max(checkpoints)}.pt: {diff:.1e})\n"
             f"evaluate: add `<name>: {{type: onnx, run: {run_dir.name}}}` to configs/policies.yaml")


if __name__ == "__main__":
    main()
