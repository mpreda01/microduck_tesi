"""Train the components of the method (REPO_STRUCTURE.md, section 5). Phase 2: `ppo`.

`ppo --situation <family>` fine-tunes a specialist policy with PPO on one family of configs/training.yaml,
starting from alpha_walking's training checkpoint (actor and critic). Writes runs/<YYYYMMDD-HHMM>_train_ppo_<family>/:
config.yaml, run.json, log.txt, the checkpoints model_<it>.pt, <run_id>.onnx (the latest checkpoint) and the
tensorboard logs.

Examples:
    python scripts/train.py ppo --situation slope_down
    python scripts/train.py ppo --situation slope_down --num-envs 32 --finetune-iterations 2
"""
from __future__ import annotations

import argparse
import logging
import time

from steerduck import config

log = logging.getLogger("steerduck")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(dest="command", required=True)
    ppo = commands.add_parser("ppo", help="fine-tune a specialist policy with PPO")
    ppo.add_argument("--situation", required=True, help="family of configs/training.yaml (ppo.families)")
    ppo.add_argument("--num-envs", type=int, help="parallel environments (default: training.yaml's)")
    ppo.add_argument("--finetune-iterations", type=int, help="fine-tuning iterations (default: training.yaml's)")
    ppo.add_argument("--device", help="cuda:0, cpu... (default: cuda:0 if available)")
    return p.parse_args()


def main():
    args = parse_args()
    import torch

    ppo = config.load_yaml(config.CONFIGS / "training.yaml")["ppo"]
    if args.situation not in ppo["families"]:
        raise SystemExit(f"unknown family {args.situation!r}; known: {', '.join(ppo['families'])}")
    if args.num_envs is not None:
        ppo["num_envs"] = args.num_envs
    if args.finetune_iterations is not None:
        ppo["finetune"]["iterations"] = args.finetune_iterations
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    run_dir = config.new_run("train_ppo", args.situation)
    config.setup_logging(run_dir / "log.txt")
    config.save_yaml({"family": args.situation, "device": device, "ppo": ppo}, run_dir / "config.yaml")
    config.write_manifest(run_dir, status="running", device=device)
    log.info(f"run {run_dir}: {args.situation} on {device}, {ppo['num_envs']} environments")

    from steerduck.mjlab_tasks import train_specialist, upload_to_huggingface
    t0 = time.time()
    diff = train_specialist(args.situation, ppo, run_dir, device)
    hours = (time.time() - t0) / 3600
    results = {"runtime_s": time.time() - t0, "actor_onnx_max_diff": diff,
               "gpu_hours": hours if device.startswith("cuda") else 0.0}
    config.write_manifest(run_dir, status="done", device=device, **results)

    if ppo["huggingface"]["upload"]:
        repo_id = upload_to_huggingface(run_dir, args.situation, ppo)
        config.write_manifest(run_dir, status="done", device=device, huggingface_repo=repo_id, **results)
        log.info(f"uploaded to https://huggingface.co/{repo_id}")

    log.info(f"done in {hours * 60:.1f} min -> {run_dir}\n"
             f"evaluate: add `specialist_{args.situation}: {{type: onnx, run: {run_dir.name}}}` to "
             f"configs/policies.yaml, then python scripts/evaluate.py specialist_{args.situation}")


if __name__ == "__main__":
    main()
