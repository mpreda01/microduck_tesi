"""Train the components of the method (REPO_STRUCTURE.md, section 5). Phase 2: `ppo`; phase 3: `sae`.

`ppo --situation <family>` fine-tunes a specialist policy with PPO on one family of configs/training.yaml,
starting from alpha_walking's training checkpoint (actor and critic). Writes runs/<YYYYMMDD-HHMM>_train_ppo_<family>/:
config.yaml, run.json, log.txt, the checkpoints model_<it>.pt, <run_id>.onnx (the latest checkpoint) and the
tensorboard logs. `--situation all` trains every family, one after the other.

`sae --run <evaluation run>` trains the SAE on one layer and the linear probes on all three, from the activations
recorded in the run (section `sae` of configs/training.yaml). Writes runs/<YYYYMMDD-HHMM>_train_sae_<layer>/:
config.yaml, run.json, log.txt, sae.pt, features.csv (a name for each dictionary element) and probes.csv.

Examples:
    python scripts/train.py ppo --situation slope_down
    python scripts/train.py ppo --situation all --num-envs 3072
    python scripts/train.py ppo --situation slope_down --num-envs 32 --finetune-iterations 2
    python scripts/train.py sae --run 20261028-1239_evaluate_alpha_walking_all
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time

from steerduck import config

log = logging.getLogger("steerduck")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(dest="command", required=True)
    ppo = commands.add_parser("ppo", help="fine-tune a specialist policy with PPO")
    ppo.add_argument("--situation", required=True, help="family of configs/training.yaml (ppo.families), or all")
    ppo.add_argument("--num-envs", type=int, help="parallel environments (default: training.yaml's)")
    ppo.add_argument("--finetune-iterations", type=int, help="fine-tuning iterations (default: training.yaml's)")
    ppo.add_argument("--device", help="cuda:0, cpu... (default: cuda:0 if available)")
    sae = commands.add_parser("sae", help="SAE and linear probes on the activations of an evaluation run")
    sae.add_argument("--run", required=True, help="evaluation run recorded with the activations")
    return p.parse_args()


def train_all(args, families: list[str]):
    """One run per family, each in its own process: GPU memory, mjlab and the W&B run start clean, and a
    family that fails does not stop the next ones."""
    options = [f"--{name.replace('_', '-')}={value}" for name, value in vars(args).items()
               if name not in ("command", "situation") and value is not None]
    failed = []
    for family in families:
        print(f"==== {family} ====", flush=True)
        if subprocess.run([sys.executable, __file__, "ppo", "--situation", family, *options]).returncode != 0:
            failed.append(family)
    if failed:
        raise SystemExit(f"training failed for: {', '.join(failed)} (see their log.txt)")
    print(f"trained: {', '.join(families)}")


def train_sae(args):
    """Phase 3, step 1: the SAE on the layer of training.yaml and the linear probes on h1, h2, h3, from the
    activations recorded by sae.controller in an evaluation run."""
    from steerduck import interpret
    conf = config.load_yaml(config.CONFIGS / "training.yaml")["sae"]
    source = config.find_run(args.run)
    run_dir = config.new_run("train_sae", conf["layer"])
    config.setup_logging(run_dir / "log.txt")
    config.save_yaml({"source_run": source.name, "sae": conf}, run_dir / "config.yaml")
    config.write_manifest(run_dir, status="running")

    acts, truth, test, situations = interpret.load_dataset(source, conf["controller"], ["h1", "h2", "h3"],
                                                           conf["quantities"], conf["step_stride"],
                                                           conf["test_every_n_episodes"])
    log.info(f"{len(test)} steps from {source.name}, {test.sum()} for testing")
    interpret.probe_r2(acts, truth, test).to_csv(run_dir / "probes.csv", index=False, float_format="%.4g")

    x = acts[conf["layer"]]
    sae = interpret.train_sae(x, test, conf["layer"], conf)
    interpret.save_sae(sae, run_dir / "sae.pt")
    z, recon = interpret.encode(sae, x[test])
    features = interpret.name_features(z, truth[test])
    features.to_csv(run_dir / "features.csv", index=False, float_format="%.4g")
    interpret.error_by_situation(x[test], recon, situations[test]).to_csv(run_dir / "error_by_situation.csv",
                                                                          index=False, float_format="%.4g")
    results = {"test_fvu": interpret.fvu(x[test], recon),
               "dead_features_pct": float((features.active_pct == 0).mean() * 100)}
    config.write_manifest(run_dir, status="done", **results)
    log.info(f"{results} -> {run_dir}\ncheck in simulation: add to configs/policies.yaml `{conf['controller']}+sae_"
             f"{conf['layer']}: {{type: sae, base: {conf['controller']}, run: {run_dir.name}}}` and evaluate it")


def main():
    args = parse_args()
    if args.command == "sae":
        return train_sae(args)
    import torch

    ppo = config.load_yaml(config.CONFIGS / "training.yaml")["ppo"]
    if args.situation == "all":
        return train_all(args, list(ppo["families"]))
    if args.situation not in ppo["families"]:
        raise SystemExit(f"unknown family {args.situation!r}; known: {', '.join(ppo['families'])}, or all")
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
