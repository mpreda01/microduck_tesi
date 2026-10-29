"""Upload a specialist to Hugging Face, as train.py does at the end of a training (REPO_STRUCTURE.md, section 5),
with the huggingface settings saved in the training run (namespace null = the logged-in account).

--run takes:
- a training run: its last checkpoint, to <namespace>/<family>-<YYYYMMDD-HHMM> (as train.py);
- a convert or choose run of a training: the checkpoint it made into ONNX (model_<it>.pt), to
  <namespace>/<family>-<YYYYMMDD-HHMM>-it<it>, with the files of that run in a folder.
Log in once with `huggingface-cli login` (or export HF_TOKEN=...).

Examples:
    python scripts/upload_to_hf.py --run 20261028-2022_convert_20261027-2240_train_ppo_slope_down_it400
    python scripts/upload_to_hf.py --run 20261027-2240_train_ppo_slope_down
"""
from __future__ import annotations

import argparse

from steerduck import config
from steerduck.mjlab_tasks import upload_to_huggingface


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="training, convert or choose run (name under runs/ or path)")
    args = p.parse_args()

    run = config.find_run(args.run)
    conf = config.load_yaml(run / "config.yaml")
    if "ppo" in conf:   # a training run
        train, checkpoint_run, iteration = run, None, None
    elif "chosen_iteration" in conf or "checkpoint" in conf:   # choose / convert run
        train, checkpoint_run = config.find_run(conf["source_run"]), run
        iteration = conf.get("chosen_iteration") or int(conf["checkpoint"].removeprefix("model_").removesuffix(".pt"))
    else:
        raise SystemExit(f"{run.name}: not a training run, nor a convert or choose run with a chosen checkpoint")

    train_conf = config.load_yaml(train / "config.yaml")
    repo_id = upload_to_huggingface(train, train_conf["family"], train_conf["ppo"], checkpoint_run, iteration)
    onnx = f"{(checkpoint_run or train).name}.onnx"
    name = f"{train_conf['family']}-{train.name.split('_')[0]}" + (f"-it{iteration}" if iteration else "")
    print(f"uploaded to https://huggingface.co/{repo_id}\n"
          f"configs/policies.yaml: `{name}: {{type: onnx, hf: [{repo_id}, {onnx}]}}`")


if __name__ == "__main__":
    main()
