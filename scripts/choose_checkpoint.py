"""From a training run to its specialist, in one command (Notes.txt, scelta del checkpoint; REPO_STRUCTURE.md,
section 5):
1. the last checkpoint is evaluated on the situations of the family, with the per-step data;
2. report.checkpoint_choice shortlists checkpoints from the training curves and their distance from the starting
   policy on those states;
3. the other candidates are evaluated on the same episodes;
4. each candidate's distance from the starting policy on its own states (report.action_distance);
5. report.choose_checkpoint applies the rule: fewest falls, then the closest to the start among the equivalent ones.
With --shortlist it stops after step 2; with --states and --source too, step 1 uses the states of an existing
evaluation run and nothing is simulated (about 30 s).
Writes runs/<YYYYMMDD-HHMM>_choose_<training run>/ with checkpoints.csv, config.yaml, run.json, log.txt and, after
step 5, the ONNX of each candidate (it<it>.onnx), the chosen one as <run_id>.onnx and choice.csv. The evaluations
are runs of their own, as with evaluate.py.

Examples:
    python scripts/choose_checkpoint.py --run 20261027-2240_train_ppo_slope_down --only "face_*_downhill,pyramid_*"
    python scripts/choose_checkpoint.py --run 20261027-2240_train_ppo_slope_down --shortlist \
        --states 20261028-1648_evaluate_slope_up-20261028-0021+slope_down-20261027-2240_all_only \
        --source slope_down-20261027-2240 --only "face_*_downhill,pyramid_*"
"""
from __future__ import annotations

import argparse
import logging
import shutil

import pandas as pd

from steerduck import config, report
from steerduck.policy import check_checkpoint_conversion, checkpoint_to_onnx, training_checkpoints
from steerduck.rollout import evaluate_run

log = logging.getLogger("steerduck")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="training run (name under runs/ or path)")
    p.add_argument("--only", help="comma-separated situations of the family: names or patterns")
    p.add_argument("--shortlist", action="store_true", help="stop after the candidates, without evaluating them")
    p.add_argument("--states", help="with --shortlist: evaluation run with per-step data, instead of step 1")
    p.add_argument("--source", help="with --states: the controller of that run whose states are used")
    p.add_argument("--episodes", type=int, help="episodes per situation (default: eval.yaml's)")
    p.add_argument("--workers", type=int, help="parallel processes (default: eval.yaml's)")
    args = p.parse_args()
    if args.states and not (args.shortlist and args.source):
        p.error("--states needs --shortlist and --source")
    if not args.states and not args.only:
        p.error("give --only: the situations to evaluate")
    return args


def main():
    args = parse_args()
    train = config.find_run(args.run)
    train_conf = config.load_yaml(train / "config.yaml")
    checkpoints, diff = training_checkpoints(train), check_checkpoint_conversion(train)
    conf = config.load_yaml(config.CONFIGS / "eval.yaml") | {"record_steps": True, "record_activations": False,
                                                             "video": False}
    conf |= {k: v for k, v in {"episodes_per_situation": args.episodes, "workers": args.workers}.items() if v}
    only = args.only.split(",") if args.only else None
    rule = config.load_yaml(config.CONFIGS / "training.yaml")["ppo"]["checkpoint_choice"]

    run_dir = config.new_run("choose", train.name)
    config.setup_logging(run_dir / "log.txt")
    config.write_manifest(run_dir, status="running", last_checkpoint_max_diff=diff)
    prefix, last = f"{train_conf['family']}-{train.name[:13]}", list(checkpoints)[-1]
    name = lambda it: f"{prefix}-it{it}"   # noqa: E731  e.g. slope_down-20261027-2240-it400

    def evaluate(iterations, run_name):
        """Evaluate checkpoints, converted into this run; the evaluation is a run of its own."""
        specs = []
        for it in iterations:
            checkpoint_to_onnx(checkpoints[it], train / f"{train.name}.onnx", run_dir / f"it{it}.onnx")
            specs.append({"name": name(it), "type": "onnx", "path": str(run_dir / f"it{it}.onnx")})
        eval_dir = evaluate_run(conf, specs, situations, run_name)
        config.setup_logging(run_dir / "log.txt")   # back to this run's log
        return eval_dir

    if args.states:
        runs, states, source = [], config.find_run(args.states), args.source
    else:
        situations = config.select_situations(conf["suite"], only)
        runs = [evaluate([last], f"{name(last)}_choose")]
        states, source = runs[0], name(last)
    table, candidates = report.checkpoint_choice(str(train), str(states), source, only)
    table.to_csv(run_dir / "checkpoints.csv", index=False, float_format="%.4g")
    log.info(f"{table.to_string(index=False, float_format='%.4g')}\n\ncandidates from the training curves "
             f"(closest to the start on the plateau, middle of the plateau, last): {candidates}")
    summary = {"source_run": train.name, "only": only, "states": states.name, "source": source,
               "candidates": candidates, "checkpoint_choice": rule}

    if not args.shortlist:
        others = [it for it in candidates if it != last]
        if others:
            runs.append(evaluate(others, f"{prefix}-candidates_choose"))
        start = train_conf["ppo"]["init_controller"]
        distance = pd.concat([report.action_distance(str(r), start) for r in runs])
        choice = report.choose_checkpoint(report.load_runs([str(r) for r in runs])[0], distance)
        choice.insert(1, "iteration", choice.controller.map({name(it): it for it in candidates}))
        choice.to_csv(run_dir / "choice.csv", index=False, float_format="%.4g")
        chosen = int(choice.iteration[choice.chosen].iloc[0])
        shutil.copy(run_dir / f"it{chosen}.onnx", run_dir / f"{run_dir.name}.onnx")
        summary |= {"start_controller": start, "evaluations": [r.name for r in runs], "chosen_iteration": chosen}
        log.info(f"{choice.to_string(index=False, float_format='%.4g')}\n\nchosen: iteration {chosen}\n"
                 f"add to configs/policies.yaml: `{name(chosen)}: {{type: onnx, run: {run_dir.name}}}`")

    config.save_yaml(summary, run_dir / "config.yaml")
    config.write_manifest(run_dir, status="done", last_checkpoint_max_diff=diff,
                          chosen_iteration=summary.get("chosen_iteration"))
    log.info(f"-> {run_dir}")


if __name__ == "__main__":
    main()
