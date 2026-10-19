"""Tables of the thesis from the runs (REPO_STRUCTURE.md, sections 6 and 8.4).

The runs are given with --run, or read from results/runs_used.yaml (the runs used in the thesis).
Writes results/tables/<phase>_summary.csv and prints the decision of the phase.

Examples:
    python scripts/make_report.py fase1
    python scripts/make_report.py fase1 --run 20261020-1015_evaluate_alpha_walking_all
"""
from __future__ import annotations

import argparse

import pandas as pd

from steerduck import paths
from steerduck.analysis.aggregate import compare_to_reference, summarize, worse_groups
from steerduck.io import load_config, load_json


def load_runs(phase: str, runs: list[str]):
    """-> (episodes of all the runs, situations in order, config of the first run, robot mass)."""
    episodes, situations, seen, conf, robot_mass = [], [], set(), None, None
    for run in runs:
        run_dir = paths.find_run(phase, run)
        run_conf = load_config(run_dir / "config.yaml")
        manifest = load_json(run_dir / "run.json")
        if manifest.get("status") != "done":
            raise SystemExit(f"{run_dir.name}: the run did not finish (status {manifest.get('status')!r})")
        conf = conf or run_conf
        robot_mass = robot_mass or manifest["robot_mass_kg"]
        episodes.append(pd.read_parquet(run_dir / "episodes.parquet"))
        for sit in run_conf["situations"]:
            if sit["name"] not in seen:
                seen.add(sit["name"])
                situations.append(sit)
    return pd.concat(episodes, ignore_index=True), situations, conf, robot_mass


def report_fase1(phase: str, runs: list[str]):
    """Phase 1: one row per (controller, situation) and the criterion "at least three situations where
    the policy gets measurably worse than flat"."""
    episodes, situations, conf, robot_mass = load_runs(phase, runs)
    rep = conf["report"]
    table = compare_to_reference(summarize(episodes, situations, robot_mass), episodes, rep["reference"], rep["worse"])
    out = paths.RESULTS / "tables" / f"{phase}_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.4g")
    print(f"{len(table)} rows -> {out}")
    print(f"\nSituations where the controller is measurably worse than {rep['reference']} "
          f"(criterion: at least {rep['min_worse_groups']}):")
    for controller, groups in worse_groups(table).items():
        print(f"  {controller}: {len(groups)} situation(s): "
              + "; ".join(f"{g} ({', '.join(names)})" for g, names in groups.items()))
        print(f"    criterion {'MET' if len(groups) >= rep['min_worse_groups'] else 'NOT met'}")


REPORTS = {"fase1": report_fase1}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=sorted(REPORTS))
    p.add_argument("--run", nargs="+", help="run folders (names under runs/<phase>/ or paths)")
    args = p.parse_args()
    runs = args.run
    if not runs:
        used = load_config(paths.RESULTS / "runs_used.yaml")
        runs = used.get(args.phase) or []
        if not runs:
            raise SystemExit(f"no runs for {args.phase} in results/runs_used.yaml: list them there or pass --run")
    REPORTS[args.phase](args.phase, runs)


if __name__ == "__main__":
    main()
