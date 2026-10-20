"""Regenerate the tables of the thesis from the runs (REPO_STRUCTURE.md, sections 5 and 7).

The runs are given with --run, or read from results/runs_used.yaml (the runs used in the thesis).

Examples:
    python scripts/report.py --phase 1 --run 20261020-1015_evaluate_alpha_walking_all
    python scripts/report.py --phase 1
"""
from __future__ import annotations

import argparse

from steerduck import config, report


def phase1(runs):
    table, worse_groups, conf = report.phase1(runs)
    out = config.RESULTS / "tables" / "fase1_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.4g")
    print(f"{len(table)} rows -> {out}\n")
    print(f"Situations where the controller is measurably worse than {conf['reference']} "
          f"(criterion: at least {conf['min_worse_groups']}):")
    for controller, groups in worse_groups.items():
        print(f"  {controller}: {len(groups)} situation(s): "
              + "; ".join(f"{g} ({', '.join(names)})" for g, names in groups.items()))
        print(f"    criterion {'MET' if len(groups) >= conf['min_worse_groups'] else 'NOT met'}")


PHASES = {1: phase1}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", type=int, choices=sorted(PHASES), required=True)
    p.add_argument("--run", nargs="+", help="run folders (names under runs/ or paths)")
    args = p.parse_args()
    runs = args.run or config.load_yaml(config.RESULTS / "runs_used.yaml").get(f"fase{args.phase}")
    if not runs:
        p.error(f"no runs: pass --run, or list them under fase{args.phase} in results/runs_used.yaml")
    PHASES[args.phase](runs)


if __name__ == "__main__":
    main()
