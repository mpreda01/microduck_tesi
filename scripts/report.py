"""Regenerate the table of one or more evaluation runs (REPO_STRUCTURE.md, sections 5 and 7).

The table is named after the runs, results/tables/<run>.csv (several runs joined with `+`), so the
table of a run never overwrites the one of another.

Examples:
    python scripts/report.py --run 20261020-1015_evaluate_alpha_walking_all
    python scripts/report.py --run 20261020-1015_evaluate_alpha_walking_all 20261021-0900_evaluate_rough_walk_g_all
"""
from __future__ import annotations

import argparse
from pathlib import Path

from steerduck import config, report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs="+", required=True, help="run folders (names under runs/ or paths)")
    args = p.parse_args()

    table, worse_groups, conf = report.phase1(args.run)
    out = config.RESULTS / "tables" / ("+".join(Path(run).name for run in args.run) + ".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.4g")
    print(f"{len(table)} rows -> {out}\n")
    print("Episodes by result (all situations):")
    for controller, rows in table.groupby("controller", sort=False):
        n = rows.episodes.sum()
        print(f"  {controller}: success {rows.success.sum()}/{n} ({100 * rows.success.sum() / n:.0f}%), "
              f"fell {rows.falls.sum()}, in place {rows.in_place.sum()}, off course {rows.off_course.sum()}")
    print(f"\nSituations where the controller is measurably worse than {conf['reference']} "
          f"(criterion: at least {conf['min_worse_groups']}):")
    for controller, groups in worse_groups.items():
        print(f"  {controller}: {len(groups)} situation(s): "
              + "; ".join(f"{g} ({', '.join(names)})" for g, names in groups.items()))
        print(f"    criterion {'MET' if len(groups) >= conf['min_worse_groups'] else 'NOT met'}")


if __name__ == "__main__":
    main()
