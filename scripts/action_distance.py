"""Distance of the controllers of an evaluation run from another controller, on the states they visited
(docs/TO_DO_fase2.md, 5.3; REPO_STRUCTURE.md, section 5). Nothing is simulated: the controller is run on the
observations recorded in data/<run>/ and its actions are compared with the recorded ones.
Writes results/tables/action_distance__<run>__<controller>.csv.

Examples:
    python scripts/action_distance.py --run <run> --controller alpha_walking
    python scripts/action_distance.py --run <run> --controller alpha_walking --source slope_down-20261027-2240 --only flat
    python scripts/action_distance.py --run <run> --controller slope_down-20261027-2240 --source slope_down-20261027-2240
        (check: a controller against its own data gives about 0)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from steerduck import config, report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="evaluation run with per-step data (name under runs/ or path)")
    p.add_argument("--controller", required=True, help="name in configs/policies.yaml: re-evaluates the observations")
    p.add_argument("--source", help="comma-separated controllers of the run whose observations are used (default: all)")
    p.add_argument("--only", help="comma-separated situation names (default: all those of the run)")
    args = p.parse_args()

    split = lambda names: names.split(",") if names else None   # noqa: E731
    table = report.action_distance(args.run, args.controller, split(args.source), split(args.only))
    out = config.RESULTS / "tables" / f"action_distance__{Path(args.run).name}__{args.controller}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.4g")
    print(f"{len(table)} rows -> {out}")
    for source, rows in table.groupby("source", sort=False):
        print(f"\n{source} vs {args.controller}: mean squared action difference, mean over the situations of a group")
        print(rows.groupby("group", sort=False).agg(situations=("situation", "size"), steps=("steps", "sum"),
                                                     mse=("mse", "mean"), mse_before_falls=("mse_before_falls", "mean"))
              .to_string(float_format="%.4g"))


if __name__ == "__main__":
    main()
