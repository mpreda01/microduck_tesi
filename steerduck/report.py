"""From the episodes.parquet of the runs to the tables of the thesis.

Walking metrics are averaged over the COMPLETED episodes only: during a fall they would measure the fall
itself, so a situation where every episode falls has falls but no walking metrics.

Each episode also gets a result (episode_results): success, or the reason it failed (fell, in_place,
off_course), with the thresholds of configs/eval.yaml.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy import stats

from steerduck.config import CONFIGS, find_run, load_situations, load_yaml

WALKING_METRICS = ("progress_m", "vel_error_m_s", "vel_error_1s_m_s", "path_speed_m_s", "yaw_rate_rad_s", "power_w",
                   "max_tilt_deg", "distance_m", "vel_error_after_event_m_s", "power_after_event_w")


def load_runs(runs: list[str]):
    """-> (episodes of all the runs, config of the first run)."""
    episodes, conf = [], None
    for run in runs:
        run_dir = find_run(run)
        if json.loads((run_dir / "run.json").read_text()).get("status") != "done":
            raise ValueError(f"{run_dir.name}: the run did not finish")
        conf = conf or load_yaml(run_dir / "config.yaml")
        episodes.append(pd.read_parquet(run_dir / "episodes.parquet"))
    return pd.concat(episodes, ignore_index=True), conf


def wilson_ci(k: int, n: int, z: float = 1.96):
    """95% Wilson confidence interval of a proportion, in percent."""
    p, den = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def add_progress(episodes: pd.DataFrame) -> pd.DataFrame:
    """progress_m: displacement along the start heading (heading_deg, the direction the heading hold keeps).
    Unlike the forward speed in the body frame, it is about zero when the robot steps in place or walks in
    circles, and negative when it walks backwards."""
    theta = np.radians(episodes.heading_deg)
    return episodes.assign(progress_m=(episodes.end_x - episodes.start_x) * np.cos(theta)
                           + (episodes.end_y - episodes.start_y) * np.sin(theta))


def episode_results(episodes: pd.DataFrame, min_progress_m_s: float) -> pd.Series:
    """success | fell | nan | in_place | off_course for each episode. Success: completed, with a progress of at
    least min_progress_m_s x the episode length. A completed episode without enough progress is in_place when
    its forward speed in the body frame is also below min_progress_m_s (it steps without moving), off_course
    when it walks at least that fast but not in the commanded direction (circles, drift)."""
    enough = episodes.progress_m >= min_progress_m_s * episodes.duration_s
    walking = episodes.path_speed_m_s >= min_progress_m_s
    completed = np.where(enough, "success", np.where(walking, "off_course", "in_place"))
    return pd.Series(np.where(episodes.outcome == "completed", completed, episodes.outcome), index=episodes.index)


def add_results(episodes: pd.DataFrame) -> pd.DataFrame:
    """progress_m, the result with min_progress_m_s of configs/eval.yaml (column `result`) and with each value of
    min_progress_sensitivity_m_s (columns result_<value>m_s). The thresholds are read from the current
    eval.yaml, not from the run: they judge the episodes, they do not change the simulation."""
    criteria = load_yaml(CONFIGS / "eval.yaml")
    episodes = add_progress(episodes)
    episodes["result"] = episode_results(episodes, criteria["min_progress_m_s"])
    for value in criteria["min_progress_sensitivity_m_s"]:
        episodes[f"result_{value:g}m_s"] = episode_results(episodes, value)
    return episodes


def summarize(episodes: pd.DataFrame) -> pd.DataFrame:
    """One row per (controller, situation), situations in the order of situations.yaml: falls and successes
    with their 95% Wilson interval, the count of each reason of failure, the success rate with the other
    thresholds, mean and standard error of the walking metrics over the completed episodes."""
    situations = {s["name"]: s for s in load_situations()[0]}
    rows = []
    for (controller, situation), eps in episodes.groupby(["controller", "situation"], sort=False):
        n, falls = len(eps), int((eps.outcome == "fell").sum())
        lo, hi = wilson_ci(falls, n)
        successes = int((eps.result == "success").sum())
        success_lo, success_hi = wilson_ci(successes, n)
        done = eps[eps.outcome == "completed"]
        row = {"controller": controller, "situation": situation, "group": eps.group.iloc[0], "episodes": n,
               "success": successes, "success_pct": 100 * successes / n,
               "success_ci95_lo_pct": success_lo, "success_ci95_hi_pct": success_hi,
               "in_place": int((eps.result == "in_place").sum()), "off_course": int((eps.result == "off_course").sum()),
               "falls": falls, "falls_pct": 100 * falls / n, "falls_ci95_lo_pct": lo, "falls_ci95_hi_pct": hi,
               "nan": int((eps.outcome == "nan").sum()), "mean_duration_s": eps.duration_s.mean(),
               "completed": len(done)}
        for col in (c for c in eps if c.startswith("result_")):
            row["success_pct_" + col.removeprefix("result_")] = 100 * (eps[col] == "success").mean()
        for key in WALKING_METRICS:
            values = done[key].dropna() if key in done else []
            if len(values):
                row[key] = values.mean()
                row[key + "_sem"] = values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else None
        s = situations.get(situation, {})
        if "event" in s:
            row["falls_after_event"] = int(eps.fell_after_event.fillna(False).astype(bool).sum())
        if "mass_pct" in s:   # shift of the centre of mass: offset x m / (M + m), with m = mass_pct% of M
            frac = s["mass_pct"] / 100
            row["com_shift_mm"] = str([round(1000 * o * frac / (1 + frac), 1) for o in s["offset_m"]])
        rows.append(row)
    table = pd.DataFrame(rows)
    table["order"] = table.situation.map({name: i for i, name in enumerate(situations)})
    return table.sort_values(["controller", "order"], ignore_index=True).drop(columns="order")


def compare_to_reference(table: pd.DataFrame, episodes: pd.DataFrame, conf: dict) -> pd.DataFrame:
    """Flags worse_falls / worse_vel / worse_power / worse. A situation is worse than the reference of the
    same controller when at least one of:
      falls: the 95% intervals do not overlap;
      velocity error (1 s windows): higher, Welch p < worse_p_value and at least worse_min_vel_error_increase_m_s more;
      power: higher, Welch p < worse_p_value and at least worse_min_power_ratio x the reference."""
    ref_name = conf["reference"]
    done = episodes[episodes.outcome == "completed"]

    def higher(controller, situation, col, enough):
        x = done[(done.controller == controller) & (done.situation == situation)][col].dropna()
        y = done[(done.controller == controller) & (done.situation == ref_name)][col].dropna()
        if len(x) < 2 or len(y) < 2:
            return False
        return bool(stats.ttest_ind(x, y, equal_var=False).pvalue < conf["worse_p_value"] and enough(x.mean(), y.mean()))

    def flags(row):
        ref = table[(table.controller == row.controller) & (table.situation == ref_name)]
        if ref.empty or row.situation == ref_name:
            return pd.Series({"worse_falls": False, "worse_vel": False, "worse_power": False})
        return pd.Series({
            "worse_falls": bool(row.falls_ci95_lo_pct > ref.falls_ci95_hi_pct.iloc[0]),
            "worse_vel": higher(row.controller, row.situation, "vel_error_1s_m_s",
                                lambda m, r: m - r >= conf["worse_min_vel_error_increase_m_s"]),
            "worse_power": higher(row.controller, row.situation, "power_w",
                                  lambda m, r: m >= conf["worse_min_power_ratio"] * r),
        })

    table = pd.concat([table, table.apply(flags, axis=1)], axis=1)
    table["worse"] = table[["worse_falls", "worse_vel", "worse_power"]].any(axis=1)
    return table


def phase1(runs: list[str]):
    """Phase 1: the table, and the criterion "at least min_worse_groups situations where the policy gets
    measurably worse than the reference" -> (table, {controller: {group: [situations]}})."""
    episodes, conf = load_runs(runs)
    episodes = add_results(episodes)
    table = compare_to_reference(summarize(episodes), episodes, conf)
    worse = table[table.worse & (table.group != "reference")]
    groups = {c: {g: w.situation.tolist() for g, w in worse[worse.controller == c].groupby("group")}
              for c in table.controller.unique()}
    return table, groups, conf
