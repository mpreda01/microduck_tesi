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

from steerduck.config import CONFIGS, DATA, find_run, load_situations, load_yaml, matches

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


def recorded_steps(run_dir, source: str, only=None):
    """Yield (situation, obs, action, before_fall) for each situation of an evaluation run with per-step data of
    `source`. only: situation names or patterns (face_*_downhill); before_fall excludes the steps less than
    fall_horizon_s before a fall."""
    conf, steps_dir = load_yaml(run_dir / "config.yaml"), DATA / run_dir.name
    for s in conf["situations"]:
        files = sorted((steps_dir / source).glob(f"{s['name']}__seed*.npz"))
        if files and matches(s["name"], only):
            data = [np.load(f) for f in files]
            obs, action, falls = (np.concatenate([d[k] for d in data]) for k in ("obs", "action", "falls_within_2s"))
            yield s, obs, action, ~falls


def action_distance(run: str, controller: str, sources=None, only=None) -> pd.DataFrame:
    """How far `controller` is from the controllers of an evaluation run, on the states those visited
    (docs/TO_DO_fase2.md, 5.3): mean squared difference between the recorded actions and the actions of
    `controller` on the same observations. One row per (source controller, situation): over all the steps,
    without the steps less than fall_horizon_s before a fall, and per joint. It is the error a steering vector
    starts from in the distillation (plan section 3.2, step 3, with the vector at zero)."""
    from steerduck.policy import relabel, resolve
    run_dir = find_run(run)
    conf, steps_dir = load_yaml(run_dir / "config.yaml"), DATA / run_dir.name
    if conf["action_noise_std"]:
        raise ValueError(f"{run_dir.name}: recorded with action noise {conf['action_noise_std']}: noisy actions")
    if not (steps_dir / "fields.json").is_file():
        raise FileNotFoundError(f"{steps_dir}: no per-step data (was the run evaluated with --no-record?)")
    recorded = [c["name"] for c in conf["controllers"]]
    unknown = set(sources or []) - set(recorded)
    if unknown:
        raise ValueError(f"{sorted(unknown)} not in {run_dir.name}; its controllers: {', '.join(recorded)}")
    joints = json.loads((steps_dir / "fields.json").read_text())["joint_order"]
    spec, rows = resolve(controller), []
    for source in sources or recorded:
        for s, obs, action, before_fall in recorded_steps(run_dir, source, only):
            sq = (action - relabel(spec, obs)) ** 2
            rows.append({"source": source, "controller": controller, "situation": s["name"], "group": s["group"],
                         "steps": len(sq), "mse": sq.mean(),
                         "mse_before_falls": sq[before_fall].mean() if before_fall.any() else np.nan,
                         **{f"mse_{j}": v for j, v in zip(joints, sq.mean(axis=0))}})
    if not rows:
        raise ValueError(f"{run_dir.name}: no per-step data for these controllers and situations")
    return pd.DataFrame(rows)


TRAINING_CURVES = {"reward": "Train/mean_reward", "episode_length_steps": "Train/mean_episode_length",
                   "level_mean": "Curriculum/terrain_levels/mean", "level_max": "Curriculum/terrain_levels/max",
                   "action_std": "Policy/mean_std"}


def training_curves(run_dir, window: int) -> pd.DataFrame:
    """The tensorboard curves of a training run, one row per iteration, averaged over a centred window of
    `window` iterations (one iteration is noisy: the fall share varies by about +-3 points). fall_pct: % of the
    episodes ended in an iteration that ended in a fall."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    files = list(run_dir.glob("events.out.tfevents.*"))
    if len(files) != 1:
        raise FileNotFoundError(f"{run_dir}: expected one tensorboard file, found {len(files)}")
    events = EventAccumulator(str(files[0]), size_guidance={"scalars": 0})
    events.Reload()
    curve = lambda tag: pd.Series({s.step: s.value for s in events.Scalars(tag)})   # noqa: E731
    ended = sum(curve(t) for t in events.Tags()["scalars"] if t.startswith("Episode_Termination/"))
    curves = pd.DataFrame({name: curve(tag) for name, tag in TRAINING_CURVES.items()})
    curves["fall_pct"] = 100 * curve("Episode_Termination/fell_over") / ended
    return curves.rolling(window, center=True, min_periods=1).mean()


def checkpoint_choice(train_run: str, states_run: str, source: str, only=None):
    """Which checkpoints of a training run to evaluate (Notes.txt, scelta del checkpoint), with the rule of
    ppo.checkpoint_choice in configs/training.yaml. Per checkpoint: the training curves around it and its
    distance from the starting policy (mean squared action difference, steps before a fall) on the states that
    `source` visited in an evaluation run. On the plateau: the whole terrain ladder reached and a fall share
    close to the best checkpoint's. -> (table, [closest to the start on the plateau, middle of the plateau,
    last]): the training curves only shortlist, the evaluation decides."""
    import tempfile
    from steerduck.policy import checkpoint_to_onnx, relabel, resolve, training_checkpoints
    rule = load_yaml(CONFIGS / "training.yaml")["ppo"]["checkpoint_choice"]
    train, states = find_run(train_run), find_run(states_run)
    ppo = load_yaml(train / "config.yaml")["ppo"]
    steps = list(recorded_steps(states, source, only))
    if not steps:
        raise ValueError(f"{states.name}: no per-step data of {source} for these situations")
    obs, before_fall = (np.concatenate([step[k] for step in steps])[::rule["state_stride"]] for k in (1, 3))
    start = relabel(resolve(ppo["init_controller"]), obs)
    curves = training_curves(train, rule["window_iterations"])
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for it, path in training_checkpoints(train).items():
            onnx = f"{tmp}/model_{it}.onnx"
            checkpoint_to_onnx(path, train / f"{train.name}.onnx", onnx)
            sq = (relabel({"name": path.stem, "type": "onnx", "path": onnx}, obs) - start) ** 2
            rows.append({"iteration": it, **curves.loc[it], "mse_before_falls": sq[before_fall].mean()})
    table = pd.DataFrame(rows)
    table["ladder_reached"] = table.level_max >= ppo["terrain"]["ladder_rows"] - 1
    best = table.fall_pct[table.ladder_reached].min()
    table["on_plateau"] = table.ladder_reached & (table.fall_pct <= best + rule["max_fall_pct_above_best"])
    plateau = table[table.on_plateau]
    if plateau.empty:
        raise ValueError(f"{train.name}: no checkpoint reached the top of the terrain ladder")
    candidates = [plateau.iteration[plateau.mse_before_falls.idxmin()], plateau.iteration.iloc[len(plateau) // 2],
                  table.iteration.iloc[-1]]
    return table, list(dict.fromkeys(int(c) for c in candidates))


def choose_checkpoint(episodes: pd.DataFrame, distance: pd.DataFrame) -> pd.DataFrame:
    """The choice among evaluated candidates (Notes.txt, scelta del checkpoint), over all the situations of the
    runs: one row per controller with falls and successes and their 95% Wilson intervals, and the mean over the
    situations of mse_before_falls (`distance`, from action_distance on its own states). The reference is the
    candidate with fewest falls (then most successes); `equivalent` when both intervals overlap the reference's;
    `chosen`: the equivalent candidate closest to the starting policy."""
    episodes, rows = add_results(episodes), []
    for controller, eps in episodes.groupby("controller", sort=False):
        n, falls, successes = len(eps), int((eps.outcome == "fell").sum()), int((eps.result == "success").sum())
        rows.append({"controller": controller, "episodes": n, "falls": falls, "successes": successes,
                     **dict(zip(["falls_ci95_lo_pct", "falls_ci95_hi_pct"], wilson_ci(falls, n))),
                     **dict(zip(["success_ci95_lo_pct", "success_ci95_hi_pct"], wilson_ci(successes, n))),
                     "mse_before_falls": distance[distance.source == controller].mse_before_falls.mean()})
    table = pd.DataFrame(rows)
    ref = table.sort_values(["falls", "successes"], ascending=[True, False]).iloc[0]
    table["equivalent"] = ((table.falls_ci95_lo_pct <= ref.falls_ci95_hi_pct)
                           & (table.success_ci95_hi_pct >= ref.success_ci95_lo_pct))
    table["chosen"] = table.index == table[table.equivalent].mse_before_falls.idxmin()
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
