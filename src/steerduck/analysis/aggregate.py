"""Episodes -> tables: one row per (controller, situation), and the comparison with the reference.

Walking metrics are averaged over the COMPLETED episodes only: during a fall they would measure the
fall itself; a situation where every episode falls has falls but no walking metrics.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

WALKING_METRICS = ("vel_error_m_s", "vel_error_1s_m_s", "path_speed_m_s", "yaw_rate_rad_s", "power_w",
                   "max_tilt_deg", "distance_m", "vel_error_after_event_m_s", "power_after_event_w")


def wilson_ci(k: int, n: int, z: float = 1.96):
    """95% Wilson confidence interval of a proportion, in percent."""
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100.0 * max(0.0, centre - half), 100.0 * min(1.0, centre + half)


def summarize(episodes: pd.DataFrame, situations: list[dict], robot_mass: float) -> pd.DataFrame:
    """One row per (controller, situation), situations in file order: falls with their 95% Wilson
    interval, NaN episodes, mean duration, and mean and standard error of the walking metrics."""
    rows = []
    for controller in episodes.controller.unique():
        for sit in situations:
            eps = episodes[(episodes.controller == controller) & (episodes.situation == sit["name"])]
            if eps.empty:
                continue
            n, falls = len(eps), int((eps.outcome == "fell").sum())
            lo, hi = wilson_ci(falls, n)
            row = {"controller": controller, "situation": sit["name"], "group": sit["group"], "episodes": n,
                   "falls": falls, "falls_pct": 100.0 * falls / n, "falls_ci95_lo_pct": lo, "falls_ci95_hi_pct": hi,
                   "nan": int((eps.outcome == "nan").sum()), "mean_duration_s": float(eps.duration_s.mean())}
            done = eps[eps.outcome == "completed"]
            row["completed"] = len(done)
            for key in WALKING_METRICS:
                vals = done[key].dropna().astype(float) if key in done else pd.Series(dtype=float)
                if len(vals):
                    row[key] = float(vals.mean())
                    row[key + "_sem"] = float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else None
            if sit.get("event"):
                row["falls_after_event"] = int(eps.fell_after_event.fillna(False).astype(bool).sum())
            if sit.get("payload"):
                mass = sit["payload"]["mass_frac"] * robot_mass
                off = np.asarray(sit["payload"].get("offset_m", [0, 0, 0]), dtype=float)
                row["payload_kg"] = mass
                row["com_shift_mm"] = str((1000 * off * mass / (robot_mass + mass)).round(1).tolist())
            rows.append(row)
    return pd.DataFrame(rows)


def compare_to_reference(summary: pd.DataFrame, episodes: pd.DataFrame, reference: str, worse: dict) -> pd.DataFrame:
    """Add worse_falls / worse_vel / worse_power / worse: a situation is worse than the reference of the
    same controller when at least one of
      falls: the 95% intervals do not overlap;
      velocity error (1 s windows): higher, Welch test p < p_value and at least min_vel_error_increase_m_s more;
      power: higher, Welch test p < p_value and at least min_power_ratio x the reference."""
    def flags(row):
        ref = summary[(summary.controller == row.controller) & (summary.situation == reference)]
        if ref.empty or row.situation == reference:
            return pd.Series({"worse_falls": False, "worse_vel": False, "worse_power": False})
        ref = ref.iloc[0]
        mine = episodes[(episodes.controller == row.controller) & (episodes.outcome == "completed")]
        a, b = mine[mine.situation == row.situation], mine[mine.situation == reference]

        def higher(col, enough):
            x, y = a[col].dropna(), b[col].dropna()
            if len(x) < 2 or len(y) < 2:
                return False
            p = stats.ttest_ind(x, y, equal_var=False).pvalue
            return bool(p < worse["p_value"] and enough(x.mean(), y.mean()))

        return pd.Series({
            "worse_falls": bool(row.falls_ci95_lo_pct > ref.falls_ci95_hi_pct),
            "worse_vel": higher("vel_error_1s_m_s", lambda m, r: m - r >= worse["min_vel_error_increase_m_s"]),
            "worse_power": higher("power_w", lambda m, r: m >= worse["min_power_ratio"] * r),
        })

    out = pd.concat([summary, summary.apply(flags, axis=1)], axis=1)
    out["worse"] = out[["worse_falls", "worse_vel", "worse_power"]].any(axis=1)
    return out


def worse_groups(compared: pd.DataFrame, reference_group: str = "reference") -> dict:
    """-> {controller: {group: [situations worse than the reference]}}."""
    out = {}
    for controller in compared.controller.unique():
        w = compared[(compared.controller == controller) & compared.worse & (compared.group != reference_group)]
        out[controller] = {g: w[w.group == g].situation.tolist() for g in sorted(w.group.unique())}
    return out
