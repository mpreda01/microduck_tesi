"""Canonical locations of the repository's folders and names of the runs (REPO_STRUCTURE.md, 3 and 8.1).

The heavy outputs (models/, data/, runs/) go under $STEERDUCK_STORAGE when it is set, for example a
Google Drive folder on Colab; otherwise inside the repository. No other module builds these paths.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
STORAGE = Path(os.environ.get("STEERDUCK_STORAGE", ROOT))
MODELS = STORAGE / "models"
DATA = STORAGE / "data"
RUNS = STORAGE / "runs"


def microduck_rl_dir() -> Path:
    """The microduck_rl checkout: $MICRODUCK_RL_DIR, else external/microduck_rl, else ./microduck_rl."""
    env = os.environ.get("MICRODUCK_RL_DIR")
    candidates = [Path(env)] if env else [ROOT / "external" / "microduck_rl", ROOT / "microduck_rl"]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(f"microduck_rl not found in {[str(c) for c in candidates]}; "
                            "clone it there or set MICRODUCK_RL_DIR")


def robot_dir() -> Path:
    """Folder with the robot's MJCF files (scene.xml)."""
    return microduck_rl_dir() / "src" / "mjlab_microduck" / "robot" / "microduck"


def steps_dir(run_id: str) -> Path:
    """Per-step data of a run: data/steps/<run_id>/<controller>/<situation>.npz."""
    return DATA / "steps" / run_id


def new_run_dir(phase: str, script: str, name: str) -> Path:
    """Create runs/<phase>/<YYYYMMDD-HHMM>_<script>_<name>/; never reuse an existing folder."""
    base = f"{time.strftime('%Y%m%d-%H%M')}_{script}_{name}"
    path, k = RUNS / phase / base, 2
    while path.exists():
        path, k = RUNS / phase / f"{base}-{k}", k + 1
    path.mkdir(parents=True)
    return path


def find_run(phase: str, run: str) -> Path:
    """A run given as a path, or as a folder name under runs/<phase>/."""
    for path in (Path(run), RUNS / phase / run):
        if run and (path / "run.json").is_file():
            return path
    raise FileNotFoundError(f"run {run!r} not found (looked in {RUNS / phase})")
