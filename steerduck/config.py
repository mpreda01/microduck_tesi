"""Configs, folders and runs (REPO_STRUCTURE.md, sections 2, 3 and 7).

models/, data/ and runs/ go under $STEERDUCK_STORAGE when it is set (e.g. Google Drive on Colab),
otherwise inside the repository. No other module builds these paths.
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import platform
import subprocess
import sys
import time
import zlib
from importlib import metadata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
STORAGE = Path(os.environ.get("STEERDUCK_STORAGE", ROOT))
MODELS, DATA, RUNS = STORAGE / "models", STORAGE / "data", STORAGE / "runs"

log = logging.getLogger("steerduck")


def load_yaml(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, width=120)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def microduck_rl_dir() -> Path:
    """The microduck_rl checkout: $MICRODUCK_RL_DIR, else external/microduck_rl, else ./microduck_rl."""
    env = os.environ.get("MICRODUCK_RL_DIR")
    for path in [Path(env)] if env else [ROOT / "external" / "microduck_rl", ROOT / "microduck_rl"]:
        if path.is_dir():
            return path
    raise FileNotFoundError("microduck_rl not found: clone it to external/microduck_rl or set MICRODUCK_RL_DIR")


def robot_dir() -> Path:
    return microduck_rl_dir() / "src" / "mjlab_microduck" / "robot" / "microduck"


# ---------------------------------------------------------------------------
# Situations
# ---------------------------------------------------------------------------
def load_situations():
    """-> (list of situations, suites). Every block of situations.yaml is expanded into one situation
    per combination of its `sweep` lists, in file order."""
    conf = load_yaml(CONFIGS / "situations.yaml")
    situations = []
    for block in conf["situations"]:
        sweep = block.get("sweep", {})
        for values in itertools.product(*sweep.values()):
            s = {k: v for k, v in block.items() if k != "sweep"} | dict(zip(sweep, values))
            s["family"] = s["name"]   # the block's name template, e.g. face_{deg}deg_{direction}
            s["name"] = s["name"].format(**s)
            if "offset" in s:
                s["offset_m"] = conf["payload_offsets_m"][s["offset"]]
            situations.append(s)
    names = [s["name"] for s in situations]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"situation names must be unique: {duplicates}")
    return situations, conf.get("suites", {})


def select_situations(suite: str = "all", only=None) -> list[dict]:
    """The situations of a suite, optionally only those named in `only`."""
    situations, suites = load_situations()
    if suite == "all":
        chosen = [s["name"] for s in situations]
    elif suite not in suites:
        raise ValueError(f"unknown suite {suite!r}; known: all, {', '.join(suites)}")
    elif isinstance(suites[suite], dict):
        chosen = [s["name"] for s in situations if s["group"] in suites[suite]["groups"]]
    else:
        chosen = suites[suite]
    unknown = (set(chosen) | set(only or [])) - {s["name"] for s in situations}
    if unknown:
        raise ValueError(f"unknown situations: {sorted(unknown)}")
    return [s for s in situations if s["name"] in chosen and (not only or s["name"] in only)]


def situation_seed(name: str, seed: int) -> int:
    """Seed of the terrain and of the episodes of a situation. It depends on the name only, not on the
    position in situations.yaml: editing the lists never changes the episodes of other situations."""
    return zlib.crc32(f"{name}/{seed}".encode())


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def new_run(command: str, name: str) -> Path:
    """Create runs/<YYYYMMDD-HHMM>_<command>_<name>/; an existing folder is never reused."""
    base = f"{time.strftime('%Y%m%d-%H%M')}_{command}_{name}"
    path, k = RUNS / base, 2
    while path.exists():
        path, k = RUNS / f"{base}-{k}", k + 1
    path.mkdir(parents=True)
    return path


def find_run(run: str) -> Path:
    """A run given as a path or as a folder name under runs/."""
    for path in (Path(run), RUNS / run):
        if run and (path / "run.json").is_file():
            return path
    raise FileNotFoundError(f"run {run!r} not found in {RUNS}")


def git_state(repo: Path) -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def write_manifest(run_dir: Path, **info):
    """run.json: code versions, machine, and the given fields (written at the start and at the end)."""
    versions = {}
    for package in ("mujoco", "mjlab", "better-actuator-models", "onnxruntime", "torch", "numpy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    save_json({"run_id": run_dir.name, "command": " ".join(sys.argv), "git": git_state(ROOT),
               "microduck_rl": git_state(microduck_rl_dir()), "python": sys.version.split()[0],
               "host": platform.node(), "cpu_count": os.cpu_count(), "versions": versions,
               "written_at": time.strftime("%Y-%m-%d %H:%M:%S"), **info}, run_dir / "run.json")


def setup_logging(log_file=None):
    """The `steerduck` logger to stdout and, if given, appended to log_file. Also the initializer of the
    worker processes, so that they write to the same file."""
    log.handlers.clear()
    log.setLevel(logging.INFO)
    log.propagate = False
    for handler in [logging.StreamHandler(sys.stdout)] + ([logging.FileHandler(log_file)] if log_file else []):
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
