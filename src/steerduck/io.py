"""Configs, run manifests, logging and file formats (REPO_STRUCTURE.md, sections 4.7 and 8)."""
from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import yaml

from steerduck import paths

log = logging.getLogger("steerduck")

TRACKED_PACKAGES = ("mujoco", "mjlab", "better-actuator-models", "onnxruntime", "onnx", "torch", "numpy")


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
def deep_merge(base: dict, override: dict) -> dict:
    """`override` on top of `base`; nested dicts are merged, everything else is replaced."""
    out = dict(base)
    for key, value in override.items():
        out[key] = deep_merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config; `inherit: <file>` (relative to this file) is loaded first and overridden."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        conf = yaml.safe_load(f) or {}
    parent = conf.pop("inherit", None)
    return deep_merge(load_config(path.parent / parent), conf) if parent else conf


def save_yaml(obj, path: str | Path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True, width=120)


def save_json(obj, path: str | Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------
def git_state(repo: Path) -> dict:
    """Commit and uncommitted changes of a git checkout (None if it is not one)."""
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def package_versions(names=TRACKED_PACKAGES) -> dict:
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def write_manifest(run_dir: Path, **info) -> dict:
    """Write run.json: code versions, machine, and the given fields. Called at the start of a run and
    again at the end with the results of the run (status, runtime...)."""
    try:
        microduck_rl = git_state(paths.microduck_rl_dir())
    except FileNotFoundError:
        microduck_rl = None
    manifest = {
        "run_id": run_dir.name,
        "command": " ".join(sys.argv),
        "git": git_state(paths.ROOT),
        "microduck_rl": microduck_rl,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "versions": package_versions(),
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **info,
    }
    save_json(manifest, run_dir / "run.json")
    return manifest


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_file: str | Path | None = None):
    """Messages of the `steerduck` logger to stdout (unbuffered) and, if given, appended to log_file.
    Also used as the initializer of the worker processes, so that they write to the same file."""
    log.handlers.clear()
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter("%(message)s")
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    for h in handlers:
        h.setFormatter(fmt)
        log.addHandler(h)
