"""Controllers by name, from configs/policies.yaml. The only place where controllers are assembled.

resolve() runs in the main process (it may download files) and returns a picklable spec;
build_controller() runs in each worker process.
"""
from __future__ import annotations

from pathlib import Path

from steerduck import paths
from steerduck.io import load_config
from steerduck.policies.controller import Controller, OnnxController


def load_registry(path=None) -> dict:
    return load_config(path or paths.CONFIGS / "policies.yaml")


def resolve(name: str, registry: dict | None = None) -> dict:
    """-> {"name", "type", "path"}, downloading the policy to models/base/<name>/ if needed."""
    registry = registry or load_registry()
    if name not in registry:
        raise KeyError(f"controller {name!r} not in configs/policies.yaml; known: {', '.join(registry)}")
    entry = registry[name]
    if "path" in entry:
        path = paths.MODELS / entry["path"]
    elif "hf" in entry:
        repo, filename = entry["hf"]
        path = paths.MODELS / "base" / name / filename
        if not path.exists():
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo, filename, local_dir=str(paths.MODELS / "base" / name))
    else:
        raise ValueError(f"controller {name!r}: give `path` or `hf`")
    if not Path(path).exists():
        raise FileNotFoundError(f"controller {name!r}: {path} not found")
    return {"name": name, "type": entry["type"], "path": str(path), "source": entry.get("hf")}


def build_controller(spec: dict, record_activations: bool = False) -> Controller:
    if spec["type"] == "onnx":
        return OnnxController(spec["name"], spec["path"], record_activations=record_activations)
    raise ValueError(f"unknown controller type {spec['type']!r}")
