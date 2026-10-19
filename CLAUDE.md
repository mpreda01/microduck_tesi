# CLAUDE.md

Master's thesis: closed-loop adaptation of the Microduck's frozen walking policy with steering vectors distilled from specialist policies. The code lives in the `steerduck` package.

## Read first (mandatory)

1. **Before writing or changing any code, read [REPO_STRUCTURE.md](REPO_STRUCTURE.md).** It fixes where every piece of code goes, the Controller interface, the configs, the run and data formats and the naming conventions. New code goes in the file it assigns; if no file fits, decide where the code goes and record it there.
2. **After writing or changing code, update [REPO_STRUCTURE.md](REPO_STRUCTURE.md) when needed:**
   - the tree and the "Stato" paragraph (section 2) when files are added, removed or renamed;
   - the file tables (sections 5-7) when what a file does changes;
   - the config fields (section 4) and the `episodes.parquet` columns (section 8.3) when they change;
   - the migration table (section 11) when a notebook part is migrated.
3. **For any request about how to go on, a phase, an experiment or a method, read [docs/Piano_lavoro_tesi_Preda.md](docs/Piano_lavoro_tesi_Preda.md).** It defines the phases (section 5), with each one's product and criterion to go on, the metrics, and the method (section 3). It is long (about 900 lines): read the sections the request touches, using Grep on its headings.

## Map

| Need | Where |
|---|---|
| Situations of the benchmark, suites | `configs/situations.yaml` (append new entries at the END: the index seeds the episodes) |
| Simulator / episode parameters | `configs/sim.yaml`; one evaluation = `configs/eval/<phase>.yaml` (inherits sim.yaml) |
| Policies / controllers by name | `configs/policies.yaml` → `src/steerduck/policies/registry.py` |
| Simulation loop, any controller | `src/steerduck/sim/rollout.py` (terrains, scene, robot, conditions, metrics, recording, render alongside) |
| Tables and phase criteria | `src/steerduck/analysis/aggregate.py`, called by `scripts/make_report.py` |
| Entry points | `scripts/evaluate.py`, `scripts/preview_situations.py`, `scripts/make_report.py` |
| Folders and run names | `src/steerduck/paths.py` only |
| Old phase 0 code, not migrated yet | `notebooks/fase_0.ipynb`, `notebooks/_build/eval_scenario.py` |
| Upstream robot / training code | `microduck_rl/` (read-only: never modify it; the notebook pins commit `6cd45fc`) |

Do not read the large notebooks (`fase_0.ipynb` is 1.4 MB) whole. To look into one, extract the cells to the scratchpad with a short Python script and read only the cells you need.

## Writing code

- **One place per operation.** Before writing a function, grep `src/steerduck` for an existing one. Scripts stay thin: parse args, load a config, call the library, write a run. Logic needed in two places goes in the library.
- **Everything evaluated is a Controller** (`policies/controller.py`: `reset`, `act`, `activations`, `episode_diagnostics`) and is built only in `registry.py`. Evaluation, data collection and DAgger reuse `sim/rollout.py`; do not write a second simulation loop.
- **Experiments are data:** a new experiment is a YAML file, not new code. Never hard-code paths, seeds or parameters that belong in `configs/`.
- **Results:** a run is a new, immutable folder from `paths.new_run_dir`, with `config.yaml`, `run.json` (from `io.write_manifest`) and `log.txt`. Per-episode results go in `episodes.parquet`, per-step data in `data/steps/<run_id>/` as `.npz` + `fields.json`. Averages go in `analysis/`, never in the evaluation. Nothing in `results/` is written by hand.
- **Style:** match the existing code. That means:
  - `from __future__ import annotations` at the top;
  - English docstrings and comments that say *why*, giving units in names (`_s`, `_m`, `_deg`, `_w`);
  - lines up to 120 characters;
  - the `logging.getLogger("steerduck")` logger instead of print in the library;
  - `ValueError` / `FileNotFoundError` with a clear message for bad input;
  - heavy optional imports (torch, onnx, bam, imageio) inside the function that needs them.
- **Names:** "situation" (not condition), "controller" (not policy) in code and data. Layers are `h1`, `h2`, `h3`. Situation names are snake_case and identical everywhere.
- **Reproducibility:** the draws of an episode come from `np.random.default_rng([seed, situation_index, episode])` in a fixed order (start, battery voltage, joint noise). Changing that order or the seeding changes every result: do it only if the user asks, and say so.
- Create only the files the current task needs, not stubs for later phases.

## Running and checking

- The simulator (mujoco, mjlab 1.3.0, bam) runs **on Colab**, inside microduck_rl's environment: `uv run --project external/microduck_rl --with-editable . python scripts/<script>.py ...` (setup in [README.md](README.md)). **It is not installed on this machine**, so simulation code cannot be run here: say so plainly, and never claim a simulation result you did not obtain.
- Local checks that work:
  - `python -m py_compile` on the changed files;
  - pure-Python logic (configs, situations, metrics, aggregation, make_report on a fake run) with Python 3.12 (`C:\Users\mpred\AppData\Local\Programs\Python\Python312\python.exe`), using a scratchpad venv with numpy/pandas/scipy/pyyaml/pyarrow, `PYTHONPATH=src`, and stub modules for `mujoco` / `mjlab`;
  - set `STEERDUCK_STORAGE` to a scratchpad folder, so that test runs do not land in the repository.

  The default `python` on PATH is an MSYS Python 3.9 without packages.
- Clean up what checks create in the repository (`__pycache__`, `results/tables` of a test).

## Working with the user

- The user writes in Italian. Answer in Italian; documents for the thesis (`docs/`, REPO_STRUCTURE.md) are in Italian; code, comments and configs are in English.
- Edit the project files the user names directly, not copies. Do not rewrite notebooks that hold results or outputs without asking.
- Commit or push only when asked.
