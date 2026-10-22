# CLAUDE.md

Master's thesis: closed-loop adaptation of the Microduck's frozen walking policy with steering vectors distilled from specialist policies. The code lives in the `steerduck` package.

## Read first (mandatory)

1. **Before writing or changing any code, read [REPO_STRUCTURE.md](REPO_STRUCTURE.md).** It fixes where every piece of code goes, the Controller interface, the configs, the run and data formats and the naming conventions. New code goes in the file it assigns; if no file fits, decide where the code goes and record it there.
2. **After writing or changing code, update [REPO_STRUCTURE.md](REPO_STRUCTURE.md) when needed:**
   - the tree and the "Stato" paragraph (section 1) when files are added, removed or renamed;
   - the config description (section 3) when YAML keys change;
   - the module and script tables (sections 4-6) when what a file does changes;
   - the formats and the `episodes.parquet` columns (section 7) when they change.
3. **For any request about how to go on, a phase, an experiment or a method, read [docs/Piano_lavoro_tesi_Preda.md](docs/Piano_lavoro_tesi_Preda.md).** It defines the phases (section 5), with each one's product and criterion to go on, the metrics, and the method (section 3). It is long (about 900 lines): read the sections the request touches, using Grep on its headings.

## Map

| Need | Where |
|---|---|
| Situations of the benchmark, suites | `configs/situations.yaml`: blocks with `sweep` lists, expanded by `config.load_situations` |
| Simulator and evaluation parameters | `configs/eval.yaml` (every parameter written there, no hidden defaults) |
| Policies / controllers by name | `configs/policies.yaml` → `steerduck/policy.py` (`resolve`, `build_controller`) |
| Terrains, MuJoCo model, robot, events | `steerduck/scene.py` |
| Simulation loop, metrics, per-step data, video | `steerduck/rollout.py` |
| Tables and phase criteria | `steerduck/report.py`, called by `scripts/report.py` |
| Entry points | `scripts/evaluate.py` (also `--preview`), `scripts/report.py` |
| Folders, runs, seeds | `steerduck/config.py` only |
| Old phase 0 code, not migrated yet | `notebooks/fase_0.ipynb`, `notebooks/_build/eval_scenario.py` |
| Upstream robot / training code | `microduck_rl/` (read-only: never modify it; the notebook pins commit `6cd45fc`) |

Do not read the large notebooks (`fase_0.ipynb` is 1.4 MB) whole. To look into one, extract the cells to the scratchpad with a short Python script and read only the cells you need.

## Writing code

- **One place per operation.** Before writing a function, grep `steerduck/` for an existing one. Scripts stay thin: parse args, load a config, call the library, write a run. Logic needed in two places goes in the library.
- **Write only the code that is needed, readable for a human review:** short functions with a docstring that says what and why, no speculative options, no stubs for later phases.
- **Everything evaluated is a Controller** (`policy.py`: `reset`, `act`, `activations`, `episode_diagnostics`) and is built only in `build_controller`. Evaluation, data collection and DAgger reuse `rollout.py`; do not write a second simulation loop.
- **Experiments are data:** a new experiment is a change in `configs/`, not new code. Values the user may want to vary go in the YAML as plain lists (e.g. a `sweep` list in situations.yaml), not in the code. Never hard-code paths, seeds or parameters.
- **Results:** a run is a new, immutable folder from `config.new_run`, with `config.yaml`, `run.json` (`config.write_manifest`) and `log.txt`. Per-episode results go in `episodes.parquet`, per-step data in `data/<run_id>/` as `.npz` + `fields.json`. Averages go in `report.py`, never in the evaluation. Nothing in `results/` is written by hand.
- **Style:** match the existing code. That means:
  - `from __future__ import annotations` at the top;
  - English docstrings and comments that say *why*, giving units in names (`_s`, `_m`, `_deg`, `_w`);
  - lines up to 120 characters;
  - the `logging.getLogger("steerduck")` logger instead of print in the library;
  - `ValueError` / `FileNotFoundError` with a clear message for bad input;
  - heavy optional imports (torch, onnx, bam, imageio) inside the function that needs them.
- **Names:** "situation" (not condition), "controller" (not policy) in code and data. Layers are `h1`, `h2`, `h3`. Situation names are snake_case and identical everywhere.
- **Reproducibility:** terrain and episodes of a situation are seeded by `config.situation_seed(name, seed)`, from the name, so editing the YAML lists never changes other situations. The draws of an episode come from `np.random.default_rng([situation_seed, episode])` in a fixed order (start, battery voltage, joint noise). Changing that order or the seeding changes every result: do it only if the user asks, and say so.

## Running and checking

- The simulator (mujoco, mjlab 1.3.0, bam, rsl_rl) is **installed locally**, in microduck_rl's environment `microduck_rl/.venv`, and simulations already run on this machine. Run the scripts with `uv run --project microduck_rl --with-editable . python scripts/<script>.py ...` (on Colab: `--project external/microduck_rl`, setup in [README.md](README.md)). Never claim a simulation result you did not obtain.
- Local checks:
  - `python -m py_compile` on the changed files;
  - short real runs (few situations and episodes, `--no-video`) with `STEERDUCK_STORAGE` set to a scratchpad folder, so that test runs do not land in the repository's `runs/` and `data/`.

  The default `python` on PATH is an MSYS Python 3.9 without packages: do not use it.
- Clean up what checks create in the repository (`__pycache__` of the new code, `results/tables` of a test). Never delete files tracked by git while cleaning up (`notebooks/_build/__pycache__` is tracked).

## Working with the user

- The user writes in Italian. Answer in Italian; documents for the thesis (`docs/`, REPO_STRUCTURE.md) are in Italian; code, comments and configs are in English.
- Edit the project files the user names directly, not copies. Do not rewrite notebooks that hold results or outputs without asking.
- Commit or push only when asked.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.
