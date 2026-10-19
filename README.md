# Microduck: steering vectors distilled from specialist policies

Code of the master's thesis described in [docs/Piano_lavoro_tesi_Preda.md](docs/Piano_lavoro_tesi_Preda.md).
How the repository is organized, and why: [REPO_STRUCTURE.md](REPO_STRUCTURE.md).

## Setup

The simulator is the one of the policy cards (CPU MuJoCo + bam servo model). Its packages (mujoco,
mjlab 1.3.0, bam) come from the environment of microduck_rl, at the same commit as phases 0 and 1:

```bash
git clone https://github.com/pollen-robotics/microduck_rl external/microduck_rl
git -C external/microduck_rl checkout 6cd45fc          # hostile-terrain branch, as phases 0 and 1
curl -LsSf https://astral.sh/uv/install.sh | sh        # if uv is missing
uv sync --project external/microduck_rl
```

Every command then runs in that environment, with this package installed on top:

```bash
uv run --project external/microduck_rl --with-editable . python scripts/<script>.py ...
```

The checkout can live elsewhere: set `MICRODUCK_RL_DIR`. A `microduck_rl/` folder in the repository
root is also found.

### Environment variables

| Variable | Effect |
|---|---|
| `STEERDUCK_STORAGE` | Folder for `models/`, `data/`, `runs/` (e.g. Google Drive on Colab). Default: the repository. |
| `MICRODUCK_RL_DIR` | microduck_rl checkout. Default: `external/microduck_rl`, then `microduck_rl`. |
| `MUJOCO_GL=egl` | Offscreen rendering on a headless machine (videos and preview). |

### Colab

```python
from google.colab import drive
drive.mount("/content/drive")
!git clone https://github.com/mpreda01/microduck_tesi /content/tesi
%cd /content/tesi
!git clone https://github.com/pollen-robotics/microduck_rl external/microduck_rl
!git -C external/microduck_rl checkout --quiet 6cd45fc
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ["PATH"] = "/root/.local/bin:" + os.environ["PATH"]
os.environ["UV_HTTP_TIMEOUT"] = "600"
os.environ["STEERDUCK_STORAGE"] = "/content/drive/MyDrive/microduck_thesis"
# offscreen rendering with the NVIDIA driver (GPU runtime), for videos and preview
icd = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if not os.path.exists(icd):
    os.makedirs(os.path.dirname(icd), exist_ok=True)
    open(icd, "w").write('{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}\n')
os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "egl"
!uv sync --project external/microduck_rl
```

## Phase 1: benchmark of situations

| Step | Command (prefix `uv run --project external/microduck_rl --with-editable .`) |
|---|---|
| Look at the situations | `python scripts/preview_situations.py configs/eval/fase1.yaml` |
| Quick test | `python scripts/evaluate.py configs/eval/fase1.yaml --only flat,friction_0.3 --episodes 2 --no-steps` |
| Full benchmark | `python scripts/evaluate.py configs/eval/fase1.yaml` |
| With rough-walk-g | `python scripts/evaluate.py configs/eval/fase1.yaml --controller alpha_walking,rough_walk_g` |
| Table and criterion | `python scripts/make_report.py fase1 --run <run folder>` |

The situations are in [configs/situations.yaml](configs/situations.yaml), the simulator and episode
parameters in [configs/sim.yaml](configs/sim.yaml), the phase settings in
[configs/eval/fase1.yaml](configs/eval/fase1.yaml).

**Outputs.** `evaluate.py` writes `runs/fase1/<YYYYMMDD-HHMM>_evaluate_<name>/` (`config.yaml`,
`run.json`, `log.txt`, `episodes.parquet`, `videos/`) and the per-step data in `data/steps/<run_id>/`
(`fields.json` + `<controller>/<situation>.npz`). `make_report.py` writes
`results/tables/fase1_summary.csv`. Once a run is the one of the thesis, add its folder name to
[results/runs_used.yaml](results/runs_used.yaml): then `python scripts/make_report.py fase1` alone
regenerates the table.

**Timing** (Colab CPUs, 68 situations x 20 episodes): about 32 minutes per policy without the per-step
data; 1-1.5 hours and about 2 GB per policy with them (about 0.5 GB with `--no-activations`).

## Tests

```bash
uv run --project external/microduck_rl --with-editable ".[test]" pytest tests
```
